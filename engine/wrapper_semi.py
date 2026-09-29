"""PyTorch Lightning implementation of the core Semi-MedRef framework.

This release contains the two architectures used as the primary models in the
paper: MMI-UNet and GuideDecoder.  The module implements the EMA
teacher--student objective (Eqs. (2)--(5)), T-PatchMix (Eqs. (6)--(8)),
and position-guided PACL (Eq. (9)).  PosAug is applied by the data pipeline.
"""

import math
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from monai.losses import DiceCELoss
from PIL import Image, ImageDraw, ImageFont
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex

from utils.model import MMIUNet_V2, LanGuideMedSeg
from utils.pos_aug import (
    sample_block_mask,
    span_mix,
    span_mix_dual,
    tokenize_text_list,
    extract_pos_phrase,
    pos_phrase_to_region,
    sample_block_mask_pos,
    sample_block_mask_prob,
)


def _to_uint8(img01: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor in [0, 1] to uint8."""
    img01 = img01.clamp(0, 1)
    return (img01 * 255.0).byte()


def _colorize_gray(g01: torch.Tensor) -> torch.Tensor:
    """Expand a single-channel visualization to RGB."""
    return g01.expand(3, -1, -1)


def _put_caption(canvas: Image.Image, text: str, xy=(4, 4)):
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.rectangle(
        [xy[0] - 2, xy[1] - 2, xy[0] + 2 + 7 * len(text), xy[1] + 12],
        fill=(0, 0, 0, 160),
    )
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def _save_xpatchmix_panel(
    step: int,
    img_s_bchw,
    img_s_perm_bchw,
    M_b1hw,
    img_mix_bchw,
    t_prob_i_b1hw,
    t_prob_j_b1hw,
    y_mix_b1hw,
    text_i: str,
    text_j: str,
    text_mix: str,
    out_dir: str,
    idx: int,
):
    """Save an optional seven-column T-PatchMix diagnostic panel."""

    def _norm01(x: torch.Tensor) -> torch.Tensor:
        if x.dtype.is_floating_point:
            return x.clamp(0, 1)
        return x

    def _to_rgb_u8(x_chw: torch.Tensor) -> torch.Tensor:
        """Return a uint8 RGB tensor accepted by ``make_grid``."""
        x = x_chw
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = _norm01(x)
        x = _to_uint8(x)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] > 3:
            x = x[:3]
        return x

    os.makedirs(out_dir, exist_ok=True)
    i = idx

    img_i = img_s_bchw[i].detach().cpu()
    img_j = img_s_perm_bchw[i].detach().cpu()
    mask_i = M_b1hw[i].detach().cpu()
    mix = img_mix_bchw[i].detach().cpu()
    prob_i = t_prob_i_b1hw[i].detach().cpu()
    prob_j = t_prob_j_b1hw[i].detach().cpu()
    target_mix = y_mix_b1hw[i].detach().cpu()

    img_i_u8 = _to_rgb_u8(img_i)
    img_j_u8 = _to_rgb_u8(img_j)
    mix_u8 = _to_rgb_u8(mix)
    mask_viz = _to_uint8(_colorize_gray(_norm01(mask_i)))
    prob_i_viz = _to_uint8(_colorize_gray(_norm01(prob_i)))
    prob_j_viz = _to_uint8(_colorize_gray(_norm01(prob_j)))
    target_viz = _to_uint8(_colorize_gray(_norm01(target_mix)))

    panel = torch.stack(
        [img_i_u8, img_j_u8, mask_viz, mix_u8, prob_i_viz, prob_j_viz, target_viz]
    )
    grid = vutils.make_grid(panel, nrow=7, padding=8)

    grid_pil = TF.to_pil_image(grid)
    _put_caption(grid_pil, f"step={step} | sample={i}", (6, 6))

    caption = f"Text i: {text_i}\nText j: {text_j}\nText mix: {text_mix}"
    caption_lines = caption.count("\n") + 1
    _put_caption(grid_pil, caption, (6, grid_pil.height - 36 * caption_lines))

    out_path = os.path.join(out_dir, f"step{step:07d}_idx{i}.png")
    grid_pil.save(out_path)
class MMIUNet_SemiWrapper(pl.LightningModule):
    """Core Semi-MedRef optimization module.

    The supervised branch uses Dice--cross-entropy.  After burn-in, an EMA
    teacher predicts the weak view and the student learns from the aligned
    strong view.  PosAug is supplied by the dataset, while this module applies
    T-PatchMix (Eqs. (6)--(8)) and the position-guided PACL objective (Eq. (9)).
    """

    def __init__(
        self,
        bert_type: str,
        vision_type: str,
        project_dim: int,
        lr: float = 3e-4,
        ema_decay: float = 0.999,
        burn_in_epochs: int = 5,
        unsup_weight: float = 1.0,
        conf_th: float = 0.6,
        load_convnext_ckpt: str = "./pretrained/convnext_tiny_22k_224.pth",
        unsup_rampup_epochs: int = 15,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 0.999,
        ema_warmup_epochs: int = 20,
        use_xpatchmix: bool = True,
        mix_block: int = 64,
        mix_prob: float = 1.0,
        xpatchmix_mode: str = "pos",  # random | pos | prob
        mix_margin: float = 0.0,
        viz_every: int = 0,
        viz_dir: str = "./outputs/visualizations",
        enable_itc: bool = False,
        itc_weight: float = 0.05,
        itc_tau: float = 0.07,
        itc_w_unsup: float = 0.01,
        pseudo_threshold_mode: str = "hard",
        pseudo_threshold_temp: float = 0.05,
    ):
        super().__init__()

        self.enable_itc = bool(enable_itc) and (float(itc_weight) > 0.0 or float(itc_w_unsup) > 0.0)
        self.itc_weight = float(itc_weight)
        self.itc_tau = itc_tau
        self.itc_w_unsup = itc_w_unsup
        self.student = MMIUNet_V2(bert_type, vision_type, project_dim, enable_project_head=self.enable_itc)

        if load_convnext_ckpt and os.path.exists(load_convnext_ckpt):
            sd = self.student.state_dict()
            pre = torch.load(load_convnext_ckpt, map_location="cpu")
            if isinstance(pre, dict) and "model" in pre:
                pre = pre["model"]
            matched = {k: v for k, v in pre.items() if k in sd and sd[k].shape == v.shape}
            sd.update(matched)
            self.student.load_state_dict(sd)
            print(f"[MMIUNet_Semi] Load convnext weights: {len(matched)} tensors matched")

        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.sup_criterion = DiceCELoss(sigmoid=True)
        self.unsup_criterion = DiceCELoss(sigmoid=True)

        self.lr = lr
        self.ema_decay = ema_decay
        self.burn_in_epochs = burn_in_epochs
        self.unsup_weight = unsup_weight
        self.conf_th = conf_th
        self.pseudo_threshold_mode = str(pseudo_threshold_mode).lower()
        if self.pseudo_threshold_mode not in ("hard", "soft"):
            raise ValueError(f"pseudo_threshold_mode must be 'hard' or 'soft', got {pseudo_threshold_mode}")
        self.pseudo_threshold_temp = max(float(pseudo_threshold_temp), 1e-6)
        self.unsup_rampup_epochs = unsup_rampup_epochs
        self.ema_decay_start = ema_decay_start
        self.ema_decay_end = ema_decay_end
        self.ema_warmup_epochs = ema_warmup_epochs


        self.use_xpatchmix = bool(use_xpatchmix)
        self.mix_block = int(mix_block)
        self.mix_prob = float(mix_prob)
        self.xpatchmix_mode = str(xpatchmix_mode).lower()
        self.mix_margin = float(mix_margin)
        self.bert_type = bert_type
        self._last_xpatchmix_rate = torch.tensor(0.0)

        metrics_dict = {
            "acc": Accuracy(task="binary"),
            "dice": Dice(),
            "MIoU": BinaryJaccardIndex(),
        }
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        self.save_hyperparameters(ignore=["load_convnext_ckpt"])
        self.viz_every = int(viz_every)
        self.viz_dir = viz_dir
        if self.viz_every > 0:
            os.makedirs(self.viz_dir, exist_ok=True)

    def _itc_loss(self, img_proj: torch.Tensor, txt_proj: torch.Tensor, tau: float, pseudo_label) -> torch.Tensor:
        """Compute the bidirectional Jaccard-weighted PACL loss in Eq. (9)."""
        img = F.normalize(img_proj, dim=1)
        txt = F.normalize(txt_proj, dim=1)

        logits = img @ txt.t() / tau
        y = pseudo_label.to(img_proj.device).float()

        inter = (y[:, None, :] * y[None, :, :]).sum(-1)
        union = ((y[:, None, :] + y[None, :, :]) > 0).float().sum(-1)
        union = torch.where(union > 0, union, torch.ones_like(union))
        P = inter / union
        P.fill_diagonal_(1.0)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss_i2t = -(P * log_prob).sum(1) / (P.sum(1) + 1e-6)

        log_prob_t = logits.t() - torch.logsumexp(logits.t(), dim=1, keepdim=True)
        P_t = P.t()
        loss_t2i = -(P_t * log_prob_t).sum(1) / (P_t.sum(1) + 1e-6)

        return 0.5 * (loss_i2t.mean() + loss_t2i.mean())

    def _threshold_target(self, prob: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Hard:  y = 1[p >= threshold]
        Soft:  y = sigmoid((p - threshold) / temp)
        """
        if self.pseudo_threshold_mode == "soft":
            return torch.sigmoid((prob - float(threshold)) / self.pseudo_threshold_temp)
        return (prob >= float(threshold)).float()

    def _align_teacher_prob(self, t_prob: torch.Tensor, geom: dict) -> torch.Tensor:
        """Align weak-view teacher probabilities with per-sample geometry."""
        B, _, H, W = t_prob.shape
        g = geom or {}
        h = g.get("hflip", False)
        v = g.get("vflip", False)

        # Normalize scalar or batched flags to BoolTensor[B].
        if not torch.is_tensor(h):
            h = torch.full((B,), bool(h), dtype=torch.bool, device=t_prob.device)
        else:
            h = h.to(t_prob.device).bool()
        if not torch.is_tensor(v):
            v = torch.full((B,), bool(v), dtype=torch.bool, device=t_prob.device)
        else:
            v = v.to(t_prob.device).bool()

        # Per-sample affine scales for horizontal and vertical flips.
        sx = torch.where(h, torch.full((B,), -1.0, device=t_prob.device),
                         torch.full((B,), 1.0, device=t_prob.device))
        sy = torch.where(v, torch.full((B,), -1.0, device=t_prob.device),
                         torch.full((B,), 1.0, device=t_prob.device))

        # Affine matrices have shape [B, 2, 3].
        theta = torch.zeros((B, 2, 3), device=t_prob.device, dtype=torch.float32)
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy

        grid = F.affine_grid(theta, size=(B, 1, H, W), align_corners=False)
        return F.grid_sample(t_prob, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _build_auxiliary_xpatchmix_target(
        self,
        img_w: torch.Tensor,
        text_w: dict,
        geom: dict,
        perm: torch.Tensor,
        mix_mask: torch.Tensor,
    ):
        """Optional training-only target transformed by the same T-PatchMix.

        Subclasses may return an additional ``(B, C, H, W)`` target.  It is
        generated from the unmixed weak pairs and transformed with exactly the
        same geometry, permutation, and accepted patch mask as the EMA target.
        The base Semi-MedRef model has no auxiliary target.
        """
        return None

    def _build_xpatchmix_batch(self, u: dict):
        """Construct aligned image, pseudo-mask, and report views (Eqs. (6)--(8))."""
        img_w = u["img_w"]
        img_s = u["img_s"]
        text_w = u["text_w"]
        text_s_str = u["text_s_str"]
        geom = u.get("geom", {"hflip": False, "vflip": False})

        B, C, H, W = img_s.shape
        device = img_s.device

        def _permute_batch(x, idx: torch.Tensor):
            if torch.is_tensor(x):
                return x.index_select(0, idx)
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    out[k] = v.index_select(0, idx) if torch.is_tensor(v) else v
                return out
            return x

        # Pair each sample with another item in the same batch.
        perm = torch.randperm(B, device=device)

        # Predict both weak views with the teacher and align their geometry.
        with torch.no_grad():
            self.teacher.eval()

            t_logits_i = self.teacher([img_w, text_w])
            t_prob_i = self._to_binary_probs(t_logits_i)
            t_prob_i = self._align_teacher_prob(t_prob_i, geom)

            img_w_perm = img_w.index_select(0, perm)
            text_w_perm = _permute_batch(text_w, perm)

            if isinstance(geom.get("hflip"), torch.Tensor) or isinstance(geom.get("vflip"), torch.Tensor):
                geom_perm = {
                    "hflip": geom["hflip"].index_select(0, perm) if torch.is_tensor(geom.get("hflip")) else geom.get(
                        "hflip", False),
                    "vflip": geom["vflip"].index_select(0, perm) if torch.is_tensor(geom.get("vflip")) else geom.get(
                        "vflip", False),
                }
            else:
                geom_perm = geom

            t_logits_j = self.teacher([img_w_perm, text_w_perm])
            t_prob_j = self._to_binary_probs(t_logits_j)
            t_prob_j = self._align_teacher_prob(t_prob_j, geom_perm)

        # Restrict the sampled patch to anatomically compatible regions.
        if self.xpatchmix_mode == "pos":
            pos_list = [extract_pos_phrase(s) for s in text_s_str]
            regions = []
            for i in range(B):
                pa = pos_list[i]
                pb = pos_list[perm[i].item()]
                if pa is None:
                    regions.append(None)
                    continue
                if pb is not None:
                    if ("left" in pa and "right" in pb) or ("right" in pa and "left" in pb):
                        regions.append(None)
                        continue
                regions.append(pos_phrase_to_region(pa, H, W, margin=self.mix_margin))
            M = sample_block_mask_pos(B, H, W, regions, block=self.mix_block, p=self.mix_prob, device=device)
        elif self.xpatchmix_mode == "prob":
            M = sample_block_mask_prob(t_prob_j, block=self.mix_block, p=self.mix_prob, thresh=self.conf_th)
        else:
            M = sample_block_mask(
                B, H, W, block=self.mix_block, p=self.mix_prob, device=device
            )

        # Apply the same accepted patch to image and teacher target.
        img_mix = img_s * (1.0 - M) + img_s.index_select(0, perm) * M

        # Reject patches without sufficiently confident foreground evidence.
        with torch.no_grad():
            eps = 1e-6
            area = M.sum(dim=(2, 3), keepdim=True) + eps
            lesion_frac = (t_prob_j * M).sum(dim=(2, 3), keepdim=True) / area
            gate = self._threshold_target(lesion_frac, self.conf_th)
            M = M * gate
            self._last_xpatchmix_rate = (M.sum(dim=(1, 2, 3)) > 0).float().mean().detach()

        y_mix_prob = t_prob_i * (1.0 - M) + t_prob_j * M
        y_mix = self._threshold_target(y_mix_prob, 0.5)

        # Synchronize the position phrase with the accepted image patch.
        if self.xpatchmix_mode == "pos":
            texts_mix = []
            for i in range(B):
                a = text_s_str[i]
                b = text_s_str[perm[i].item()]
                if M[i].sum().item() <= 0:
                    texts_mix.append(a)
                else:
                    texts_mix.append(span_mix_dual(a, b))
        elif self.xpatchmix_mode == "prob":
            texts_mix = list(text_s_str)
        else:
            texts_mix = []
            for i in range(B):
                a = text_s_str[i]
                b = text_s_str[perm[i].item()]
                if M[i].sum().item() <= 0:
                    texts_mix.append(a)
                else:
                    texts_mix.append(span_mix(a, b))

        text_mix_tok = tokenize_text_list(texts_mix, tokenizer_name=self.bert_type, max_len=24, device=device)

        # Optional qualitative diagnostics; disabled in released configs.
        if (self.viz_every > 0
                and (self.global_step % self.viz_every == 0)
                and (self.current_epoch >= self.burn_in_epochs)):
            try:
                img_s_perm = img_s[perm]
                idx = 0
                _save_xpatchmix_panel(
                    step=int(self.global_step),
                    img_s_bchw=img_s,
                    img_s_perm_bchw=img_s_perm,
                    M_b1hw=M,
                    img_mix_bchw=img_mix,
                    t_prob_i_b1hw=t_prob_i,
                    t_prob_j_b1hw=t_prob_j,
                    y_mix_b1hw=y_mix,
                    text_i=text_s_str[idx],
                    text_j=text_s_str[perm[idx].item()],
                    text_mix=texts_mix[idx],
                    out_dir=self.viz_dir,
                    idx=idx
                )
            except Exception as error:
                self.print(f"[visualization skipped] {error!r}")

        # Optional training-only guidance must follow the supervision
        # transformation; do not predict it anew from the already mixed view.
        auxiliary_target = self._build_auxiliary_xpatchmix_target(
            img_w=img_w,
            text_w=text_w,
            geom=geom,
            perm=perm,
            mix_mask=M,
        )
        if auxiliary_target is not None:
            if auxiliary_target.ndim != 4 or auxiliary_target.shape[0] != B:
                raise ValueError(
                    "auxiliary T-PatchMix target must have shape (B,C,H,W), "
                    f"got {tuple(auxiliary_target.shape)}"
                )
            y_mix = torch.cat([y_mix, auxiliary_target.to(y_mix.dtype)], dim=1)

        # Emit a one-time shape check at the first optimization step.
        try:
            sanity_checking = bool(self.trainer.sanity_checking)
        except RuntimeError:
            sanity_checking = False
        if self.global_step == 0 and not sanity_checking:
            msg = f"[xpatchmix] img_mix {img_mix.shape}, y_mix {y_mix.shape}, text_ids {text_mix_tok['input_ids'].shape}"
            try:
                self.print(msg)
            except RuntimeError:
                print(msg)

        return img_mix, y_mix, text_mix_tok

    def _unsup_w_now(self) -> float:
        e = self.current_epoch
        if e < self.burn_in_epochs:
            return 0.0
        if self.unsup_rampup_epochs <= 0:
            return float(self.unsup_weight)
        t = (e - self.burn_in_epochs) / float(self.unsup_rampup_epochs)
        t = max(0.0, min(1.0, t))
        return float(self.unsup_weight) * t

    def _ema_m_now(self) -> float:
        # Start the EMA momentum warm-up after burn-in.
        e = max(0, self.current_epoch - self.burn_in_epochs)
        if self.ema_warmup_epochs <= 0:
            return float(self.ema_decay)
        t = max(0.0, min(1.0, e / float(self.ema_warmup_epochs)))
        return self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * t

    def on_train_start(self):
        # Lightning calls train() recursively; keep the EMA teacher in eval mode.
        self.teacher.eval()

    def _to_binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert segmentation logits to foreground probabilities."""
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 1:2]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.student.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch}

    def training_step(self, batch, batch_idx):
        """Optimize labeled segmentation and unlabeled consistency losses."""
        (x_l, y_l) = batch["labeled"]
        u = batch["unlabeled"]

        img_w = u["img_w"]
        img_s = u["img_s"]
        text_w = u["text_w"]
        text_s = u["text_s"]
        geom = u["geom"] if "geom" in u else {"hflip": False, "vflip": False}

        B = y_l.size(0)

        # Labeled Dice--cross-entropy (Eq. (2)) plus labeled PACL (Eq. (9)).
        if self.enable_itc and (self.itc_weight > 0):
            logits_l, imgp_l, txtp_l = self.student(x_l, return_project=True)
            sup_loss = self.sup_criterion(logits_l, y_l)
            pseudo_label_l = x_l[1].get("pseudo_label", None)
            if pseudo_label_l is None:
                if not getattr(self, "_warned_no_plabel", False):
                    print("[warn] pseudo_label missing in labeled batch; ITC(labeled) disabled.")
                    self._warned_no_plabel = True
                itc_loss_l = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)
            else:
                itc_loss_l = self._itc_loss(imgp_l, txtp_l, tau=self.itc_tau, pseudo_label=pseudo_label_l)
            sup_total = sup_loss + self.itc_weight * itc_loss_l
        else:
            logits_l = self.student(x_l)
            sup_loss = self.sup_criterion(logits_l, y_l)
            itc_loss_l = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)
            sup_total = sup_loss

        w_unsup = float(self._unsup_w_now())

        # Unlabeled consistency (Eqs. (3)--(4)) and PACL (Eq. (9)) start after burn-in.
        if w_unsup > 0.0:
            if self.use_xpatchmix:
                img_mix, y_mix, text_mix_tok = self._build_xpatchmix_batch(u)
                s_logits = self.student([img_mix, text_mix_tok])
                unsup_loss = self.unsup_criterion(s_logits, y_mix)

                if self.enable_itc and (self.itc_w_unsup > 0.0):
                    _, imgp_u, txtp_u = self.student([img_w, text_w], return_project=True)
                    pseudo_label_u = text_w.get("pseudo_label", None)
                    if pseudo_label_u is None:
                        if not getattr(self, "_warned_no_plabel_u", False):
                            print("[warn] pseudo_label missing in unlabeled batch; ITC(unlabeled) disabled.")
                            self._warned_no_plabel_u = True
                        itc_loss_u = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)
                    else:
                        itc_loss_u = self._itc_loss(imgp_u, txtp_u, tau=self.itc_tau, pseudo_label=pseudo_label_u)
                else:
                    itc_loss_u = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)

                total = sup_total + w_unsup * (unsup_loss + self.itc_w_unsup * itc_loss_u)
            else:
                with torch.no_grad():
                    self.teacher.eval()
                    t_logits_w = self.teacher([img_w, text_w])
                    t_prob_w = self._to_binary_probs(t_logits_w)
                    pseudo = self._threshold_target(t_prob_w, self.conf_th)

                s_logits_s = self.student([img_s, text_s])
                unsup_loss = self.unsup_criterion(s_logits_s, pseudo)

                if self.enable_itc and (self.itc_w_unsup > 0.0):
                    _, imgp_u, txtp_u = self.student([img_w, text_w], return_project=True)
                    pseudo_label_u = text_w.get("pseudo_label", None)
                    if pseudo_label_u is None:
                        if not getattr(self, "_warned_no_plabel_u", False):
                            print("[warn] pseudo_label missing in unlabeled batch; ITC(unlabeled) disabled.")
                            self._warned_no_plabel_u = True
                        itc_loss_u = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)
                    else:
                        itc_loss_u = self._itc_loss(imgp_u, txtp_u, tau=self.itc_tau, pseudo_label=pseudo_label_u)
                else:
                    itc_loss_u = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)

                total = sup_total + w_unsup * (unsup_loss + self.itc_w_unsup * itc_loss_u)
        else:
            unsup_loss = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)
            itc_loss_u = torch.zeros(1, device=self.device, dtype=sup_loss.dtype)
            total = sup_total

        # Optimization diagnostics recorded by the experiment logger.
        self.log_dict(
            {
                "train_sup_loss": sup_loss,
                "train_unsup_loss": unsup_loss,
                "train_total_loss": total,
                "w_unsup": w_unsup,
                "conf_th": self.conf_th,
                "use_xpatchmix": float(self.use_xpatchmix),
                "enable_itc": float(self.enable_itc),
                "itc_loss_l": itc_loss_l,
                "itc_loss_u": itc_loss_u,
                "itc_weight": self.itc_weight,
                "itc_tau": self.itc_tau,
                "pseudo_threshold_soft": float(self.pseudo_threshold_mode == "soft"),
                "pseudo_threshold_temp": self.pseudo_threshold_temp,

            },
            on_step=True, on_epoch=True, prog_bar=True, batch_size=B
        )
        if self.use_xpatchmix and w_unsup > 0.0:
            mix_rate = getattr(self, "_last_xpatchmix_rate", torch.tensor(0.0, device=self.device))
            if not torch.is_tensor(mix_rate):
                mix_rate = torch.tensor(float(mix_rate), device=self.device)
            self.log("xpatchmix_rate", mix_rate, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)

        # EMA and augmentation diagnostics.
        self.log("ema_m_now", float(self._ema_m_now()),
                 on_step=True, on_epoch=False, prog_bar=False)

        h = geom.get("hflip", False)
        v = geom.get("vflip", False)
        if torch.is_tensor(h):
            h_rate = h.float().mean()
        else:
            h_rate = torch.tensor(float(bool(h)), device=self.device)
        if torch.is_tensor(v):
            v_rate = v.float().mean()
        else:
            v_rate = torch.tensor(float(bool(v)), device=self.device)

        self.log("hflip_rate", h_rate, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("vflip_rate", v_rate, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)

        # Fraction whose student report differs from the teacher report.
        ids_s = text_s["input_ids"]
        ids_w = text_w["input_ids"]
        if ids_s.dim() == 2 and ids_w.dim() == 2 and ids_s.shape == ids_w.shape:
            pos_aug_hit = (ids_s != ids_w).any(dim=1).float().mean()
            self.log("pos_aug_hit_rate", pos_aug_hit, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)

        # Training metrics are computed on the labeled branch only.
        preds_for_metric = logits_l.detach()
        return {"loss": total, "preds": preds_for_metric, "y": y_l.detach()}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student(x)
        loss = self.sup_criterion(logits, y)

        # Log a Python scalar to avoid MetaTensor formatting issues.
        val_loss_scalar = float(loss.detach().mean().item())
        self.log("val_loss", val_loss_scalar, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=y.size(0))

        return {"val_loss": loss.detach(), "preds": logits.detach(), "y": y.detach()}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student(x)
        loss = self.sup_criterion(logits, y)
        return {"test_loss": loss.detach(), "preds": logits.detach(), "y": y.detach()}

    def _shared_step_end(self, outputs, stage: str):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics
        )
        probs = self._to_binary_probs(outputs["preds"])
        y = outputs["y"]
        for name in metrics:
            v = metrics[name](probs, y).item()
            if stage == "train":
                self.log(name, v, prog_bar=True)
        base = outputs.get("loss", outputs.get(f"{stage}_loss"))
        if not torch.is_tensor(base):
            base = torch.tensor(base, device=self.device)
        return base.mean()

    def training_step_end(self, outputs):
        return {'loss': self._shared_step_end(outputs, "train")}

    def validation_step_end(self, outputs):
        return {'val_loss': self._shared_step_end(outputs, "val")}

    def test_step_end(self, outputs):
        return {'test_loss': self._shared_step_end(outputs, "test")}

    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        epoch = self.trainer.current_epoch

        # Aggregate loss and streaming segmentation metrics.
        key = (stage + "_loss").replace('train_', '')
        stage_loss = torch.mean(torch.stack([
            t[key] if torch.is_tensor(t[key]) else torch.tensor(t[key], device=self.device)
            for t in outputs
        ])).item()
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric

        # Include optimization diagnostics in the training history.
        if stage == "train":
            cm = self.trainer.callback_metrics

            def _get(k, default=None):
                v = cm.get(k, default)
                try:
                    return float(v)
                except Exception:
                    return default if default is not None else float('nan')

            dic["w_unsup"] = _get("w_unsup", -1.0)
            dic["use_xpatchmix"] = _get("use_xpatchmix", float(self.use_xpatchmix))
            dic["train_sup_loss"] = _get("train_sup_loss")
            dic["train_unsup_loss"] = _get("train_unsup_loss")
            dic["train_total_loss"] = _get("train_total_loss")

        if stage != 'test':
            if not hasattr(self, "history"):
                self.history = {}
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print("\n" + "="*80)
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger=True)

    def on_before_zero_grad(self, optimizer):
        """Update teacher parameters and buffers with dynamic EMA momentum."""
        d = float(self._ema_m_now())
        with torch.no_grad():
            s_params = dict(self.student.named_parameters())
            t_params = dict(self.teacher.named_parameters())
            for name, t_p in t_params.items():
                s_p = s_params[name]
                if t_p.dtype.is_floating_point:
                    t_p.data.mul_(d).add_(s_p.data, alpha=1.0 - d)
                else:
                    t_p.data.copy_(s_p.data)

            s_bufs = dict(self.student.named_buffers())
            t_bufs = dict(self.teacher.named_buffers())
            for name, t_b in t_bufs.items():
                s_b = s_bufs[name]
                if t_b.dtype.is_floating_point:
                    t_b.data.mul_(d).add_(s_b.data, alpha=1.0 - d)
                else:
                    t_b.data.copy_(s_b.data)


class LanGuideMedSeg_SemiWrapper(MMIUNet_SemiWrapper):
    """
    Semi-supervised wrapper for LanGuideMedSeg with PACL projections.
    """

    def __init__(
        self,
        bert_type: str,
        vision_type: str,
        project_dim: int,
        lr: float = 3e-4,
        ema_decay: float = 0.999,
        burn_in_epochs: int = 5,
        unsup_weight: float = 1.0,
        conf_th: float = 0.6,
        load_convnext_ckpt: str = "",
        unsup_rampup_epochs: int = 15,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 0.999,
        ema_warmup_epochs: int = 20,
        use_xpatchmix: bool = True,
        mix_block: int = 64,
        mix_prob: float = 1.0,
        xpatchmix_mode: str = "pos",
        mix_margin: float = 0.0,
        viz_every: int = 0,
        viz_dir: str = "./outputs/visualizations",
        enable_itc: bool = False,
        itc_weight: float = 0.0,
        itc_tau: float = 0.07,
        itc_w_unsup: float = 0.0,
        pseudo_threshold_mode: str = "hard",
        pseudo_threshold_temp: float = 0.05,
    ):
        super().__init__(
            bert_type=bert_type,
            vision_type=vision_type,
            project_dim=project_dim,
            lr=lr,
            ema_decay=ema_decay,
            burn_in_epochs=burn_in_epochs,
            unsup_weight=unsup_weight,
            conf_th=conf_th,
            load_convnext_ckpt="",
            unsup_rampup_epochs=unsup_rampup_epochs,
            ema_decay_start=ema_decay_start,
            ema_decay_end=ema_decay_end,
            ema_warmup_epochs=ema_warmup_epochs,
            use_xpatchmix=use_xpatchmix,
            mix_block=mix_block,
            mix_prob=mix_prob,
            xpatchmix_mode=xpatchmix_mode,
            mix_margin=mix_margin,
            viz_every=viz_every,
            viz_dir=viz_dir,
            enable_itc=enable_itc,
            itc_weight=itc_weight,
            itc_tau=itc_tau,
            itc_w_unsup=itc_w_unsup,
            pseudo_threshold_mode=pseudo_threshold_mode,
            pseudo_threshold_temp=pseudo_threshold_temp,
        )

        # Replace student/teacher with LanGuideMedSeg
        self.student = LanGuideMedSeg(bert_type, vision_type, project_dim)
        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # LanGuideMedSeg already applies sigmoid in forward
        self.sup_criterion = DiceCELoss(sigmoid=False)
        self.unsup_criterion = DiceCELoss(sigmoid=False)

    def _to_binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # LanGuideMedSeg outputs probabilities in [0, 1].
        if logits.shape[1] == 1:
            if logits.min() >= 0 and logits.max() <= 1:
                return logits
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 1:2]
