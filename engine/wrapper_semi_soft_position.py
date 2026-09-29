"""ISPG implementation for the image-derived soft token in Eq. (10)."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from engine.wrapper_semi import MMIUNet_SemiWrapper
from engine.wrapper_semi_position import (
    PositionGuidedSemiWrapper,
    complete_position_reports,
    remove_position_expressions,
)
from position_predictor_models import load_position_checkpoint
from utils.model_soft_position import MMIUNetSoftPosition
from utils.pos_aug import tokenize_text_list


class SoftPositionSemiWrapper(PositionGuidedSemiWrapper):
    """Two Eq. (10) routes sharing one segmentation network.

    Text route:
        full report + report-derived six-dimensional position token.

    Image route:
        position-stripped report + differentiable image-predicted soft token.

    Report-derived labels supervise the image position predictor on both the
    mask-labeled and mask-unlabeled subsets.  No segmentation mask is used by
    the position loss on the unlabeled subset.
    """

    def __init__(
        self,
        *args,
        position_init_checkpoint: str = "",
        image_route_start_probability: float = 0.25,
        image_route_max_probability: float = 0.80,
        image_route_ramp_epochs: int = 15,
        position_bce_mix: float = 0.5,
        position_gamma_negative: float = 2.0,
        position_gamma_positive: float = 0.0,
        position_cardinality_weight: float = 0.1,
        validation_text_mode: str = "soft_image",
        segmentation_arch: str = "mmiunet",
        **kwargs,
    ):
        """Replace the base student with an ISPG-capable segmentation model."""
        super().__init__(*args, validation_text_mode=validation_text_mode, **kwargs)
        old_state = self.student.state_dict()
        soft_student = MMIUNetSoftPosition(
            self.hparams.bert_type,
            self.hparams.vision_type,
            self.hparams.project_dim,
            enable_project_head=self.enable_itc,
        )
        incompatible = soft_student.load_state_dict(old_state, strict=False)
        expected_missing = {
            key for key in incompatible.missing_keys if key.startswith("position_")
        }
        unexpected_missing = set(incompatible.missing_keys) - expected_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected soft-position initialization mismatch: "
                f"missing={sorted(unexpected_missing)}, unexpected={incompatible.unexpected_keys}"
            )
        self.student = soft_student
        self.segmentation_arch = str(segmentation_arch).lower()
        if self.segmentation_arch in {"guidedecoder", "guide_decoder", "guide"}:
            self.student.use_guide_decoder()
        elif self.segmentation_arch not in {"mmiunet", "default"}:
            raise ValueError(
                f"Unsupported segmentation_arch={segmentation_arch!r}; "
                "expected 'mmiunet' or 'guidedecoder'."
            )
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.position_init_checkpoint = str(position_init_checkpoint)
        if self.position_init_checkpoint:
            pretrained, metadata = load_position_checkpoint(
                self.position_init_checkpoint, torch.device("cpu")
            )
            if pretrained.head_type != self.position_student.head_type:
                raise ValueError(
                    f"Position init head {pretrained.head_type} does not match "
                    f"joint head {self.position_student.head_type}"
                )
            self.position_student.load_state_dict(pretrained.state_dict(), strict=True)
            self.position_teacher = copy.deepcopy(self.position_student)
            self.position_teacher.eval()
            for parameter in self.position_teacher.parameters():
                parameter.requires_grad = False
            del pretrained
            print(
                f"[soft-position] initialized predictor from "
                f"{self.position_init_checkpoint} (epoch={metadata.get('epoch')})"
            )

        self.image_route_start_probability = float(image_route_start_probability)
        self.image_route_max_probability = float(image_route_max_probability)
        self.image_route_ramp_epochs = int(image_route_ramp_epochs)
        self.position_bce_mix = float(position_bce_mix)
        self.position_gamma_negative = float(position_gamma_negative)
        self.position_gamma_positive = float(position_gamma_positive)
        self.position_cardinality_weight = float(position_cardinality_weight)
        self.validation_text_mode = str(validation_text_mode)
        self._active_image_route = False
        self._decode_tokenizer = AutoTokenizer.from_pretrained(
            self.bert_type, trust_remote_code=True
        )
        self.save_hyperparameters(
            {
                "position_init_checkpoint": self.position_init_checkpoint,
                "image_route_start_probability": self.image_route_start_probability,
                "image_route_max_probability": self.image_route_max_probability,
                "image_route_ramp_epochs": self.image_route_ramp_epochs,
                "position_bce_mix": self.position_bce_mix,
                "position_gamma_negative": self.position_gamma_negative,
                "position_gamma_positive": self.position_gamma_positive,
                "position_cardinality_weight": self.position_cardinality_weight,
                "validation_text_mode": self.validation_text_mode,
                "segmentation_arch": self.segmentation_arch,
            }
        )

    def _route_probability(self) -> float:
        if self.image_route_ramp_epochs <= 0:
            return self.image_route_max_probability
        progress = min(1.0, max(0.0, self.current_epoch / self.image_route_ramp_epochs))
        return self.image_route_start_probability + progress * (
            self.image_route_max_probability - self.image_route_start_probability
        )

    @staticmethod
    def _copy_tokens(tokens: Mapping[str, object]) -> Dict[str, object]:
        return dict(tokens)

    def _full_text_tokens(
        self, tokens: Mapping[str, object], position: torch.Tensor
    ) -> Dict[str, object]:
        output = self._copy_tokens(tokens)
        output["position_vector"] = position.float()
        return output

    def _image_route_tokens(
        self,
        reports: Sequence[str],
        position_probability: torch.Tensor,
        pseudo_label=None,
    ) -> Dict[str, object]:
        stripped = [remove_position_expressions(report) for report in reports]
        output = tokenize_text_list(
            stripped, self.bert_type, max_len=24, device=position_probability.device
        )
        output["position_vector"] = position_probability
        if pseudo_label is not None:
            output["pseudo_label"] = pseudo_label
        output["raw_text"] = stripped
        return output

    def _position_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Train Eq. (10)'s image-only predictor from report-derived labels."""
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        positive = -target * ((1.0 - probability) ** self.position_gamma_positive) * probability.log()
        negative = -(1.0 - target) * (probability ** self.position_gamma_negative) * (
            1.0 - probability
        ).log()
        asymmetric = (positive + negative).mean()
        cardinality = F.smooth_l1_loss(
            probability.sum(dim=1) / 6.0, target.sum(dim=1) / 6.0
        )
        return (
            self.position_bce_mix * bce
            + (1.0 - self.position_bce_mix) * asymmetric
            + self.position_cardinality_weight * cardinality
        )

    def _build_xpatchmix_batch(self, u: dict):
        image_mix, target_mix, text_mix = MMIUNet_SemiWrapper._build_xpatchmix_batch(
            self, u
        )
        position_probability = self.position_student(image_mix).sigmoid()
        if self._active_image_route:
            decoded = self._decode_tokenizer.batch_decode(
                text_mix["input_ids"].detach().cpu(), skip_special_tokens=True
            )
            stripped = [remove_position_expressions(text) for text in decoded]
            text_mix = tokenize_text_list(
                stripped, self.bert_type, max_len=24, device=image_mix.device
            )
        else:
            text_mix = dict(text_mix)
        text_mix["position_vector"] = position_probability
        return image_mix, target_mix, text_mix

    def training_step(self, batch, batch_idx):
        (x_l, _) = batch["labeled"]
        u = batch["unlabeled"]
        image_l, text_l = x_l
        image_w, image_s = u["img_w"], u["img_s"]
        label_l = text_l["pseudo_label"].float()
        label_u = u["text_w"]["pseudo_label"].float()

        position_logits_l = self.position_student(image_l)
        position_logits_u = self.position_student(image_s)
        probability_l = position_logits_l.sigmoid()
        probability_u = position_logits_u.sigmoid()
        position_loss_l = self._position_loss(position_logits_l, label_l)
        position_loss_u = self._position_loss(position_logits_u, label_u)

        route_probability = self._route_probability()
        self._active_image_route = random.random() < route_probability
        if self._active_image_route:
            x_l[1] = self._image_route_tokens(
                self._reports(text_l), probability_l, pseudo_label=label_l
            )
            u["text_s"] = self._image_route_tokens(
                self._reports(u), probability_u, pseudo_label=label_u
            )
        else:
            x_l[1] = self._full_text_tokens(text_l, label_l)
            u["text_s"] = self._full_text_tokens(u["text_s"], label_u)

        # As specified after Eq. (10), the EMA teacher keeps the complete report.
        u["text_w"] = self._full_text_tokens(u["text_w"], label_u)

        output = MMIUNet_SemiWrapper.training_step(self, batch, batch_idx)
        position_total = (
            self.position_loss_weight * position_loss_l
            + self.position_unsup_weight * position_loss_u
        )
        output["loss"] = output["loss"] + position_total
        self.log_dict(
            {
                "position_loss_l": position_loss_l,
                "position_loss_u": position_loss_u,
                "position_total": position_total,
                "image_route_probability": route_probability,
                "image_route_active": float(self._active_image_route),
                "soft_position_token_gate": torch.sigmoid(
                    self.student.position_token_gate
                ),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=image_l.shape[0],
        )
        return output

    def evaluation_inputs(self, x, mode: str | None = None):
        """Construct report/token inputs for one robustness evaluation mode."""
        image, text = x
        mode = str(mode or self.validation_text_mode)
        reports = self._reports(text)
        label = text.get("pseudo_label")
        if mode == "full_text":
            return [image, self._full_text_tokens(text, label.float())]
        if mode == "full_text_image":
            probability = self.position_student(image).sigmoid()
            return [image, self._full_text_tokens(text, probability)]
        if mode == "removed_zero":
            zeros = image.new_zeros((image.shape[0], 6))
            return [image, self._image_route_tokens(reports, zeros, pseudo_label=label)]
        probability = self.position_student(image).sigmoid()
        if mode == "soft_image":
            return [
                image,
                self._image_route_tokens(reports, probability, pseudo_label=label),
            ]
        if mode == "hard_completed":
            completed = complete_position_reports(
                reports, probability, self.position_phrase_threshold
            )
            tokens = tokenize_text_list(
                completed, self.bert_type, max_len=24, device=image.device
            )
            tokens["position_vector"] = probability
            tokens["pseudo_label"] = label
            tokens["raw_text"] = completed
            return [image, tokens]
        if mode == "oracle_soft":
            return [
                image,
                self._image_route_tokens(reports, label.float(), pseudo_label=label),
            ]
        raise ValueError(f"Unknown soft-position evaluation mode: {mode}")

    def validation_step(self, batch, batch_idx):
        x, target = batch
        inputs = self.evaluation_inputs(x, self.validation_text_mode)
        return MMIUNet_SemiWrapper.validation_step(self, (inputs, target), batch_idx)

    def test_step(self, batch, batch_idx):
        x, target = batch
        inputs = self.evaluation_inputs(x, self.validation_text_mode)
        return MMIUNet_SemiWrapper.test_step(self, (inputs, target), batch_idx)
