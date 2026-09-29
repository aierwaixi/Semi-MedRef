"""Semi-MedRef with an image-only six-region position completion branch."""

from __future__ import annotations

import copy
import os
import random
import re
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from engine.wrapper_semi import MMIUNet_SemiWrapper
from position_predictor_models import PositionPredictor
from utils.pos_aug import tokenize_text_list


LOCATION_RE = re.compile(
    r"\b(?:(?:upper|middle|lower)\s+){0,3}(?:left|right)\s+lung\b",
    re.IGNORECASE,
)
COUNT_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+infected\s+areas?\b",
    re.IGNORECASE,
)
FUZZY_RE = re.compile(r"\ba region of the lung\b", re.IGNORECASE)


def remove_position_expressions(report: str) -> str:
    """Remove explicit or augmented lung-location phrases from one report."""
    text = LOCATION_RE.sub(" ", str(report or ""))
    text = text.replace("[UNK_POS]", " ")
    text = FUZZY_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*(?:,|and)\s*", ", ", text)
    return text.strip(" ,")


def probabilities_to_phrases(probabilities: torch.Tensor, threshold: float) -> List[List[str]]:
    """Convert six image-derived probabilities to canonical lung phrases."""
    decisions = probabilities >= float(threshold)
    phrases = []
    vertical = ("upper", "middle", "lower")
    for row_index in range(probabilities.shape[0]):
        row_phrases = []
        for side_index, side in enumerate(("left", "right")):
            offset = side_index * 3
            active = [
                vertical[index]
                for index in range(3)
                if bool(decisions[row_index, offset + index])
            ]
            if active:
                row_phrases.append(" ".join(active + [side, "lung"]))
        if not row_phrases:
            best = int(probabilities[row_index].argmax().item())
            side = "left" if best < 3 else "right"
            row_phrases.append(f"{vertical[best % 3]} {side} lung")
        phrases.append(row_phrases)
    return phrases


def complete_position_reports(
    reports: Sequence[str],
    probabilities: torch.Tensor,
    threshold: float,
) -> List[str]:
    """Replace report locations with phrases predicted from the image."""
    phrase_lists = probabilities_to_phrases(probabilities.detach(), threshold)
    completed = []
    for report, phrases in zip(reports, phrase_lists):
        base = remove_position_expressions(report)
        base = COUNT_RE.sub(f"{len(phrases)} infected areas", base)
        location_text = " and ".join(phrases)
        completed.append(f"{base}, {location_text}" if base else location_text)
    return completed


def load_imagenet_convnext(model: PositionPredictor, checkpoint_path: str) -> int:
    """Load matching ImageNet ConvNeXt tensors into the position backbone."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    target = model.backbone.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    model.backbone.load_state_dict(matched, strict=False)
    if len(matched) != len(target):
        raise RuntimeError(
            f"Position backbone initialization incomplete: {len(matched)}/{len(target)} tensors"
        )
    return len(matched)


class PositionGuidedSemiWrapper(MMIUNet_SemiWrapper):
    """Joint segmentation and image-only position learning.

    ``position_supervision=report`` uses report-derived six-region labels on
    both labeled and unlabeled images.

    ``position_supervision=ema`` uses report-derived labels only on the
    segmentation-labeled subset.  After burn-in, an EMA position teacher
    supplies soft targets for the unlabeled weak view.
    """

    def __init__(
        self,
        *args,
        position_supervision: str = "report",
        position_head: str = "gap_mlp",
        position_loss_weight: float = 0.1,
        position_unsup_weight: float = 0.1,
        position_confidence: float = 0.8,
        position_phrase_threshold: float = 0.5,
        position_completion_start_epoch: int = 5,
        position_completion_probability: float = 1.0,
        position_pretrained: str = "./pretrained/convnext_tiny_22k_224.pth",
        validation_text_mode: str = "position_free_completed",
        **kwargs,
    ):
        """Configure joint segmentation and six-region position learning."""
        super().__init__(*args, **kwargs)
        self.position_supervision = str(position_supervision).lower()
        if self.position_supervision not in ("report", "ema"):
            raise ValueError("position_supervision must be report or ema")
        self.position_loss_weight = float(position_loss_weight)
        self.position_unsup_weight = float(position_unsup_weight)
        self.position_confidence = float(position_confidence)
        self.position_phrase_threshold = float(position_phrase_threshold)
        self.position_completion_start_epoch = int(position_completion_start_epoch)
        self.position_completion_probability = float(position_completion_probability)
        self.validation_text_mode = str(validation_text_mode)

        self.position_student = PositionPredictor(head_type=position_head)
        loaded = load_imagenet_convnext(self.position_student, position_pretrained)
        print(f"[position] loaded ImageNet ConvNeXt tensors: {loaded}")
        self.position_teacher = copy.deepcopy(self.position_student)
        self.position_teacher.eval()
        for parameter in self.position_teacher.parameters():
            parameter.requires_grad = False

        self.save_hyperparameters(
            {
                "position_supervision": self.position_supervision,
                "position_head": position_head,
                "position_loss_weight": self.position_loss_weight,
                "position_unsup_weight": self.position_unsup_weight,
                "position_confidence": self.position_confidence,
                "position_phrase_threshold": self.position_phrase_threshold,
                "position_completion_start_epoch": self.position_completion_start_epoch,
                "position_completion_probability": self.position_completion_probability,
                "position_pretrained": position_pretrained,
                "validation_text_mode": self.validation_text_mode,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.student.parameters(), "lr": self.lr},
                {"params": self.position_student.parameters(), "lr": self.lr * 0.1},
            ],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def _reports(text_or_batch) -> List[str]:
        reports = text_or_batch.get("raw_text")
        if reports is None:
            raise KeyError("Position-guided training requires raw_text in the dataset batch")
        if isinstance(reports, str):
            return [reports]
        return list(reports)

    def _tokens_from_completed(
        self,
        reports: Sequence[str],
        probabilities: torch.Tensor,
        pseudo_label=None,
    ) -> Dict[str, torch.Tensor]:
        completed = complete_position_reports(
            reports, probabilities, self.position_phrase_threshold
        )
        tokens = tokenize_text_list(
            completed, tokenizer_name=self.bert_type, max_len=24, device=probabilities.device
        )
        if pseudo_label is not None:
            tokens["pseudo_label"] = pseudo_label
        tokens["raw_text"] = completed
        return tokens

    def _should_complete(self) -> bool:
        return (
            self.current_epoch >= self.position_completion_start_epoch
            and random.random() < self.position_completion_probability
        )

    def training_step(self, batch, batch_idx):
        (x_l, _) = batch["labeled"]
        u = batch["unlabeled"]
        image_l, text_l = x_l
        image_w, image_s = u["img_w"], u["img_s"]
        unlabeled_reports = self._reports(u)

        position_logits_l = self.position_student(image_l)
        label_l = text_l["pseudo_label"].float()
        position_loss_l = F.binary_cross_entropy_with_logits(position_logits_l, label_l)

        position_logits_u = self.position_student(image_s)
        teacher_probability = None
        if self.position_supervision == "report":
            label_u = u["text_w"]["pseudo_label"].float()
            position_loss_u = F.binary_cross_entropy_with_logits(position_logits_u, label_u)
        elif self.current_epoch >= self.burn_in_epochs:
            with torch.no_grad():
                self.position_teacher.eval()
                teacher_probability = self.position_teacher(image_w).sigmoid()
                confidence = torch.maximum(teacher_probability, 1.0 - teacher_probability)
                mask = (confidence >= self.position_confidence).float()
            element_loss = F.binary_cross_entropy_with_logits(
                position_logits_u, teacher_probability, reduction="none"
            )
            position_loss_u = (element_loss * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            position_loss_u = position_loss_l.new_zeros(())

        if self.position_supervision == "ema":
            # Strict setting: the unlabeled report-derived vector is not visible
            # to either the position loss or unlabeled PACL.  Location words
            # are also removed from both segmentation views during burn-in.
            position_free_reports = [
                remove_position_expressions(report) for report in unlabeled_reports
            ]
            position_free_tokens = tokenize_text_list(
                position_free_reports,
                tokenizer_name=self.bert_type,
                max_len=24,
                device=position_logits_u.device,
            )
            u["text_w"] = dict(position_free_tokens)
            u["text_s"] = dict(position_free_tokens)
            u["text_s_str"] = position_free_reports
            for key in ("text_w", "text_s", "text"):
                if key in u and isinstance(u[key], dict):
                    u[key] = dict(u[key])
                    u[key].pop("pseudo_label", None)

        if self._should_complete():
            x_l[1] = self._tokens_from_completed(
                self._reports(text_l),
                position_logits_l.sigmoid(),
                pseudo_label=text_l.get("pseudo_label"),
            )
            completed_u = complete_position_reports(
                unlabeled_reports, position_logits_u.sigmoid(), self.position_phrase_threshold
            )
            u["text_s_str"] = completed_u
            u["text_s"] = tokenize_text_list(
                completed_u,
                tokenizer_name=self.bert_type,
                max_len=24,
                device=position_logits_u.device,
            )
            if self.position_supervision == "report":
                u["text_s"]["pseudo_label"] = u["text_w"]["pseudo_label"]
            else:
                if teacher_probability is None:
                    with torch.no_grad():
                        self.position_teacher.eval()
                        teacher_probability = self.position_teacher(image_w).sigmoid()
                teacher_completed = complete_position_reports(
                    unlabeled_reports,
                    teacher_probability,
                    self.position_phrase_threshold,
                )
                u["text_w"] = tokenize_text_list(
                    teacher_completed,
                    tokenizer_name=self.bert_type,
                    max_len=24,
                    device=position_logits_u.device,
                )

        output = super().training_step(batch, batch_idx)
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
                "position_completion_active": float(
                    self.current_epoch >= self.position_completion_start_epoch
                ),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=image_l.shape[0],
        )
        return output

    def _evaluation_inputs(self, x):
        image, text = x
        if self.validation_text_mode != "position_free_completed":
            return x
        probabilities = self.position_student(image).sigmoid()
        tokens = self._tokens_from_completed(
            self._reports(text), probabilities, pseudo_label=text.get("pseudo_label")
        )
        return [image, tokens]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return super().validation_step((self._evaluation_inputs(x), y), batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return super().test_step((self._evaluation_inputs(x), y), batch_idx)

    def on_train_start(self):
        super().on_train_start()
        self.position_teacher.eval()

    def on_before_zero_grad(self, optimizer):
        super().on_before_zero_grad(optimizer)
        decay = float(self._ema_m_now())
        with torch.no_grad():
            student = self.position_student.state_dict()
            teacher = self.position_teacher.state_dict()
            for key, target in teacher.items():
                source = student[key]
                if target.dtype.is_floating_point:
                    target.mul_(decay).add_(source, alpha=1.0 - decay)
                else:
                    target.copy_(source)
