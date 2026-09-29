"""Removal-aware Image-Derived Soft Position Guidance (ISPG)."""

from __future__ import annotations

import hashlib
import random
from typing import Mapping, Sequence

import torch

from engine.wrapper_semi import MMIUNet_SemiWrapper
from engine.wrapper_semi_position import remove_position_expressions
from engine.wrapper_semi_soft_position import SoftPositionSemiWrapper
from utils.pos_aug import tokenize_text_list


class SoftPositionRemovalWrapper(SoftPositionSemiWrapper):
    """Train ISPG under controlled case-level removal of position phrases.

    For a selected report, every explicit position expression is removed while
    all remaining report content is retained. Training selection is stochastic;
    validation selection is deterministic so that checkpoint selection is
    reproducible. This implements the removal-aware route in Eq. (10).
    """

    def __init__(
        self,
        *args,
        position_removal_probability: float = 0.0,
        position_validation_removal_probability: float | None = None,
        position_removal_seed: int = 42,
        **kwargs,
    ):
        """Configure deterministic case-level removal-aware ISPG training."""
        super().__init__(*args, **kwargs)
        self.position_removal_probability = float(position_removal_probability)
        self.position_validation_removal_probability = float(
            self.position_removal_probability
            if position_validation_removal_probability is None
            else position_validation_removal_probability
        )
        self.position_removal_seed = int(position_removal_seed)
        for value in (
            self.position_removal_probability,
            self.position_validation_removal_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Position-removal probabilities must be in [0, 1]")
        self.save_hyperparameters(
            {
                "position_removal_probability": self.position_removal_probability,
                "position_validation_removal_probability": (
                    self.position_validation_removal_probability
                ),
                "position_removal_seed": self.position_removal_seed,
            }
        )

    def _deterministic_remove(self, report: str, probability: float) -> bool:
        payload = f"{self.position_removal_seed}|{report}".encode("utf-8")
        score = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        return score < probability

    def _corrupt_reports(
        self,
        reports: Sequence[str],
        probability: float,
        deterministic: bool,
    ) -> list[str]:
        output = []
        for report in reports:
            remove = (
                self._deterministic_remove(report, probability)
                if deterministic
                else random.random() < probability
            )
            output.append(
                remove_position_expressions(report) if remove else str(report)
            )
        return output

    def _tokens_at_rate(
        self,
        reports: Sequence[str],
        position_vector: torch.Tensor,
        pseudo_label=None,
        probability: float = 0.0,
        deterministic: bool = False,
    ) -> Mapping[str, object]:
        corrupted = self._corrupt_reports(reports, probability, deterministic)
        output = tokenize_text_list(
            corrupted,
            self.bert_type,
            max_len=24,
            device=position_vector.device,
        )
        output["position_vector"] = position_vector
        if pseudo_label is not None:
            output["pseudo_label"] = pseudo_label
        output["raw_text"] = corrupted
        return output

    def _image_route_tokens(
        self,
        reports: Sequence[str],
        position_probability: torch.Tensor,
        pseudo_label=None,
    ) -> Mapping[str, object]:
        return self._tokens_at_rate(
            reports,
            position_probability,
            pseudo_label=pseudo_label,
            probability=self.position_removal_probability,
            deterministic=False,
        )

    def _build_xpatchmix_batch(self, unlabeled: dict):
        image_mix, target_mix, text_mix = (
            MMIUNet_SemiWrapper._build_xpatchmix_batch(self, unlabeled)
        )
        position_probability = self.position_student(image_mix).sigmoid()
        decoded = self._decode_tokenizer.batch_decode(
            text_mix["input_ids"].detach().cpu(), skip_special_tokens=True
        )
        text_mix = self._tokens_at_rate(
            decoded,
            position_probability,
            probability=self.position_removal_probability,
            deterministic=False,
        )
        return image_mix, target_mix, text_mix

    def evaluation_inputs(self, inputs, mode: str | None = None):
        """Apply the requested removal rate before selecting an ISPG route."""
        image, text = inputs
        mode = str(mode or self.validation_text_mode)
        if mode not in {"removed_zero", "soft_image", "oracle_soft"}:
            return super().evaluation_inputs(inputs, mode)
        reports = self._reports(text)
        label = text.get("pseudo_label")
        if mode == "removed_zero":
            position_vector = image.new_zeros((image.shape[0], 6))
        elif mode == "oracle_soft":
            if label is None:
                raise KeyError("oracle_soft requires report-derived pseudo_label")
            position_vector = label.float()
        else:
            position_vector = self.position_student(image).sigmoid()
        return [
            image,
            self._tokens_at_rate(
                reports,
                position_vector,
                pseudo_label=label,
                probability=self.position_validation_removal_probability,
                deterministic=True,
            ),
        ]
