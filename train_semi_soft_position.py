#!/usr/bin/env python3
"""Train the optional image-derived soft position guidance (ISPG) extension."""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch

import train_semi as core
from engine.wrapper_semi_soft_position_dropout import SoftPositionRemovalWrapper
from utils.dataset_semi_position import (
    MosMedPositionEval,
    MosMedPositionLabeled,
    MosMedPositionUnlabeled,
    QaTaPositionEval,
    QaTaPositionLabeled,
    QaTaPositionUnlabeled,
)


ISPG_ARGUMENTS = (
    "position_supervision",
    "position_head",
    "position_loss_weight",
    "position_unsup_weight",
    "position_confidence",
    "position_phrase_threshold",
    "position_completion_start_epoch",
    "position_completion_probability",
    "position_pretrained",
    "position_init_checkpoint",
    "validation_text_mode",
    "image_route_start_probability",
    "image_route_max_probability",
    "image_route_ramp_epochs",
    "position_bce_mix",
    "position_gamma_negative",
    "position_gamma_positive",
    "position_cardinality_weight",
    "position_removal_probability",
    "position_validation_removal_probability",
    "position_removal_seed",
)


def parse_args() -> argparse.Namespace:
    """Parse removal-aware ISPG training arguments."""
    parser = argparse.ArgumentParser(description="Train Semi-MedRef with ISPG")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ratio", choices=core.SUPPORTED_RATIOS, default=None)
    parser.add_argument(
        "--model-arch", choices=tuple(core.WRAPPERS), default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--position-init-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train the optional ISPG extension with controlled position removal."""
    cli = parse_args()
    cfg = core.load_config(cli)
    if cli.position_init_checkpoint is not None:
        cfg.position_init_checkpoint = cli.position_init_checkpoint
    ratio = cli.ratio or core._ratio_from_path(cfg.train_csv_path)

    pl.seed_everything(int(cfg.seed), workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Reuse the exact loader construction from the core method while attaching
    # raw reports required by the position-removal route.
    core.QaTaSemiLabeled = QaTaPositionLabeled
    core.QaTaSemiUnlabeled = QaTaPositionUnlabeled
    core.QaTa = QaTaPositionEval
    core.MosMedSemiLabeled = MosMedPositionLabeled
    core.MosMedSemiUnlabeled = MosMedPositionUnlabeled
    core.MosMed = MosMedPositionEval
    train_loader, validation_loader = core.build_loaders(cfg, ratio)

    ispg_kwargs = {
        name: getattr(cfg, name)
        for name in ISPG_ARGUMENTS
        if hasattr(cfg, name)
    }
    ispg_kwargs["segmentation_arch"] = cfg.model_arch
    module = SoftPositionRemovalWrapper(**core.module_kwargs(cfg), **ispg_kwargs)
    core.run_training(cfg, ratio, cli, module, train_loader, validation_loader)


if __name__ == "__main__":
    main()
