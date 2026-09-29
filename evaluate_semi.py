#!/usr/bin/env python3
"""Evaluate a Semi-MedRef checkpoint on the fixed test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from engine.wrapper_semi import (
    LanGuideMedSeg_SemiWrapper,
    MMIUNet_SemiWrapper,
)
from utils import config as config_utils
from utils.dataset import MosMed, QaTa


WRAPPERS = {
    "mmiunet": MMIUNet_SemiWrapper,
    "guidedecoder": LanGuideMedSeg_SemiWrapper,
}


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, architecture, and fixed-test-set options."""
    parser = argparse.ArgumentParser(description="Evaluate Semi-MedRef")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-arch", choices=tuple(WRAPPERS), default=None,
        help="Override MODEL.model_arch from the YAML file",
    )
    parser.add_argument("--weights", choices=("student", "teacher"), default="student")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    """Load a student or EMA teacher checkpoint and evaluate the fixed split."""
    cli = parse_args()
    cfg = config_utils.load_cfg_from_cfg_file(cli.config)
    model_arch = str(cli.model_arch or cfg.model_arch).lower()
    model_arch = {"medseg": "guidedecoder", "languide": "guidedecoder"}.get(
        model_arch, model_arch
    )
    if model_arch not in WRAPPERS:
        raise ValueError(f"Unsupported model architecture: {model_arch!r}")

    checkpoint = Path(cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    module = WRAPPERS[model_arch].load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    if cli.weights == "teacher":
        module.student.load_state_dict(module.teacher.state_dict(), strict=True)
    module.eval()

    if cfg.dataset == "qata":
        dataset = QaTa(
            csv_path=cfg.test_csv_path,
            root_path=cfg.test_root_path,
            tokenizer=cfg.bert_type,
            image_size=cfg.image_size,
            mode="test",
        )
    elif cfg.dataset == "mosmed":
        dataset = MosMed(
            csv_path=cfg.test_csv_path,
            root_path=cfg.test_root_path,
            tokenizer=cfg.bert_type,
            image_size=cfg.image_size,
            mode="test",
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset!r}")

    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size or cfg.valid_batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=True,
        persistent_workers=cli.num_workers > 0,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[cli.device],
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.test(module, loader)

    if cli.output_json:
        output = Path(cli.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": cfg.dataset,
            "model_arch": model_arch,
            "weights": cli.weights,
            "checkpoint": str(checkpoint),
            "metrics": results[0] if results else {},
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
