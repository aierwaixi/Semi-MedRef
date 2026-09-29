#!/usr/bin/env python3
"""Evaluate dual-route soft-position Semi-MedRef on QaTa-COV19 or MosMedData+."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm

from engine.wrapper_semi_soft_position_dropout import SoftPositionRemovalWrapper
from position_predictor_utils import multilabel_metrics
from utils.config import load_cfg_from_cfg_file
from utils.dataset_semi_position import MosMedPositionEval, QaTaPositionEval


MODES = (
    "full_text",
    "full_text_image",
    "full_text_no_token",
    "removed_zero",
    "hard_completed",
    "soft_image",
    "oracle_soft",
)


def parse_args():
    """Parse ISPG robustness modes and the position-removal rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", "--out_dir", dest="output_dir", required=True)
    parser.add_argument(
        "--removal-rate",
        type=float,
        default=1.0,
        help="Fraction of reports whose position phrases are removed (default: 1.0)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Evaluation routes to run; defaults to all routes.",
    )
    return parser.parse_args()


def metric_bundle(device):
    """Create fresh segmentation metrics on the requested device."""
    return {
        "accuracy": Accuracy(task="binary").to(device),
        "dice": Dice().to(device),
        "iou": BinaryJaccardIndex().to(device),
    }


def main():
    """Compare missing-position inference modes with one ISPG checkpoint."""
    args = parse_args()
    if not 0.0 <= args.removal_rate <= 1.0:
        raise ValueError("--removal-rate must be in [0, 1]")
    cfg = load_cfg_from_cfg_file(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_class = MosMedPositionEval if cfg.dataset == "mosmed" else QaTaPositionEval
    dataset = dataset_class(
        csv_path=cfg.test_csv_path,
        root_path=cfg.test_root_path,
        tokenizer=cfg.bert_type,
        mode="test",
        image_size=cfg.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = SoftPositionRemovalWrapper.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.position_validation_removal_probability = args.removal_rate
    model.to(device).eval()
    metrics = {mode: metric_bundle(device) for mode in args.modes}
    criterion = DiceCELoss(sigmoid=True)
    losses = {mode: 0.0 for mode in args.modes}
    targets_position, probabilities_position = [], []
    sample_count = 0

    with torch.inference_mode():
        for (image, text), target in tqdm(loader, desc="soft-position test"):
            image, target = image.to(device), target.to(device)
            text = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in text.items()
            }
            probability_position = model.position_student(image).sigmoid()
            probabilities_position.append(probability_position.cpu().numpy())
            targets_position.append(text["pseudo_label"].cpu().numpy())
            for mode in args.modes:
                if mode == "full_text_no_token":
                    inputs = [
                        image,
                        {
                            "input_ids": text["input_ids"],
                            "attention_mask": text["attention_mask"],
                        },
                    ]
                else:
                    inputs = model.evaluation_inputs([image, text], mode)
                logits = model.student(inputs)
                losses[mode] += float(criterion(logits, target).item()) * image.shape[0]
                probability = logits.sigmoid()
                for metric in metrics[mode].values():
                    metric.update(probability, target)
            sample_count += image.shape[0]

    segmentation = {
        mode: {
            "loss": losses[mode] / sample_count,
            **{name: float(metric.compute().item()) for name, metric in bundle.items()},
        }
        for mode, bundle in metrics.items()
    }
    output = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "position_removal_rate": args.removal_rate,
        "samples": sample_count,
        "segmentation": segmentation,
        "position_prediction": multilabel_metrics(
            np.concatenate(targets_position), np.concatenate(probabilities_position), 0.5
        ),
    }
    if "full_text" in segmentation and "removed_zero" in segmentation:
        full = segmentation["full_text"]["dice"]
        removed = segmentation["removed_zero"]["dice"]
        output["recovery"] = {
            mode: (
                (segmentation[mode]["dice"] - removed) / (full - removed)
                if full > removed
                else float("nan")
            )
            for mode in ("hard_completed", "soft_image", "oracle_soft")
            if mode in segmentation
        }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"Saved: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
