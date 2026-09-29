#!/usr/bin/env python3
"""Evaluate a trained six-region predictor and save per-sample probabilities."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from position_predictor_models import REGION_NAMES, load_position_checkpoint
from position_predictor_utils import (
    build_position_dataset,
    limit_dataset,
    multilabel_metrics,
    save_json,
    seed_everything,
)
from utils.config import load_cfg_from_cfg_file


def parse_args():
    """Parse position-predictor evaluation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", "--ckpt", dest="ckpt", required=True)
    parser.add_argument("--dataset", choices=("qata", "mosmed"), default=None)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", "--out_dir", dest="out_dir", default=None)
    return parser.parse_args()


def main():
    """Evaluate a selected image-only position predictor on one fixed split."""
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_cfg_from_cfg_file(args.config)
    checkpoint_meta = torch.load(args.ckpt, map_location="cpu")
    dataset_name = (args.dataset or checkpoint_meta.get("dataset") or cfg.dataset).lower()
    annotation_default = getattr(cfg, f"{args.split}_csv_path")
    root_default = getattr(cfg, f"{args.split}_root_path")
    dataset = build_position_dataset(
        dataset_name,
        args.annotations or annotation_default,
        args.root or root_default,
        cfg.bert_type,
        args.split,
        cfg.image_size,
    )
    dataset = limit_dataset(dataset, args.max_samples, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_position_checkpoint(args.ckpt, device)
    model.to(device).eval()

    all_targets, all_probabilities, all_indices = [], [], []
    with torch.no_grad():
        for image, label, index in tqdm(loader, desc=f"evaluate {args.split}"):
            logits = model(image.to(device))
            all_targets.append(label.numpy())
            all_probabilities.append(logits.sigmoid().cpu().numpy())
            all_indices.append(index.numpy())
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    indices = np.concatenate(all_indices)
    metrics = multilabel_metrics(targets, probabilities, args.threshold)

    output_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), f"eval_{args.split}"
    )
    os.makedirs(output_dir, exist_ok=True)
    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    with open(
        os.path.join(output_dir, "predictions.csv"), "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["dataset_index"]
        for name in REGION_NAMES:
            fieldnames.extend((f"target_{name}", f"prob_{name}", f"pred_{name}"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, dataset_index in enumerate(indices):
            row = {"dataset_index": int(dataset_index)}
            for region_index, name in enumerate(REGION_NAMES):
                probability = float(probabilities[row_index, region_index])
                row[f"target_{name}"] = int(targets[row_index, region_index])
                row[f"prob_{name}"] = probability
                row[f"pred_{name}"] = int(probability >= args.threshold)
            writer.writerow(row)

    print(f"Results: {os.path.abspath(output_dir)}")
    print(
        f"mAP={metrics['mAP']:.4f} macroF1={metrics['macro_f1']:.4f} "
        f"microF1={metrics['micro_f1']:.4f} exact={metrics['exact_match']:.4f} "
        f"Jaccard={metrics['sample_jaccard']:.4f}"
    )


if __name__ == "__main__":
    main()
