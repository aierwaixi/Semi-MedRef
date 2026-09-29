#!/usr/bin/env python3
"""Train one of three image-only six-region position predictors."""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from position_predictor_models import PositionPredictor, load_semimedref_backbone
from position_predictor_utils import (
    build_position_dataset,
    labels_from_dataset,
    limit_dataset,
    multilabel_metrics,
    save_json,
    seed_everything,
)
from utils.config import load_cfg_from_cfg_file


def parse_args():
    """Parse image-only position-predictor training options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--base-checkpoint", "--base_ckpt", dest="base_ckpt", required=True
    )
    parser.add_argument("--weights", choices=("student", "teacher"), default="student")
    parser.add_argument(
        "--head",
        choices=("gap_mlp", "region_pool", "six_query"),
        required=True,
    )
    parser.add_argument("--dataset", choices=("qata", "mosmed"), default=None)
    parser.add_argument(
        "--train-annotations", "--train_annotations", dest="train_annotations", default=None
    )
    parser.add_argument(
        "--val-annotations", "--val_annotations", dest="val_annotations", default=None
    )
    parser.add_argument("--train-root", "--train_root", dest="train_root", default=None)
    parser.add_argument("--val-root", "--val_root", dest="val_root", default=None)
    parser.add_argument(
        "--output-dir", "--out_dir", dest="out_dir", default="outputs/position_predictor"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--lr-head", "--lr_head", dest="lr_head", type=float, default=3e-4)
    parser.add_argument(
        "--lr-backbone", "--lr_backbone", dest="lr_backbone", type=float, default=3e-5
    )
    parser.add_argument(
        "--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4
    )
    parser.add_argument(
        "--freeze-epochs", "--freeze_epochs", dest="freeze_epochs", type=int, default=5
    )
    parser.add_argument(
        "--unfreeze-last-n-stages",
        "--unfreeze_last_n_stages",
        dest="unfreeze_last_n_stages",
        type=int,
        default=2,
    )
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--query-layers", "--query_layers", dest="query_layers", type=int, default=2)
    parser.add_argument("--query-heads", "--query_heads", dest="query_heads", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-pos-weight", "--no_pos_weight", dest="no_pos_weight", action="store_true")
    parser.add_argument(
        "--image-left-is-anatomical-left",
        "--image_left_is_anatomical_left",
        dest="image_left_is_anatomical_left",
        action="store_true",
    )
    parser.add_argument(
        "--max-train-samples", "--max_train_samples", dest="max_train_samples", type=int, default=0
    )
    parser.add_argument(
        "--max-val-samples", "--max_val_samples", dest="max_val_samples", type=int, default=0
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):
    """Measure validation loss and six-region multi-label performance."""
    model.eval()
    losses, targets, probabilities = [], [], []
    for image, label, _ in tqdm(loader, desc="validate", leave=False):
        image, label = image.to(device), label.to(device)
        logits = model(image)
        losses.append(float(criterion(logits, label).item()) * image.shape[0])
        targets.append(label.cpu().numpy())
        probabilities.append(logits.sigmoid().cpu().numpy())
    targets = np.concatenate(targets)
    probabilities = np.concatenate(probabilities)
    metrics = multilabel_metrics(targets, probabilities, threshold)
    metrics["loss"] = sum(losses) / len(targets)
    return metrics


def main():
    """Train and select the image-only predictor used to initialise ISPG."""
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_cfg_from_cfg_file(args.config)
    dataset_name = (args.dataset or cfg.dataset).lower()
    train_annotations = args.train_annotations or cfg.train_csv_path
    val_annotations = args.val_annotations or cfg.valid_csv_path
    train_root = args.train_root or cfg.train_root_path
    val_root = args.val_root or cfg.valid_root_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = build_position_dataset(
        dataset_name, train_annotations, train_root, cfg.bert_type, "train", cfg.image_size
    )
    val_dataset = build_position_dataset(
        dataset_name, val_annotations, val_root, cfg.bert_type, "valid", cfg.image_size
    )
    train_dataset = limit_dataset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = limit_dataset(val_dataset, args.max_val_samples, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_config = {
        "head_type": args.head,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "query_layers": args.query_layers,
        "query_heads": args.query_heads,
        "anatomical_left_is_image_right": not args.image_left_is_anatomical_left,
    }
    model = PositionPredictor(**model_config)
    load_info = load_semimedref_backbone(model, args.base_ckpt, args.weights)
    model.freeze_backbone()
    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": args.lr_head},
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    labels = labels_from_dataset(train_dataset)
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = torch.as_tensor(
            negatives / np.maximum(positives, 1), dtype=torch.float32, device=device
        ).clamp(0.25, 4.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run_name = (
        f"{dataset_name}_{args.head}_"
        f"{os.path.splitext(os.path.basename(args.base_ckpt))[0]}_{args.weights}"
    )
    output_dir = os.path.abspath(os.path.join(args.out_dir, run_name))
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history.csv")
    best_path = os.path.join(output_dir, "best.ckpt")
    history = []
    best_map = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    print(f"Output: {output_dir}")
    print(f"Loaded pure visual tensors: {load_info['loaded']}/{load_info['total']}")
    print(f"Train/valid samples: {len(train_dataset)}/{len(val_dataset)}")
    print(f"Position prevalence: {labels.mean(axis=0).round(4).tolist()}")
    print(f"pos_weight: {None if pos_weight is None else pos_weight.cpu().tolist()}")

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            model.unfreeze_backbone(args.unfreeze_last_n_stages)
            print(f"Epoch {epoch}: unfroze last {args.unfreeze_last_n_stages} visual stages")
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for image, label, _ in progress:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if scaler.is_enabled()
                else nullcontext()
            )
            with autocast:
                logits = model(image)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * image.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        metrics = evaluate(model, val_loader, criterion, device, args.threshold)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_dataset),
            **{key: value for key, value in metrics.items() if key != "per_class"},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.4f} val={metrics['loss']:.4f} "
            f"mAP={metrics['mAP']:.4f} macroF1={metrics['macro_f1']:.4f} "
            f"exact={metrics['exact_match']:.4f}"
        )
        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "base_checkpoint": os.path.abspath(args.base_ckpt),
                    "base_weights": args.weights,
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "metrics": metrics,
                    "args": vars(args),
                },
                best_path,
            )
            save_json(metrics, os.path.join(output_dir, "best_metrics.json"))

        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
