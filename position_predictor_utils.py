"""Data and metric helpers shared by position-prediction experiments."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset, Subset

from utils.dataset import MosMed, QaTa

from position_predictor_models import REGION_NAMES


class PositionDataset(Dataset):
    """Expose only image and six-region label from the existing datasets."""

    def __init__(self, base: Dataset):
        """Validate and cache the six-region report-derived targets."""
        self.base = base
        labels = getattr(base, "pseudo_label_list", None)
        if labels is None:
            raise ValueError("This annotation file has no pseudo_label column/list.")
        bad = [i for i, label in enumerate(labels) if label is None or len(label) != 6]
        if bad:
            raise ValueError(
                f"{len(bad)} samples have missing/non-6D pseudo labels; first indices: {bad[:5]}"
            )
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        """Return the number of underlying image-report samples."""
        return len(self.base)

    def __getitem__(self, index):
        """Return an image, its position target, and the source index."""
        (image, text), _ = self.base[index]
        label = text.get("pseudo_label")
        if label is None:
            label = torch.as_tensor(self.labels[index])
        return image.float(), label.float(), int(index)


def build_position_dataset(
    dataset_name: str,
    annotation_path: str,
    root_path: str,
    tokenizer_path: str,
    split: str,
    image_size,
) -> PositionDataset:
    """Wrap a QaTa or MosMed split for image-only position prediction."""
    cls = QaTa if dataset_name.lower() == "qata" else MosMed
    base = cls(
        csv_path=annotation_path,
        root_path=root_path,
        tokenizer=tokenizer_path,
        mode=split,
        image_size=image_size,
    )
    return PositionDataset(base)


def limit_dataset(dataset: Dataset, max_samples: int, seed: int = 42) -> Dataset:
    """Select a deterministic subset for inexpensive diagnostic runs."""
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """Recover position targets from a full or deterministically subset dataset."""
    if isinstance(dataset, Subset):
        base = labels_from_dataset(dataset.dataset)
        return base[np.asarray(dataset.indices)]
    if isinstance(dataset, PositionDataset):
        return dataset.labels
    raise TypeError(f"Cannot extract labels from {type(dataset)}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for position-predictor experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute thresholded and ranking metrics for six-region predictions."""
    targets = targets.astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = (predictions * targets).sum(axis=0)
    fp = (predictions * (1 - targets)).sum(axis=0)
    fn = ((1 - predictions) * targets).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    jaccard_per_sample = (
        (predictions & targets).sum(axis=1)
        / np.maximum((predictions | targets).sum(axis=1), 1)
    )

    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(
        micro_precision + micro_recall, 1e-12
    )

    per_class = {}
    aucs, aps = [], []
    for index, name in enumerate(REGION_NAMES):
        target = targets[:, index]
        try:
            auc = float(roc_auc_score(target, probabilities[:, index]))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(target, probabilities[:, index]))
        except ValueError:
            ap = float("nan")
        aucs.append(auc)
        aps.append(ap)
        per_class[name] = {
            "prevalence": float(target.mean()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "auroc": auc,
            "average_precision": ap,
        }

    return {
        "threshold": float(threshold),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(micro_f1),
        "macro_auroc": float(np.nanmean(aucs)),
        "mAP": float(np.nanmean(aps)),
        "sample_jaccard": float(jaccard_per_sample.mean()),
        "exact_match": float((predictions == targets).all(axis=1).mean()),
        "hamming_accuracy": float((predictions == targets).mean()),
        "per_class": per_class,
    }


def save_json(data: Dict, path: str) -> None:
    """Write a UTF-8 JSON result after creating its parent directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=True)
