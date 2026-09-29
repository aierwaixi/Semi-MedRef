#!/usr/bin/env python3
"""Recreate deterministic labeled/unlabeled manifests used in the paper."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd


DEFAULT_RATIOS = (0.01, 0.02, 0.05, 0.15)
LOCATION_PATTERN = re.compile(
    r"\b(?:(?:upper|middle|lower)\s+){0,3}(left|right)\s+lung\b",
    flags=re.IGNORECASE,
)


def ratio_tag(value: float) -> str:
    """Format a split ratio without binary floating-point artefacts."""
    return format(Decimal(str(value)).normalize(), "f")


def report_position_vector(report: str):
    """Return [left upper/middle/lower, right upper/middle/lower]."""
    text = str(report or "").lower()
    text = text.replace("all left lung", "upper middle lower left lung")
    text = text.replace("all right lung", "upper middle lower right lung")
    vector = [0] * 6
    for match in LOCATION_PATTERN.finditer(text):
        phrase = match.group(0)
        side = 0 if "left" in phrase else 1
        active = [name in phrase for name in ("upper", "middle", "lower")]
        if not any(active):
            active = [True, True, True]
        for level, present in enumerate(active):
            if present:
                vector[side * 3 + level] = 1
    return vector


def prepare_qata(input_path: Path, output_dir: Path, ratios, seed: int):
    """Create image-grouped labeled and unlabeled QaTa manifests."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    for split in ("train", "valid", "test"):
        if not isinstance(payload.get(split), list):
            raise ValueError(f"QaTa annotation must contain a {split!r} list")

    buckets = defaultdict(list)
    for sample in payload["train"]:
        buckets[str(sample["image_path"])].append(sample)
    keys = list(buckets)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ratio in ratios:
        shuffled = list(keys)
        random.Random(seed).shuffle(shuffled)
        labeled_keys = set(shuffled[: max(1, round(len(shuffled) * ratio))])
        labeled, unlabeled = [], []
        for key, samples in buckets.items():
            (labeled if key in labeled_keys else unlabeled).extend(samples)
        tag = ratio_tag(ratio)
        common = {"valid": payload["valid"], "test": payload["test"]}
        (output_dir / f"split_labeled_{tag}.json").write_text(
            json.dumps({"train": labeled, **common}, indent=2), encoding="utf-8"
        )
        (output_dir / f"split_unlabeled_{tag}.json").write_text(
            json.dumps({"train": unlabeled, **common}, indent=2), encoding="utf-8"
        )
        print(f"QaTa r={tag}: labeled={len(labeled)}, unlabeled={len(unlabeled)}")


def prepare_mosmed(input_path: Path, output_dir: Path, ratios, seed: int):
    """Create image-grouped MosMed CSVs with report-derived position labels."""
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    frame.columns = [str(name).strip().lstrip("\ufeff") for name in frame.columns]
    if not {"Image", "text"}.issubset(frame.columns):
        raise ValueError("MosMed CSV must contain Image and text columns")
    frame = frame.dropna(subset=["Image", "text"]).copy()
    frame["Image"] = frame["Image"].astype(str).str.strip()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[(frame["Image"] != "") & (frame["text"] != "")]
    frame["pseudo_label"] = frame["text"].map(
        lambda text: json.dumps(report_position_vector(text))
    )

    images = sorted(frame["Image"].unique())
    random.Random(seed).shuffle(images)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ratio in ratios:
        count = max(1, math.floor(len(images) * ratio))
        labeled_keys = set(images[:count])
        labeled = frame[frame["Image"].isin(labeled_keys)]
        unlabeled = frame[~frame["Image"].isin(labeled_keys)]
        tag = ratio_tag(ratio)
        complement = ratio_tag(1.0 - ratio)
        labeled.to_csv(
            output_dir / f"labeled_{tag}.plabel.csv",
            index=False,
            encoding="utf-8-sig",
        )
        unlabeled.to_csv(
            output_dir / f"unlabeled_{complement}.plabel.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"MosMed r={tag}: labeled={len(labeled)}, unlabeled={len(unlabeled)}")


def main():
    """Validate arguments and generate the requested dataset partitions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("qata", "mosmed"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for ratio in args.ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"Ratios must be in (0, 1), got {ratio}")
    if args.dataset == "qata":
        prepare_qata(args.input, args.output_dir, args.ratios, args.seed)
    else:
        prepare_mosmed(args.input, args.output_dir, args.ratios, args.seed)


if __name__ == "__main__":
    main()
