#!/usr/bin/env python3
"""Prepare deterministic, leakage-audited PosMed SSL manifests.

The six natural-answer QA tables are the only text source.  Brain MRI uses
the original Cheng patient-disjoint folds, lung CT is split by study ID, and
all modalities are checked for exact image/mask reuse across partitions.

This is the preprocessing used by the paper's ``Cross-domain Generalization``
experiment.  It creates fixed labeled, unlabeled, validation, and test
manifests without using the test set for model or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_posmed import (  # noqa: E402
    MODALITIES,
    parse_answer_positions,
    posmed_pair_path_candidates,
    read_posmed_qa_csv,
)


DEFAULT_POSMED_DIR = PROJECT_ROOT / "datasets/PosMed"
DEFAULT_QA_DIR = DEFAULT_POSMED_DIR / "annotations/qa"
DEFAULT_DATA_ROOT = DEFAULT_POSMED_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_splits/PosMed/generated"
DEFAULT_BRAIN_PID_MAP = (
    PROJECT_ROOT / "data_splits/PosMed/brain_mri_slice_to_pid.csv"
)

OFFICIAL_QA_FILENAMES = {
    "brain_mri": "brain_tumors_ct_scan.csv",
    "breast_ultrasound": "breast_tumors_ct_scan.csv",
    "lung_ct": "lung_CT.csv",
    "lung_xray": "lung_Xray.csv",
    "polyp_endoscopy": "polyp_endoscopy.csv",
    "skin_dermoscopy": "skin_rgbimage.csv",
}
DERIVED_CASE_VALIDATION = {"polyp_endoscopy", "skin_dermoscopy"}
SPLIT_NAMES = ("train_labeled", "train_unlabeled", "val", "test")
_CT_STUDY_RE = re.compile(r"^(?P<study>.+)_\d+$")


def _stable_modality_seed(seed: int, modality: str, purpose: int) -> int:
    return int(seed) + 10_000 * int(purpose) + 101 * MODALITIES.index(modality)


def _sample_id(record: Mapping[str, Any]) -> str:
    return str(record["sample_id"])


def _count_by_modality(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, int]:
    counts = Counter(str(record["modality"]) for record in records)
    return {modality: int(counts.get(modality, 0)) for modality in MODALITIES}


def _assert_unique(
    records: Sequence[Mapping[str, Any]], name: str
) -> None:
    counts = Counter(_sample_id(record) for record in records)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(
            f"{name} contains duplicate sample IDs: {duplicates[:10]}"
        )


def _assert_pairwise_disjoint(
    named_records: Mapping[str, Sequence[Mapping[str, Any]]],
    key,
    description: str,
) -> Dict[str, int]:
    names = list(named_records)
    key_sets = {
        name: {key(record) for record in records}
        for name, records in named_records.items()
    }
    audit: Dict[str, int] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = key_sets[left_name] & key_sets[right_name]
            audit[f"{left_name}__{right_name}"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"{description} leakage between {left_name} and "
                    f"{right_name}: {len(overlap)}; "
                    f"examples={sorted(map(str, overlap))[:5]}"
                )
    return audit


def _pairwise_overlap_audit(
    named_records: Mapping[str, Sequence[Mapping[str, Any]]],
    key,
) -> Dict[str, int]:
    """Count pairwise overlaps without treating them as leakage."""

    names = list(named_records)
    key_sets = {
        name: {key(record) for record in records}
        for name, records in named_records.items()
    }
    audit: Dict[str, int] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = key_sets[left_name] & key_sets[right_name]
            audit[f"{left_name}__{right_name}"] = len(overlap)
    return audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_pair(
    record: Mapping[str, Any], data_root: Path
) -> Tuple[Path, Path]:
    mask_path, image_candidates = posmed_pair_path_candidates(
        record, data_root
    )
    image_path = next(
        (candidate for candidate in image_candidates if candidate.is_file()),
        image_candidates[0],
    )
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            f"Missing PosMed pair for {_sample_id(record)}: "
            f"image={image_path}, mask={mask_path}"
        )
    return image_path, mask_path


def _attach_content_hashes(
    records: Sequence[Dict[str, Any]], data_root: Path
) -> Dict[str, Any]:
    path_hash_cache: Dict[str, str] = {}
    total_bytes = 0
    for index, record in enumerate(records, start=1):
        image_path, mask_path = _resolve_pair(record, data_root)
        for field, path in (
            ("_image_sha256", image_path),
            ("_mask_sha256", mask_path),
        ):
            key = str(path.resolve())
            if key not in path_hash_cache:
                path_hash_cache[key] = _sha256_file(path)
                total_bytes += path.stat().st_size
            record[field] = path_hash_cache[key]
        record["_resolved_image"] = str(image_path)
        record["_resolved_mask"] = str(mask_path)
        if index % 5000 == 0 or index == len(records):
            print(f"[content-audit] hashed {index}/{len(records)} records")
    return {
        "records": len(records),
        "unique_files": len(path_hash_cache),
        "bytes_read": int(total_bytes),
        "algorithm": "sha256",
    }


def _duplicate_groups(
    records: Sequence[Mapping[str, Any]], field: str
) -> List[List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    return [group for group in grouped.values() if len(group) > 1]


def _qc_exclude_cross_test_duplicates(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Keep official-test rows and exclude matching non-test content."""

    excluded_ids: set[str] = set()
    duplicate_audit: Dict[str, Any] = {
        "image_groups_crossing_official_test": [],
        "mask_only_groups_crossing_official_test": [],
    }

    for group in _duplicate_groups(records, "_image_sha256"):
        splits = {str(record["split"]) for record in group}
        if "test" not in splits or len(splits) == 1:
            continue
        keep = [record for record in group if record["split"] == "test"]
        drop = [record for record in group if record["split"] != "test"]
        excluded_ids.update(_sample_id(record) for record in drop)
        duplicate_audit["image_groups_crossing_official_test"].append(
            {
                "sha256": group[0]["_image_sha256"],
                "kept": [_sample_id(record) for record in keep],
                "excluded": [_sample_id(record) for record in drop],
                "original_splits": sorted(splits),
            }
        )

    remaining = [
        record for record in records if _sample_id(record) not in excluded_ids
    ]
    for group in _duplicate_groups(remaining, "_mask_sha256"):
        splits = {str(record["split"]) for record in group}
        image_hashes = {str(record["_image_sha256"]) for record in group}
        if (
            "test" not in splits
            or len(splits) == 1
            or len(image_hashes) == 1
        ):
            continue
        keep = [record for record in group if record["split"] == "test"]
        drop = [record for record in group if record["split"] != "test"]
        excluded_ids.update(_sample_id(record) for record in drop)
        duplicate_audit["mask_only_groups_crossing_official_test"].append(
            {
                "sha256": group[0]["_mask_sha256"],
                "kept": [_sample_id(record) for record in keep],
                "excluded": [_sample_id(record) for record in drop],
                "original_splits": sorted(splits),
            }
        )

    retained: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for record in records:
        if _sample_id(record) in excluded_ids:
            item = copy.deepcopy(record)
            item["exclusion_reason"] = (
                "exact image/mask content reused by official test"
            )
            excluded.append(item)
        else:
            retained.append(record)
    duplicate_audit["num_excluded"] = len(excluded)
    return retained, excluded, duplicate_audit


def _load_brain_map(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Brain PID mapping is absent: {path}")
    mapping: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            slice_id = int(row["slice_id"])
            if slice_id in mapping:
                raise RuntimeError(f"Duplicate brain slice ID {slice_id}")
            mapping[slice_id] = {
                "pid": str(row["pid"]).strip(),
                "fold": int(row["fold"]),
            }
    expected = set(range(1, 3065))
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise RuntimeError(
            f"Brain mapping must cover slices 1..3064; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if len({item["pid"] for item in mapping.values()}) != 233:
        raise RuntimeError("Brain mapping must contain exactly 233 patients")
    if {item["fold"] for item in mapping.values()} != {1, 2, 3, 4, 5}:
        raise RuntimeError("Brain mapping must contain folds 1..5")
    return mapping


def _record_image_stem(record: Mapping[str, Any]) -> str:
    image_name = str(record.get("image_name", "")).strip()
    if image_name:
        return Path(image_name).stem
    return Path(str(record["image_path"])).stem


def _assign_group_metadata(
    records: Sequence[Dict[str, Any]],
    brain_mapping: Mapping[int, Mapping[str, Any]],
) -> None:
    for record in records:
        modality = str(record["modality"])
        if modality == "brain_mri":
            slice_id = int(_record_image_stem(record))
            metadata = brain_mapping[slice_id]
            record["group_id"] = f"brain_pid:{metadata['pid']}"
            record["group_source"] = "Cheng cjdata.PID"
            record["brain_fold"] = int(metadata["fold"])
        elif modality == "lung_ct":
            stem = _record_image_stem(record)
            match = _CT_STUDY_RE.match(stem)
            if match is None:
                raise RuntimeError(f"Cannot parse lung-CT study ID: {stem}")
            record["group_id"] = f"lung_ct_study:{match.group('study')}"
            record["group_source"] = "filename_without_slice_suffix"
        else:
            record["group_id"] = (
                f"{modality}_case:{_record_image_stem(record)}"
            )
            record["group_source"] = "public_case_id_no_patient_metadata"


def _assign_partition_groups(
    records: Sequence[Dict[str, Any]],
) -> None:
    """Build leakage-safe groups from IDs and exact image content.

    Connected components are formed independently within each modality using
    both the recoverable patient/study/case ID and the exact image hash. This
    preserves patient/study separation while preventing differently named
    copies of the same image from crossing partition boundaries.
    """

    by_modality: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_modality[str(record["modality"])].append(record)

    for modality, modality_records in by_modality.items():
        parent = list(range(len(modality_records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        first_by_group: Dict[str, int] = {}
        first_by_image: Dict[str, int] = {}
        for index, record in enumerate(modality_records):
            group_id = str(record["group_id"])
            image_hash = str(record["_image_sha256"])
            if group_id in first_by_group:
                union(index, first_by_group[group_id])
            else:
                first_by_group[group_id] = index
            if image_hash in first_by_image:
                union(index, first_by_image[image_hash])
            else:
                first_by_image[image_hash] = index

        component_members: Dict[int, List[int]] = defaultdict(list)
        for index in range(len(modality_records)):
            component_members[find(index)].append(index)

        if modality == "brain_mri":
            source = "patient_and_exact_image_component"
        elif modality == "lung_ct":
            source = "study_and_exact_image_component"
        else:
            source = "case_and_exact_image_component"

        for member_indices in component_members.values():
            sample_ids = sorted(
                _sample_id(modality_records[index])
                for index in member_indices
            )
            digest = hashlib.sha256(
                "\n".join(sample_ids).encode("utf-8")
            ).hexdigest()[:16]
            partition_group_id = f"{modality}_partition:{digest}"
            for index in member_indices:
                record = modality_records[index]
                record["partition_group_id"] = partition_group_id
                record["partition_group_source"] = source


def _copy_with_split(
    records: Sequence[Mapping[str, Any]],
    split: str,
    *,
    derived: bool = False,
    note: str | None = None,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source in records:
        record = copy.deepcopy(source)
        current_split = str(record["split"])
        original_split = str(record.get("original_split", current_split))
        record["original_split"] = original_split
        record["split"] = split
        if derived or current_split != split:
            record["derived_split"] = True
        if note is not None:
            record["split_note"] = note
        output.append(record)
    return output


def _choose_groups_closest(
    records: Sequence[Mapping[str, Any]],
    target_fraction: float,
    seed: int,
    group_field: str = "partition_group_id",
) -> set[str]:
    """Select whole groups whose image count is closest to the target."""

    target = int(round(len(records) * float(target_fraction)))
    return _choose_groups_closest_to_count(
        records,
        target,
        seed,
        group_field=group_field,
    )


def _choose_groups_closest_to_count(
    records: Sequence[Mapping[str, Any]],
    target: int,
    seed: int,
    group_field: str = "partition_group_id",
    require_nonempty: bool = True,
) -> set[str]:
    """Select whole groups whose image count is closest to ``target``.

    This is the count-targeted form of the existing deterministic subset-sum
    selector.  Keeping the original fraction wrapper above means an invocation
    without a required parent manifest follows exactly the previous 5% path.
    """

    if target < 0:
        raise ValueError("target labeled image count cannot be negative")
    if not records:
        if target == 0 or not require_nonempty:
            return set()
        raise RuntimeError(
            f"Cannot select {target} labeled images from an empty pool"
        )
    if target == 0:
        return set()

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[group_field])].append(record)
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)

    # Keep only a backpointer per reachable count. Storing a full tuple for
    # every state is quadratic for modalities with many singleton cases.
    maximum_group_size = max(len(groups[group_id]) for group_id in ordered)
    limit = min(len(records), target + maximum_group_size)
    predecessor: Dict[int, Tuple[int, str] | None] = {0: None}
    for group_id in ordered:
        size = len(groups[group_id])
        additions: Dict[int, Tuple[int, str]] = {}
        for current_sum in tuple(predecessor):
            candidate = current_sum + size
            if (
                candidate <= limit
                and candidate not in predecessor
                and candidate not in additions
            ):
                additions[candidate] = (current_sum, group_id)
        predecessor.update(additions)
        if target in predecessor:
            break
    best_sum = min(
        predecessor,
        key=lambda value: (
            abs(value - target),
            value > target,
            abs(value),
        ),
    )
    selected: set[str] = set()
    cursor = best_sum
    while cursor:
        backpointer = predecessor[cursor]
        if backpointer is None:
            raise RuntimeError("Invalid subset-sum backpointer")
        cursor, group_id = backpointer
        selected.add(group_id)
    if not selected and require_nonempty:
        smallest = min(ordered, key=lambda key: len(groups[key]))
        selected = {smallest}
    return selected


def _load_required_labeled_manifest(
    path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Load and structurally validate a parent labeled manifest."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required labeled manifest is absent: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"Required labeled manifest must contain a JSON object: {path}"
        )
    records_value = payload.get("records")
    if not isinstance(records_value, list):
        raise RuntimeError(
            f"Required labeled manifest lacks a records list: {path}"
        )
    records: List[Dict[str, Any]] = []
    for index, value in enumerate(records_value):
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"Required manifest record {index} is not an object"
            )
        record = dict(value)
        for field in ("sample_id", "modality", "partition_group_id"):
            if not str(record.get(field, "")).strip():
                raise RuntimeError(
                    f"Required manifest record {index} lacks {field}"
                )
        modality = str(record["modality"])
        if modality not in MODALITIES:
            raise RuntimeError(
                f"Required manifest record {index} has unknown modality "
                f"{modality!r}"
            )
        subset = record.get("subset")
        if subset is not None and str(subset) != "labeled":
            raise RuntimeError(
                f"Required manifest record {_sample_id(record)} is not labeled"
            )
        records.append(record)
    _assert_unique(records, "required labeled manifest")

    group_modalities: Dict[str, str] = {}
    groups_by_modality: Dict[str, set[str]] = defaultdict(set)
    records_by_modality: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = str(record["partition_group_id"])
        modality = str(record["modality"])
        previous = group_modalities.setdefault(group_id, modality)
        if previous != modality:
            raise RuntimeError(
                "Required partition group occurs in multiple modalities: "
                f"{group_id!r} ({previous!r}, {modality!r})"
            )
        groups_by_modality[modality].add(group_id)
        records_by_modality[modality].append(record)

    metadata_value = payload.get("metadata", {})
    metadata = (
        dict(metadata_value)
        if isinstance(metadata_value, Mapping)
        else {}
    )
    absolute_path = path.resolve()
    return {
        "path": absolute_path,
        "records": records,
        "records_by_modality": {
            modality: records_by_modality.get(modality, [])
            for modality in MODALITIES
        },
        "groups_by_modality": {
            modality: sorted(groups_by_modality.get(modality, set()))
            for modality in MODALITIES
        },
        "metadata": metadata,
        "audit_metadata": {
            "enabled": True,
            "path_argument": str(path),
            "relative_to_output_dir": os.path.relpath(
                absolute_path, output_dir
            ),
            "sha256": _sha256_file(absolute_path),
            "parent_manifest": metadata.get("manifest"),
            "parent_labeled_fraction_target": metadata.get(
                "labeled_fraction_target"
            ),
            "required_records": len(records),
            "required_groups": len(group_modalities),
            "duplicate_required_records": 0,
            "cross_modality_required_groups": 0,
            "required_records_by_modality": _count_by_modality(records),
            "required_groups_by_modality": {
                modality: len(groups_by_modality.get(modality, set()))
                for modality in MODALITIES
            },
        },
    }


def _validate_required_groups_in_train(
    train_records: Sequence[Mapping[str, Any]],
    required_records: Sequence[Mapping[str, Any]],
    modality: str,
) -> Tuple[set[str], Dict[str, Any]]:
    """Verify that a parent selection is a whole-group subset of train."""

    current_by_group: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    current_by_sample: Dict[str, Mapping[str, Any]] = {}
    for record in train_records:
        group_id = str(record["partition_group_id"])
        current_by_group[group_id].append(record)
        current_by_sample[_sample_id(record)] = record

    required_by_group: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in required_records:
        if str(record["modality"]) != modality:
            raise RuntimeError(
                f"{modality}: required record has cross-modality value "
                f"{record['modality']!r}"
            )
        required_by_group[str(record["partition_group_id"])].append(record)

    required_groups = set(required_by_group)
    missing_groups = sorted(required_groups - set(current_by_group))
    required_sample_ids = {
        _sample_id(record) for record in required_records
    }
    missing_records = sorted(required_sample_ids - set(current_by_sample))
    group_mismatches: List[str] = []
    for record in required_records:
        current = current_by_sample.get(_sample_id(record))
        if (
            current is not None
            and str(current["partition_group_id"])
            != str(record["partition_group_id"])
        ):
            group_mismatches.append(_sample_id(record))

    partial_groups: List[str] = []
    for group_id in required_groups & set(current_by_group):
        current_ids = {
            _sample_id(record) for record in current_by_group[group_id]
        }
        required_ids = {
            _sample_id(record) for record in required_by_group[group_id]
        }
        if current_ids != required_ids:
            partial_groups.append(group_id)

    audit = {
        "enabled": True,
        "required_groups": len(required_groups),
        "required_records": len(required_records),
        "missing_required_groups": len(missing_groups),
        "missing_required_group_examples": missing_groups[:10],
        "missing_required_records": len(missing_records),
        "missing_required_record_examples": missing_records[:10],
        "group_id_mismatches": len(group_mismatches),
        "group_id_mismatch_examples": group_mismatches[:10],
        "partial_required_groups": len(partial_groups),
        "partial_required_group_examples": sorted(partial_groups)[:10],
    }
    if (
        missing_groups
        or missing_records
        or group_mismatches
        or partial_groups
    ):
        raise RuntimeError(
            f"{modality}: required labeled manifest is not a whole-group "
            f"subset of the current train partition: {audit}"
        )
    return required_groups, audit


def _derive_group_validation(
    records: Sequence[Mapping[str, Any]],
    fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_groups = _choose_groups_closest(records, fraction, seed)
    validation_source = [
        record
        for record in records
        if str(record["partition_group_id"]) in selected_groups
    ]
    train_source = [
        record
        for record in records
        if str(record["partition_group_id"]) not in selected_groups
    ]
    return (
        _copy_with_split(train_source, "train"),
        _copy_with_split(
            validation_source,
            "val",
            derived=True,
            note=(
                "partition-group validation derived from official train; "
                "patient/study/case groups and exact duplicate images stay "
                "together"
            ),
        ),
    )


def _derive_case_validation(
    records: Sequence[Mapping[str, Any]],
    fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return _derive_group_validation(records, fraction, seed)


def _partition_modality(
    modality: str,
    records: Sequence[Mapping[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if modality == "brain_mri":
        train = [record for record in records if int(record["brain_fold"]) >= 3]
        validation = [
            record for record in records if int(record["brain_fold"]) == 2
        ]
        test = [record for record in records if int(record["brain_fold"]) == 1]
        return (
            _copy_with_split(
                train,
                "train",
                derived=True,
                note="Cheng patient-disjoint folds 3-5",
            ),
            _copy_with_split(
                validation,
                "val",
                derived=True,
                note="Cheng patient-disjoint fold 2",
            ),
            _copy_with_split(
                test,
                "test",
                derived=True,
                note="Cheng patient-disjoint fold 1",
            ),
        )

    official_train = [
        record for record in records if record["split"] == "train"
    ]
    official_validation = [
        record for record in records if record["split"] == "val"
    ]
    official_test = [
        record for record in records if record["split"] == "test"
    ]

    if modality == "lung_ct":
        train, validation = _derive_group_validation(
            official_train,
            validation_fraction,
            _stable_modality_seed(seed, modality, purpose=1),
        )
        held_out = official_validation + official_test
        test = _copy_with_split(
            held_out,
            "test",
            derived=True,
            note=(
                "official val merged into test because both contain the same "
                "held-out studies"
            ),
        )
        return train, validation, test

    if modality in DERIVED_CASE_VALIDATION:
        if official_validation:
            raise RuntimeError(
                f"{modality} unexpectedly has official validation rows"
            )
        train, validation = _derive_case_validation(
            official_train,
            validation_fraction,
            _stable_modality_seed(seed, modality, purpose=1),
        )
        return train, validation, _copy_with_split(official_test, "test")

    if not official_validation:
        raise RuntimeError(f"{modality} lacks an official validation split")
    return (
        _copy_with_split(official_train, "train"),
        _copy_with_split(official_validation, "val"),
        _copy_with_split(official_test, "test"),
    )


def _split_labeled_unlabeled(
    train_records: Sequence[Mapping[str, Any]],
    modality: str,
    fraction: float,
    seed: int,
    required_records: Sequence[Mapping[str, Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selection_seed = _stable_modality_seed(seed, modality, purpose=2)
    target_images = int(round(len(train_records) * float(fraction)))
    required_records = list(required_records or ())
    if required_records:
        required_groups, nesting_audit = (
            _validate_required_groups_in_train(
                train_records,
                required_records,
                modality,
            )
        )
        current_by_group: Dict[
            str, List[Mapping[str, Any]]
        ] = defaultdict(list)
        for record in train_records:
            current_by_group[str(record["partition_group_id"])].append(
                record
            )
        required_images = sum(
            len(current_by_group[group_id])
            for group_id in required_groups
        )
        if required_images > target_images:
            raise RuntimeError(
                f"{modality}: required parent groups contain "
                f"{required_images} images, exceeding the requested target "
                f"of {target_images}; labeled fractions must increase "
                "monotonically"
            )
        remaining = [
            record
            for record in train_records
            if str(record["partition_group_id"]) not in required_groups
        ]
        additional_groups = _choose_groups_closest_to_count(
            remaining,
            target_images - required_images,
            selection_seed,
            require_nonempty=False,
        )
        selected_groups = required_groups | additional_groups
        nesting_audit.update(
            {
                "target_images": target_images,
                "required_images_in_current_train": required_images,
                "added_groups": len(additional_groups),
                "selected_groups_total": len(selected_groups),
            }
        )
    else:
        # Preserve the original deterministic path when no parent is given.
        selected_groups = _choose_groups_closest(
            train_records,
            fraction,
            selection_seed,
        )
        required_groups = set()
        nesting_audit = None
    labeled_source = [
        record
        for record in train_records
        if str(record["partition_group_id"]) in selected_groups
    ]
    unlabeled_source = [
        record
        for record in train_records
        if str(record["partition_group_id"]) not in selected_groups
    ]
    if modality == "brain_mri":
        selection_unit = "patient_and_exact_image_component"
    elif modality == "lung_ct":
        selection_unit = "study_and_exact_image_component"
    else:
        selection_unit = "case_and_exact_image_component"

    labeled = _copy_with_split(labeled_source, "train")
    unlabeled = _copy_with_split(unlabeled_source, "train")
    for record in labeled:
        record["subset"] = "labeled"
    for record in unlabeled:
        record["subset"] = "unlabeled"
    labeled_groups = {
        record["partition_group_id"] for record in labeled
    }
    unlabeled_groups = {
        record["partition_group_id"] for record in unlabeled
    }
    if labeled_groups & unlabeled_groups:
        raise RuntimeError(
            f"{modality}: labeled/unlabeled group overlap detected"
        )
    if nesting_audit is not None:
        missing_required_groups = required_groups - labeled_groups
        missing_required_records = {
            _sample_id(record) for record in required_records
        } - {_sample_id(record) for record in labeled}
        nesting_audit.update(
            {
                "missing_required_groups_after_selection": len(
                    missing_required_groups
                ),
                "missing_required_group_examples_after_selection": sorted(
                    missing_required_groups
                )[:10],
                "missing_required_records_after_selection": len(
                    missing_required_records
                ),
                "missing_required_record_examples_after_selection": sorted(
                    missing_required_records
                )[:10],
            }
        )
        if missing_required_groups or missing_required_records:
            raise RuntimeError(
                f"{modality}: nested labeled selection dropped parent "
                f"content: {nesting_audit}"
            )
    selection_metadata = {
        "selection_unit": selection_unit,
        "target_fraction": float(fraction),
        "actual_image_fraction": len(labeled) / max(len(train_records), 1),
        "labeled_images": len(labeled),
        "unlabeled_images": len(unlabeled),
        "labeled_groups": len(labeled_groups),
        "unlabeled_groups": len(unlabeled_groups),
    }
    if nesting_audit is not None:
        selection_metadata["target_images"] = target_images
        selection_metadata["nesting_audit"] = nesting_audit
    return labeled, unlabeled, selection_metadata


def _position_audit(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    invalid = [
        record
        for record in records
        if not parse_answer_positions(record.get("answer", ""))
    ]
    return {
        "total": len(records),
        "answer_parsed": len(records) - len(invalid),
        "answer_unparsed": len(invalid),
        "answer_parse_coverage": (
            (len(records) - len(invalid)) / max(len(records), 1)
        ),
        "unparsed_sample_ids": [_sample_id(record) for record in invalid],
        "source": "answer only; position column is audit-only and never a fallback",
    }


def _strip_private(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not str(key).startswith("_")
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_manifest(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    _write_json(
        path,
        {
            "metadata": dict(metadata),
            "records": [_strip_private(record) for record in records],
        },
    )


def _backup_existing_manifests(output_dir: Path) -> None:
    existing = [
        output_dir / name
        for name in (
            "train_labeled_5pct.json",
            "train_unlabeled_5pct.json",
            "val.json",
            "test.json",
            "all_splits.json",
            "summary.json",
        )
        if (output_dir / name).is_file()
    ]
    if not existing:
        return
    backup_dir = output_dir.parent / "manifests_original_official"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for source in existing:
        target = backup_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)


def _fraction_tag(fraction: float) -> str:
    percentage = 100.0 * float(fraction)
    if percentage.is_integer():
        return f"{int(percentage)}pct"
    return f"{percentage:g}pct".replace(".", "p")


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    """Build one deterministic PosMed partition and its leakage audit."""
    qa_dir = Path(args.qa_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    brain_map_path = Path(args.brain_pid_map).expanduser().resolve()
    required_manifest_value = getattr(
        args, "required_labeled_manifest", None
    )
    required_context = None
    if required_manifest_value:
        required_context = _load_required_labeled_manifest(
            Path(required_manifest_value).expanduser(),
            output_dir,
        )
        parent_fraction = required_context["metadata"].get(
            "labeled_fraction_target"
        )
        if parent_fraction is not None:
            try:
                parent_fraction_value = float(parent_fraction)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "Required labeled manifest has a non-numeric "
                    "labeled_fraction_target"
                ) from error
            if parent_fraction_value >= float(args.labeled_fraction):
                raise RuntimeError(
                    "Required labeled manifest must have a lower target "
                    f"fraction ({parent_fraction_value:g}) than the requested "
                    f"fraction ({float(args.labeled_fraction):g})"
                )
    _backup_existing_manifests(output_dir)

    all_records: List[Dict[str, Any]] = []
    for modality in MODALITIES:
        path = qa_dir / OFFICIAL_QA_FILENAMES[modality]
        if not path.is_file():
            raise FileNotFoundError(path)
        modality_records = read_posmed_qa_csv(path, modality=modality)
        # Persist a portable provenance path instead of the machine-local QA
        # directory used to generate the release manifests.
        for record in modality_records:
            record["source_csv"] = str(Path("annotations/qa") / path.name)
        all_records.extend(modality_records)
    _assert_unique(all_records, "official QA rows")

    brain_mapping = _load_brain_map(brain_map_path)
    _assign_group_metadata(all_records, brain_mapping)
    content_audit = _attach_content_hashes(all_records, data_root)
    retained, excluded, duplicate_audit = (
        _qc_exclude_cross_test_duplicates(all_records)
    )
    _assign_partition_groups(retained)

    by_modality: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in retained:
        by_modality[str(record["modality"])].append(record)

    labeled: List[Dict[str, Any]] = []
    unlabeled: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    label_selection: Dict[str, Any] = {}
    partition_counts: Dict[str, Any] = {}
    for modality in MODALITIES:
        train_records, val_records, test_records = _partition_modality(
            modality,
            by_modality[modality],
            args.derived_val_fraction,
            args.seed,
        )
        modality_labeled, modality_unlabeled, selection = (
            _split_labeled_unlabeled(
                train_records,
                modality,
                args.labeled_fraction,
                args.seed,
                required_records=(
                    required_context["records_by_modality"][modality]
                    if required_context is not None
                    else None
                ),
            )
        )
        labeled.extend(modality_labeled)
        unlabeled.extend(modality_unlabeled)
        validation.extend(val_records)
        test.extend(test_records)
        label_selection[modality] = selection
        partition_counts[modality] = {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        }

    named_splits = {
        "train_labeled": labeled,
        "train_unlabeled": unlabeled,
        "val": validation,
        "test": test,
    }
    for name, records in named_splits.items():
        _assert_unique(records, name)
    sample_overlap = _assert_pairwise_disjoint(
        named_splits, _sample_id, "sample-ID"
    )
    image_overlap = _assert_pairwise_disjoint(
        named_splits,
        lambda record: str(record["_image_sha256"]),
        "exact-image-content",
    )
    mask_overlap = _pairwise_overlap_audit(
        named_splits,
        lambda record: str(record["_mask_sha256"]),
    )
    group_overlap = _assert_pairwise_disjoint(
        named_splits,
        lambda record: str(record["group_id"]),
        "recoverable patient/study/case group",
    )
    partition_group_overlap = _assert_pairwise_disjoint(
        named_splits,
        lambda record: str(record["partition_group_id"]),
        "partition group",
    )

    used = labeled + unlabeled + validation + test
    if len(used) + len(excluded) != len(all_records):
        raise RuntimeError(
            "Used and QC-excluded records do not account for official rows"
        )
    used_ids = {_sample_id(record) for record in used}
    excluded_ids = {_sample_id(record) for record in excluded}
    if used_ids & excluded_ids:
        raise RuntimeError("QC exclusions still occur in a final split")

    if required_context is not None:
        required_records = required_context["records"]
        required_record_ids = {
            _sample_id(record) for record in required_records
        }
        required_group_ids = {
            str(record["partition_group_id"])
            for record in required_records
        }
        labeled_record_ids = {
            _sample_id(record) for record in labeled
        }
        labeled_group_ids = {
            str(record["partition_group_id"]) for record in labeled
        }
        missing_required_record_ids = sorted(
            required_record_ids - labeled_record_ids
        )
        missing_required_group_ids = sorted(
            required_group_ids - labeled_group_ids
        )
        nested_labeled_audit = {
            **required_context["audit_metadata"],
            "missing_required_records": len(
                missing_required_record_ids
            ),
            "missing_required_record_examples": (
                missing_required_record_ids[:10]
            ),
            "missing_required_groups": len(
                missing_required_group_ids
            ),
            "missing_required_group_examples": (
                missing_required_group_ids[:10]
            ),
            "retained_required_records": len(
                required_record_ids & labeled_record_ids
            ),
            "retained_required_groups": len(
                required_group_ids & labeled_group_ids
            ),
            "strictly_nested": not (
                missing_required_record_ids
                or missing_required_group_ids
            ),
        }
        if (
            missing_required_record_ids
            or missing_required_group_ids
        ):
            raise RuntimeError(
                "Final labeled manifest is not nested in its required "
                f"parent: {nested_labeled_audit}"
            )
    else:
        nested_labeled_audit = None

    position_audit = _position_audit(used)
    if args.strict_position and position_audit["answer_unparsed"]:
        raise RuntimeError(
            "Some answer texts lack a recoverable five-zone cue: "
            f"{position_audit['unparsed_sample_ids']}"
        )

    tag = _fraction_tag(args.labeled_fraction)
    split_protocol = {
        "brain_mri": (
            "Original Cheng patient-disjoint fold 1 test, fold 2 val, "
            "folds 3-5 train; PID/fold mapping frozen offline."
        ),
        "lung_ct": (
            "Official val merged into held-out test because both use the same "
            "40 studies; new val derived from official-train study IDs."
        ),
        "breast_ultrasound": "Official case-disjoint train/val/test.",
        "lung_xray": "Official case-disjoint train/val/test.",
        "polyp_endoscopy": (
            "Official test retained; exact-image-linked case-group val derived "
            "from train because video/patient mapping is unavailable."
        ),
        "skin_dermoscopy": (
            "Official test retained; exact-image-linked case-group val derived "
            "from train because patient mapping is unavailable; cross-test "
            "content reused rows excluded."
        ),
    }
    common_metadata: Dict[str, Any] = {
        "dataset": "PosMed/PRS-Med",
        "text_field": "answer",
        "position_label_source": "answer",
        "position_column_usage": "offline_audit_only",
        "modalities": list(MODALITIES),
        "seed": int(args.seed),
        "labeled_fraction_target": float(args.labeled_fraction),
        "derived_val_fraction": float(args.derived_val_fraction),
        "split_protocol": split_protocol,
        "label_selection": label_selection,
        "partition_counts": partition_counts,
        "position_audit": position_audit,
        "content_audit": content_audit,
        "duplicate_qc": duplicate_audit,
        "overlap_audit": {
            "sample_id": sample_overlap,
            "image_sha256": image_overlap,
            "mask_sha256_audit_only": mask_overlap,
            "recoverable_group": group_overlap,
            "partition_group": partition_group_overlap,
        },
        "group_limitations": (
            "Brain MRI and lung CT are patient/study-disjoint. Public files for "
            "breast US, lung X-ray, polyp, and skin expose case IDs but not "
            "patient IDs; those splits are case/content-disjoint only. Polyp "
            "train/test share Kvasir and ClinicDB sources."
        ),
        "brain_pid_map": {
            "path": str(Path("annotations") / brain_map_path.name),
            "sha256": _sha256_file(brain_map_path),
            "patients": 233,
            "source": (
                "Cheng/Figshare cjdata.PID with folds validated against "
                "official cvind.mat"
            ),
        },
    }
    if nested_labeled_audit is not None:
        common_metadata["nested_labeled_selection"] = (
            nested_labeled_audit
        )

    output_paths = {
        "train_labeled": output_dir / f"train_labeled_{tag}.json",
        "train_unlabeled": output_dir / f"train_unlabeled_{tag}.json",
        "val": output_dir / "val.json",
        "test": output_dir / "test.json",
        "all_splits": output_dir / "all_splits.json",
        "excluded_qc": output_dir / "excluded_qc.json",
    }
    for name in SPLIT_NAMES:
        records = named_splits[name]
        _write_manifest(
            output_paths[name],
            records,
            {
                **common_metadata,
                "manifest": name,
                "num_records": len(records),
                "counts_by_modality": _count_by_modality(records),
            },
        )
    _write_manifest(
        output_paths["all_splits"],
        used,
        {
            **common_metadata,
            "manifest": "all_splits",
            "num_records": len(used),
            "counts_by_subset": {
                name: len(records) for name, records in named_splits.items()
            },
            "counts_by_modality": _count_by_modality(used),
        },
    )
    _write_manifest(
        output_paths["excluded_qc"],
        excluded,
        {
            **common_metadata,
            "manifest": "excluded_qc",
            "num_records": len(excluded),
            "counts_by_modality": _count_by_modality(excluded),
        },
    )

    summary = {
        "output_paths": {
            name: path.name for name, path in output_paths.items()
        },
        "official_rows": len(all_records),
        "used_rows": len(used),
        "qc_excluded_rows": len(excluded),
        "counts": {
            name: {
                "total": len(records),
                "by_modality": _count_by_modality(records),
            }
            for name, records in {
                **named_splits,
                "excluded_qc": excluded,
            }.items()
        },
        "label_selection": label_selection,
        "partition_counts": partition_counts,
        "position_audit": position_audit,
        "duplicate_qc": duplicate_audit,
        "overlap_audit": common_metadata["overlap_audit"],
        "content_audit": content_audit,
    }
    if nested_labeled_audit is not None:
        summary["nested_labeled_selection"] = nested_labeled_audit
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = summary_path.name
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Define portable PosMed preprocessing and split-generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", default=str(DEFAULT_QA_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--brain-pid-map", default=str(DEFAULT_BRAIN_PID_MAP)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-fraction", type=float, default=0.05)
    parser.add_argument(
        "--required-labeled-manifest",
        default=None,
        help=(
            "Optional lower-ratio train_labeled manifest whose complete "
            "partition groups must be retained. Use the 5%% manifest when "
            "building 15%%, then the 15%% manifest when building 25%%."
        ),
    )
    parser.add_argument(
        "--derived-val-fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--strict-position",
        action="store_true",
        help="Fail if any answer lacks a recoverable five-zone position.",
    )
    return parser


def main() -> None:
    """Generate PosMed manifests and print a concise audit summary."""
    args = build_parser().parse_args()
    if not (0.0 < args.labeled_fraction < 1.0):
        raise ValueError("--labeled-fraction must be in (0, 1)")
    if not (0.0 < args.derived_val_fraction < 1.0):
        raise ValueError("--derived-val-fraction must be in (0, 1)")
    summary = prepare(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
