"""PosMed/PRS-Med preprocessing and dataset adapters.

The public QA tables store the segmentation-mask path in ``image_path``.  This
module pairs that mask with the corresponding image, uses *only* ``answer`` as
the referring text, and derives PACL position labels from that same answer.
The auxiliary ``position`` column is retained for audit purposes, but is never
used as a fallback supervision signal.

The manifest helpers support the paper's ``Cross-domain Generalization``
experiment; the five-zone labels are the PosMed counterpart of the report-only
PACL supervision defined in Eq. (9).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandZoomd,
    Resized,
    ToTensord,
)
from torch.utils.data import Dataset, Sampler, Subset
from transformers import AutoTokenizer


DEFAULT_POSMED_ROOT = str(
    Path(__file__).resolve().parents[1] / "datasets/PosMed/data"
)

# The ordering is part of the optional 30-D modality-position label contract.
MODALITIES: Tuple[str, ...] = (
    "brain_mri",
    "breast_ultrasound",
    "lung_ct",
    "lung_xray",
    "polyp_endoscopy",
    "skin_dermoscopy",
)
SEMANTIC_CLASSES: Dict[str, Tuple[str, ...]] = {
    "brain_mri": ("tumor",),
    "breast_ultrasound": ("benign", "malignant", "unspecified_tumor"),
    "lung_ct": ("abnormality",),
    "lung_xray": (
        "normal",
        "opacity",
        "covid",
        "viral_pneumonia",
        "unspecified_anatomy",
    ),
    "polyp_endoscopy": ("polyp",),
    "skin_dermoscopy": ("benign", "malignant", "unspecified_lesion"),
}
MIX_GROUPS: Tuple[Tuple[str, str], ...] = tuple(
    (modality, semantic_class)
    for modality in MODALITIES
    for semantic_class in SEMANTIC_CLASSES[modality]
)
ZONES: Tuple[str, ...] = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
)

CSV_STEM_TO_MODALITY: Dict[str, str] = {
    "brain_tumors_ct_scan": "brain_mri",
    "breast_tumors_ct_scan": "breast_ultrasound",
    "lung_ct": "lung_ct",
    "lung_xray": "lung_xray",
    "polyp_endoscopy": "polyp_endoscopy",
    "skin_rgbimage": "skin_dermoscopy",
}

_CENTER_RE = re.compile(
    r"\b(?:center|centre|central|middle|centrally|centered|centred)\b",
    flags=re.IGNORECASE,
)
_VERTICAL_RE = {
    "top": re.compile(r"\b(?:top|upper)\b", flags=re.IGNORECASE),
    "bottom": re.compile(r"\b(?:bottom|lower)\b", flags=re.IGNORECASE),
}
_HORIZONTAL_RE = {
    "left": re.compile(r"\bleft\b", flags=re.IGNORECASE),
    "right": re.compile(r"\bright\b", flags=re.IGNORECASE),
}
_EXPLICIT_ZONE_RE = {
    "top_left": (
        re.compile(r"\b(?:top|upper)\s+left\b", flags=re.IGNORECASE),
        re.compile(r"\bleft\s+(?:top|upper)\b", flags=re.IGNORECASE),
    ),
    "top_right": (
        re.compile(r"\b(?:top|upper)\s+right\b", flags=re.IGNORECASE),
        re.compile(r"\bright\s+(?:top|upper)\b", flags=re.IGNORECASE),
    ),
    "bottom_left": (
        re.compile(r"\b(?:bottom|lower)\s+left\b", flags=re.IGNORECASE),
        re.compile(r"\bleft\s+(?:bottom|lower)\b", flags=re.IGNORECASE),
    ),
    "bottom_right": (
        re.compile(r"\b(?:bottom|lower)\s+right\b", flags=re.IGNORECASE),
        re.compile(r"\bright\s+(?:bottom|lower)\b", flags=re.IGNORECASE),
    ),
}


def _normalise_position_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("_", " ").replace("/", " ")
    value = re.sub(r"[\u2010-\u2015-]", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def parse_answer_positions(text: Any) -> List[str]:
    """Extract the five PosMed spatial zones from an input answer.

    Direct phrases such as ``upper right`` or ``left lower`` are preferred.
    Some PosMed lung-CT answers express a coarse constraint, for example
    ``the right lung ... one upper and one lower``.  In those cases this
    function truthfully returns both compatible zones.  It never consults the
    dataset's separate ``position`` annotation.
    """

    # Keep parsing identical to PosAug/T-PatchMix spatial utilities so that a
    # report cannot receive one label in the dataset and another in the
    # augmentation path.
    from utils.posmed_spatial import parse_posmed_position

    target, valid = parse_posmed_position("" if text is None else str(text))
    if not valid:
        return []
    return [zone for index, zone in enumerate(ZONES) if bool(target[index] > 0)]


def parse_answer_semantic_class(text: Any, modality: str) -> str:
    """Return a coarse target class using only the input answer text."""

    if modality not in MODALITIES:
        raise ValueError(f"Unknown PosMed modality: {modality!r}")
    answer = _normalise_position_text(text)
    if modality == "brain_mri":
        return "tumor"
    if modality == "lung_ct":
        return "abnormality"
    if modality == "polyp_endoscopy":
        return "polyp"
    if modality in {"breast_ultrasound", "skin_dermoscopy"}:
        if re.search(r"\bmalignan(?:t|cy)\b", answer):
            return "malignant"
        if re.search(r"\bbenign\b", answer):
            return "benign"
        return (
            "unspecified_tumor"
            if modality == "breast_ultrasound"
            else "unspecified_lesion"
        )
    if re.search(r"\bviral\s+pneumonia\b", answer):
        return "viral_pneumonia"
    if re.search(r"\bcovid(?:-?19)?\b", answer):
        return "covid"
    if re.search(r"\bopacity\b", answer):
        return "opacity"
    if re.search(r"\bnormal\b", answer):
        return "normal"
    return "unspecified_anatomy"


def mix_group_index(modality: str, semantic_class: str) -> int:
    key = (str(modality), str(semantic_class))
    try:
        return MIX_GROUPS.index(key)
    except ValueError as exc:
        raise ValueError(f"Unknown PosMed mix group: {key!r}") from exc


def parse_position_annotation(text: Any) -> List[str]:
    """Parse the audit-only ``position`` sentence into canonical zones."""

    return parse_answer_positions(text)


def positions_to_pseudo_label(
    positions: Sequence[str],
    modality: str,
    mode: str = "position",
) -> torch.Tensor:
    """Encode positions as a 5-D PACL label or a 30-D modality-aware label."""

    unknown = sorted(set(positions) - set(ZONES))
    if unknown:
        raise ValueError(f"Unknown PosMed position(s): {unknown}")
    if modality not in MODALITIES:
        raise ValueError(f"Unknown PosMed modality: {modality!r}")

    if mode == "position":
        label = torch.zeros(len(ZONES), dtype=torch.int64)
        offset = 0
    elif mode == "modality_position":
        label = torch.zeros(len(MODALITIES) * len(ZONES), dtype=torch.int64)
        offset = MODALITIES.index(modality) * len(ZONES)
    else:
        raise ValueError(
            "pseudo_label_mode must be 'position' or 'modality_position', "
            f"got {mode!r}"
        )

    for position in positions:
        label[offset + ZONES.index(position)] = 1
    return label


def _replace_mask_directory(mask_relative_path: str) -> str:
    parts = list(Path(mask_relative_path).parts)
    for index, part in enumerate(parts):
        if part.endswith("_masks"):
            parts[index] = f"{part[:-6]}_images"
            break
        if part.lower() == "masks":
            parts[index] = "images"
            break
    return str(Path(*parts))


def _candidate_image_names(record: Mapping[str, Any]) -> List[str]:
    mask_name = Path(str(record["image_path"])).name
    image_name = str(record.get("image_name", "")).strip()
    names: List[str] = []

    def add_with_extensions(name: str) -> None:
        if not name:
            return
        candidate = Path(name).name
        stem = Path(candidate).stem
        suffix = Path(candidate).suffix
        if suffix:
            names.append(candidate)
        else:
            for extension in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                names.append(f"{stem}{extension}")

    add_with_extensions(image_name)
    add_with_extensions(mask_name)
    mask_stem = Path(mask_name).stem
    for suffix in ("_Segmentation", "_segmentation", "_mask", "_Mask"):
        if mask_stem.endswith(suffix):
            add_with_extensions(mask_stem[: -len(suffix)])

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(names))


def posmed_pair_path_candidates(
    record: Mapping[str, Any],
    root_path: Union[str, os.PathLike[str]] = DEFAULT_POSMED_ROOT,
) -> Tuple[Path, List[Path]]:
    """Return the mask path and all plausible paired-image paths."""

    mask_value = Path(str(record["image_path"]))
    root = Path(root_path)
    mask_path = mask_value if mask_value.is_absolute() else root / mask_value

    image_relative = Path(_replace_mask_directory(str(mask_value)))
    image_directory = (
        image_relative.parent
        if mask_value.is_absolute()
        else (root / image_relative).parent
    )
    candidates = [image_directory / name for name in _candidate_image_names(record)]
    if not candidates:
        candidates = [root / image_relative]
    return mask_path, candidates


def resolve_posmed_pair_paths(
    record: Mapping[str, Any],
    root_path: Union[str, os.PathLike[str]] = DEFAULT_POSMED_ROOT,
    require_exists: bool = False,
) -> Tuple[str, str]:
    """Resolve a PosMed record to ``(image_path, mask_path)``."""

    mask_path, image_candidates = posmed_pair_path_candidates(record, root_path)
    image_path = next((path for path in image_candidates if path.is_file()), image_candidates[0])
    if require_exists:
        missing = []
        if not mask_path.is_file():
            missing.append(str(mask_path))
        if not image_path.is_file():
            missing.append(
                f"image for {record['image_path']} (tried "
                + ", ".join(str(path) for path in image_candidates)
                + ")"
            )
        if missing:
            raise FileNotFoundError("Missing PosMed pair: " + "; ".join(missing))
    return str(image_path), str(mask_path)


def _modality_from_csv(path: Union[str, os.PathLike[str]]) -> str:
    stem = Path(path).stem.lower()
    try:
        return CSV_STEM_TO_MODALITY[stem]
    except KeyError as exc:
        raise ValueError(
            f"Cannot infer PosMed modality from {path!s}; known stems are "
            f"{sorted(CSV_STEM_TO_MODALITY)}"
        ) from exc


def record_from_qa_row(
    row: Mapping[str, Any],
    modality: str,
    source_csv: Optional[str] = None,
    source_row: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert one official QA CSV row into the common manifest schema."""

    required = ("image_path", "answer", "split")
    missing = [column for column in required if not str(row.get(column, "")).strip()]
    if missing:
        raise ValueError(f"QA row is missing required field(s) {missing}: {dict(row)}")

    answer = str(row["answer"]).strip()
    canonical_positions = parse_answer_positions(answer)
    annotation_positions = parse_position_annotation(row.get("position", ""))
    mask_relative = str(row["image_path"]).strip()
    record: Dict[str, Any] = {
        "sample_id": f"{modality}:{mask_relative}",
        "image_path": mask_relative,
        "image_name": str(row.get("image_name", "")).strip(),
        "question": str(row.get("question", "")).strip(),
        "answer": answer,
        "position": str(row.get("position", "")).strip(),
        "split": str(row["split"]).strip().lower(),
        "modality": modality,
        "semantic_class": parse_answer_semantic_class(answer, modality),
        "canonical_positions": canonical_positions,
        "position_source": "answer",
        "position_valid": bool(canonical_positions),
        "annotation_positions": annotation_positions,
    }
    if source_csv is not None:
        record["source_csv"] = str(source_csv)
    if source_row is not None:
        record["source_row"] = int(source_row)
    return record


def read_posmed_qa_csv(
    path: Union[str, os.PathLike[str]],
    modality: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read one official natural-answer QA CSV."""

    csv_path = Path(path)
    inferred_modality = modality or _modality_from_csv(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            record_from_qa_row(
                row,
                modality=inferred_modality,
                source_csv=str(csv_path),
                source_row=row_index,
            )
            for row_index, row in enumerate(reader, start=2)
        ]
    return rows


def _load_manifest(path: Union[str, os.PathLike[str]]) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    else:
        raise ValueError(
            f"Manifest {path!s} must be a record list or contain a 'records' list"
        )
    return [dict(record) for record in records]


def _normalise_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    item = dict(record)
    modality = str(item.get("modality", "")).strip()
    if modality not in MODALITIES:
        raise ValueError(f"Invalid or missing modality in record: {item}")
    if not str(item.get("answer", "")).strip():
        raise ValueError(f"Invalid or missing answer in record: {item}")
    if not str(item.get("image_path", "")).strip():
        raise ValueError(f"Invalid or missing image_path in record: {item}")

    positions = item.get("canonical_positions")
    if positions is None:
        positions = parse_answer_positions(item["answer"])
    elif isinstance(positions, str):
        positions = [positions] if positions else []
    positions = [str(position).strip().lower().replace(" ", "_") for position in positions]
    item["canonical_positions"] = [zone for zone in ZONES if zone in positions]
    unknown = sorted(set(positions) - set(ZONES))
    if unknown:
        raise ValueError(f"Unknown canonical_positions {unknown} in record {item}")
    item["position_source"] = "answer"
    item["position_valid"] = bool(item["canonical_positions"])
    item["semantic_class"] = parse_answer_semantic_class(
        item["answer"], modality
    )
    item["split"] = str(item.get("split", "")).strip().lower()
    item.setdefault("sample_id", f"{modality}:{item['image_path']}")
    item.setdefault("annotation_positions", parse_position_annotation(item.get("position", "")))
    return item


def _ensure_three_channels(value: Any) -> Any:
    if not hasattr(value, "shape") or len(value.shape) != 3:
        return value
    channels = int(value.shape[0])
    if channels == 3:
        return value
    if channels == 1:
        if torch.is_tensor(value):
            return value.repeat(3, 1, 1)
        return value.repeat(3, axis=0)
    return value[:3]


class PosMed(Dataset):
    """Natural-answer PosMed dataset with the QaTa/MosMed tensor interface."""

    def __init__(
        self,
        csv_path: Optional[Union[str, os.PathLike[str], Sequence[Union[str, os.PathLike[str]]]]] = None,
        csv_paths: Optional[Sequence[Union[str, os.PathLike[str]]]] = None,
        manifest_path: Optional[Union[str, os.PathLike[str]]] = None,
        records: Optional[Sequence[Mapping[str, Any]]] = None,
        root_path: Union[str, os.PathLike[str]] = DEFAULT_POSMED_ROOT,
        tokenizer: Optional[Union[str, os.PathLike[str]]] = None,
        mode: str = "train",
        split: Optional[Union[str, Sequence[str]]] = None,
        image_size: Sequence[int] = (224, 224),
        max_txt_len: int = 48,
        pseudo_label_mode: str = "position",
        transform: Optional[Any] = None,
        validate_paths: bool = False,
    ) -> None:
        super().__init__()
        supplied = sum(
            value is not None
            for value in (csv_path, csv_paths, manifest_path, records)
        )
        if supplied != 1:
            raise ValueError(
                "Supply exactly one of csv_path, csv_paths, manifest_path, or records"
            )
        if tokenizer is None:
            raise ValueError("tokenizer must be a local Hugging Face tokenizer path")

        if csv_path is not None:
            paths = (
                list(csv_path)
                if isinstance(csv_path, Sequence) and not isinstance(csv_path, (str, os.PathLike))
                else [csv_path]
            )
            loaded = [
                record
                for path in paths
                for record in read_posmed_qa_csv(path)  # type: ignore[arg-type]
            ]
        elif csv_paths is not None:
            loaded = [
                record
                for path in csv_paths
                for record in read_posmed_qa_csv(path)
            ]
        elif manifest_path is not None:
            loaded = _load_manifest(manifest_path)
        else:
            loaded = [dict(record) for record in records or []]

        loaded = [_normalise_record(record) for record in loaded]
        requested_split: Optional[set[str]]
        if split is None:
            default_split = "val" if mode in ("valid", "validation") else mode
            requested_split = {default_split.lower()}
        elif isinstance(split, str):
            requested_split = None if split.lower() in ("all", "*") else {split.lower()}
        else:
            requested_split = {str(value).lower() for value in split}
        if requested_split is not None:
            loaded = [
                record for record in loaded if record.get("split") in requested_split
            ]

        sample_ids = [str(record["sample_id"]) for record in loaded]
        if len(sample_ids) != len(set(sample_ids)):
            duplicates = sorted(
                sample_id
                for sample_id in set(sample_ids)
                if sample_ids.count(sample_id) > 1
            )
            raise ValueError(f"Duplicate PosMed samples: {duplicates[:10]}")

        self.records = loaded
        self.mode = mode
        self.root_path = str(root_path)
        self.image_size = tuple(int(value) for value in image_size)
        self.max_txt_len = int(max_txt_len)
        self.pseudo_label_mode = pseudo_label_mode
        self._transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.image_list = [str(record["image_path"]) for record in self.records]
        self.caption_list = [str(record["answer"]) for record in self.records]
        self.modality_list = [str(record["modality"]) for record in self.records]
        self.position_label_list = [
            positions_to_pseudo_label(
                record["canonical_positions"],
                record["modality"],
                mode="position",
            )
            for record in self.records
        ]
        self.pseudo_label_list = [
            positions_to_pseudo_label(
                record["canonical_positions"],
                record["modality"],
                mode=self.pseudo_label_mode,
            )
            for record in self.records
        ]

        if validate_paths:
            for record in self.records:
                resolve_posmed_pair_paths(record, self.root_path, require_exists=True)
        print(
            f"{self.mode} set (PosMed, {self.pseudo_label_mode}): "
            f"{len(self.records)} samples"
        )

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer.encode_plus(
            caption,
            padding="max_length",
            max_length=self.max_txt_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def _text_payload(
        self,
        index: int,
        caption: str,
        token: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        record = self.records[index]
        modality = self.modality_list[index]
        return {
            "input_ids": token.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "pseudo_label": self.pseudo_label_list[index].clone(),
            "position_label": self.position_label_list[index].clone(),
            "modality": modality,
            "modality_index": torch.tensor(
                MODALITIES.index(modality), dtype=torch.int64
            ),
            "semantic_class": record["semantic_class"],
            "mix_group_index": torch.tensor(
                mix_group_index(modality, record["semantic_class"]),
                dtype=torch.int64,
            ),
            "position_valid": torch.tensor(
                bool(record["position_valid"]), dtype=torch.bool
            ),
            "pseudo_label_valid": torch.tensor(
                bool(record["position_valid"]), dtype=torch.bool
            ),
            "raw_text": caption,
        }

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path, mask_path = resolve_posmed_pair_paths(
            record,
            self.root_path,
            require_exists=True,
        )
        caption = self.caption_list[index]
        token_output = self._tokenize_caption(caption)
        data = {
            "image": image_path,
            "gt": mask_path,
            "token": token_output["input_ids"],
            "mask": token_output["attention_mask"],
        }
        data = self.transform(self.image_size)(data)

        image = torch.as_tensor(data["image"]).float()
        gt = torch.as_tensor(data["gt"]).float()
        token = torch.as_tensor(data["token"]).long()
        attention_mask = torch.as_tensor(data["mask"]).long()
        image = _ensure_three_channels(image)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        elif gt.ndim == 3 and gt.shape[0] > 1:
            gt = gt.max(dim=0, keepdim=True).values
        gt = (gt >= (128.0 if gt.max() > 1.5 else 0.5)).to(torch.int64)

        text = self._text_payload(
            index, caption, token, attention_mask
        )
        return [image, text], gt

    def transform(self, image_size: Sequence[int] = (224, 224)):
        if self._transform is not None:
            return self._transform
        common = [
            LoadImaged(["image", "gt"], reader="PILReader"),
            EnsureChannelFirstd(["image", "gt"]),
            Lambdad(["image"], func=_ensure_three_channels),
        ]
        if self.mode == "train":
            common.append(
                RandZoomd(
                    ["image", "gt"],
                    min_zoom=0.95,
                    max_zoom=1.2,
                    mode=["bicubic", "nearest"],
                    prob=0.1,
                )
            )
        common.extend(
            [
                Resized(["image"], spatial_size=image_size, mode="bicubic"),
                Resized(["gt"], spatial_size=image_size, mode="nearest"),
                NormalizeIntensityd(["image"], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ]
        )
        return Compose(common)


class PosMedSemiLabeled(PosMed):
    """PosMed labeled branch with an injectable weak/labeled transform."""

    def __init__(self, *args, labeled_tf: Optional[Any] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labeled_tf = labeled_tf

    def transform(self, image_size: Sequence[int] = (224, 224)):
        if self._labeled_tf is not None:
            return self._labeled_tf
        return super().transform(image_size)


class PosMedSemiUnlabeled(PosMed):
    """PosMed unlabeled branch compatible with Semi-MedRef's batch contract."""

    def __init__(
        self,
        *args,
        weak_tf: Any,
        strong_aug: Any,
        use_pos_aug: bool = False,
        pos_aug_p: float = 0.5,
        pos_aug_mode: str = "mix",
        **kwargs,
    ) -> None:
        if kwargs.get("mode", "train") != "train":
            raise ValueError("PosMedSemiUnlabeled is training-only")
        if not 0.0 <= float(pos_aug_p) <= 1.0:
            raise ValueError("pos_aug_p must be in [0, 1]")
        if pos_aug_mode not in ("mask", "fuzzy", "mix"):
            raise ValueError("pos_aug_mode must be 'mask', 'fuzzy', or 'mix'")
        validate_images = bool(kwargs.pop("validate_paths", False))
        super().__init__(*args, validate_paths=False, **kwargs)
        if validate_images:
            for record in self.records:
                image_path, _ = resolve_posmed_pair_paths(
                    record,
                    self.root_path,
                    require_exists=False,
                )
                if not Path(image_path).is_file():
                    raise FileNotFoundError(
                        f"Missing PosMed unlabeled image: {image_path}"
                    )
        self._weak_tf = weak_tf
        self._strong_aug = strong_aug
        self.use_pos_aug = bool(use_pos_aug)
        self.pos_aug_p = float(pos_aug_p)
        self.pos_aug_mode = str(pos_aug_mode)

    def transform(self, image_size: Sequence[int] = (224, 224)):
        return self._weak_tf

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image_path, _ = resolve_posmed_pair_paths(
            record,
            self.root_path,
            require_exists=False,
        )
        if not Path(image_path).is_file():
            raise FileNotFoundError(
                f"Missing PosMed unlabeled image: {image_path}"
            )
        caption = self.caption_list[index]
        token_output = self._tokenize_caption(caption)
        data = self._weak_tf(
            {
                "image": image_path,
                "token": token_output["input_ids"],
                "mask": token_output["attention_mask"],
            }
        )
        image_w = _ensure_three_channels(
            torch.as_tensor(data["image"]).float()
        )
        text_w = self._text_payload(
            index,
            caption,
            torch.as_tensor(data["token"]).long(),
            torch.as_tensor(data["mask"]).long(),
        )
        image_s = self._strong_aug(image_w.clone())
        text_s_str = caption
        if self.use_pos_aug:
            # Local import keeps the base data layer usable without importing
            # PosMed augmentation utilities in evaluation-only programs.
            from utils.posmed_spatial import posmed_position_dropout

            text_s_str = posmed_position_dropout(
                text_s_str,
                p_mask=self.pos_aug_p,
                mode=self.pos_aug_mode,
            )
        token_output = self.tokenizer.encode_plus(
            text_s_str,
            padding="max_length",
            max_length=self.max_txt_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_s = dict(text_w)
        text_s["input_ids"] = token_output["input_ids"].squeeze(0)
        text_s["attention_mask"] = token_output["attention_mask"].squeeze(0)
        text_s["raw_text"] = text_s_str
        modality = self.modality_list[index]
        return {
            "img_w": image_w,
            "img_s": image_s,
            "text": text_w,
            "text_w": text_w,
            "text_s": text_s,
            "text_s_str": text_s_str,
            "modality": modality,
            "modality_index": torch.tensor(
                MODALITIES.index(modality), dtype=torch.int64
            ),
            "mix_group_index": text_w["mix_group_index"].clone(),
            "geom": {"hflip": False, "vflip": False},
        }


class PosMedEval(PosMed):
    """Named alias for validation/test construction."""


def _dataset_modalities(dataset: Dataset) -> List[str]:
    if hasattr(dataset, "modality_list"):
        return list(getattr(dataset, "modality_list"))
    if isinstance(dataset, Subset):
        parent = _dataset_modalities(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError(
        "ModalityBatchSampler requires a dataset with modality_list or a Subset of one"
    )


class ModalityBatchSampler(Sampler[List[int]]):
    """Yield homogeneous-modality batches for safe within-modality T-PatchMix."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_modality: Dict[str, List[int]] = defaultdict(list)
        for index, modality in enumerate(_dataset_modalities(dataset)):
            if modality not in MODALITIES:
                raise ValueError(f"Unknown modality at dataset index {index}: {modality}")
            self.indices_by_modality[modality].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for modality in MODALITIES:
            indices = list(self.indices_by_modality.get(modality, ()))
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self.indices_by_modality.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.indices_by_modality.values()
        )


class BalancedModalityBatchSampler(Sampler[List[int]]):
    """Build mixed batches containing an equal number from all six modalities.

    Smaller modalities are cycled with replacement only after their shuffled
    pool is exhausted.  This makes a batch of 24 contain four examples from
    each modality while retaining deterministic ``set_epoch`` behaviour.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_batches: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0 or batch_size % len(MODALITIES) != 0:
            raise ValueError(
                f"batch_size must be positive and divisible by {len(MODALITIES)}"
            )
        self.batch_size = int(batch_size)
        self.per_modality = self.batch_size // len(MODALITIES)
        self.seed = int(seed)
        self.epoch = 0
        self.num_batches = (
            int(num_batches)
            if num_batches is not None
            else math.ceil(len(dataset) / self.batch_size)
        )
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")

        self.indices_by_modality: Dict[str, List[int]] = defaultdict(list)
        for index, modality in enumerate(_dataset_modalities(dataset)):
            if modality not in MODALITIES:
                raise ValueError(f"Unknown modality at dataset index {index}: {modality}")
            self.indices_by_modality[modality].append(index)
        empty = [modality for modality in MODALITIES if not self.indices_by_modality[modality]]
        if empty:
            raise ValueError(
                "BalancedModalityBatchSampler requires every modality; missing "
                + ", ".join(empty)
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _take_cyclic(
        pool: List[int],
        pointer: int,
        count: int,
        rng: random.Random,
    ) -> Tuple[List[int], int]:
        selected: List[int] = []
        while len(selected) < count:
            if pointer >= len(pool):
                rng.shuffle(pool)
                pointer = 0
            take = min(count - len(selected), len(pool) - pointer)
            selected.extend(pool[pointer : pointer + take])
            pointer += take
        return selected, pointer

    def __iter__(self) -> Iterator[List[int]]:
        modality_rngs = {
            modality: random.Random(
                self.seed + self.epoch * 1_000_003 + MODALITIES.index(modality) * 10_007
            )
            for modality in MODALITIES
        }
        pools = {
            modality: list(self.indices_by_modality[modality])
            for modality in MODALITIES
        }
        pointers = {modality: 0 for modality in MODALITIES}
        for modality in MODALITIES:
            modality_rngs[modality].shuffle(pools[modality])
        batch_rng = random.Random(self.seed + self.epoch * 1_000_003 + 97)

        for _ in range(self.num_batches):
            batch: List[int] = []
            for modality in MODALITIES:
                selected, pointers[modality] = self._take_cyclic(
                    pools[modality],
                    pointers[modality],
                    self.per_modality,
                    modality_rngs[modality],
                )
                batch.extend(selected)
            batch_rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


__all__ = [
    "CSV_STEM_TO_MODALITY",
    "DEFAULT_POSMED_ROOT",
    "MIX_GROUPS",
    "MODALITIES",
    "SEMANTIC_CLASSES",
    "ZONES",
    "BalancedModalityBatchSampler",
    "ModalityBatchSampler",
    "PosMed",
    "PosMedEval",
    "PosMedSemiLabeled",
    "PosMedSemiUnlabeled",
    "parse_answer_positions",
    "parse_answer_semantic_class",
    "parse_position_annotation",
    "mix_group_index",
    "positions_to_pseudo_label",
    "posmed_pair_path_candidates",
    "read_posmed_qa_csv",
    "record_from_qa_row",
    "resolve_posmed_pair_paths",
]
