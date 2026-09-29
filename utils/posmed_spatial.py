"""Spatial-language utilities for the PosMed generalisation experiment.

The PosMed QA files contain free-form answers whose spatial vocabulary differs
from the lung-specific phrases used by :mod:`utils.pos_aug`.  This module keeps
the PosMed rules separate so that the existing QaTa-COV19/MosMed behaviour is
unchanged.

The five PACL targets are ordered as::

    [top-left, top-right, bottom-left, bottom-right, center]

Only the free-form answer is parsed.  In particular, none of the helpers in
this module consumes PosMed's mask-derived ``position`` CSV column.

These deterministic five-zone targets provide the report-only affinity labels
used by PACL in Eq. (9); they are not segmentation annotations.
"""

from __future__ import annotations

import random
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


ZONE_NAMES: Tuple[str, ...] = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "center",
)
ZONE_TO_INDEX = {name: index for index, name in enumerate(ZONE_NAMES)}

_TOP = frozenset(("top", "upper"))
_BOTTOM = frozenset(("bottom", "lower"))
_LEFT = "left"
_RIGHT = "right"
_CENTER = frozenset(
    (
        "center",
        "centre",
        "central",
        "centrally",
        "centered",
        "centred",
    )
)
_VERTICAL = _TOP | _BOTTOM
_HORIZONTAL = frozenset((_LEFT, _RIGHT))
_DIRECTIONAL = _VERTICAL | _HORIZONTAL
_SPATIAL = _DIRECTIONAL | _CENTER

_WORD_RE = re.compile(r"[a-z]+", flags=re.IGNORECASE)
_SPATIAL_WORD_RE = re.compile(
    r"\b(?:top|upper|bottom|lower|left|right|center|centre|central|"
    r"centrally|centered|centred|middle)\b",
    flags=re.IGNORECASE,
)

# Tokens allowed between the two axes of descriptions such as
# "upper third of the left side" or "right side in the lower portion".
_PAIR_FILLERS = frozenset(
    (
        "third",
        "part",
        "portion",
        "section",
        "region",
        "area",
        "quadrant",
        "lobe",
        "side",
        "segment",
        "zone",
        "division",
        "of",
        "the",
        "on",
        "toward",
        "towards",
        "to",
        "in",
        "at",
        "lung",
        "image",
        "scan",
    )
)


def _canonical_vertical(word: str) -> str:
    return "top" if word.lower() in _TOP else "bottom"


def _canonical_horizontal(word: str) -> str:
    return word.lower()


def _zone_index(vertical: str, horizontal: str) -> int:
    return ZONE_TO_INDEX[f"{vertical}-{horizontal}"]


def _word_tokens(text: str):
    """Return ``(normalised_word, start, end)`` tokens for *text*."""

    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in _WORD_RE.finditer(text or "")
    ]


def _explicit_quadrants(tokens) -> Tuple[set[int], List[Tuple[int, int]]]:
    """Find two-axis phrases without pairing directions across clauses."""

    zones: set[int] = set()
    spans: List[Tuple[int, int]] = []

    for index, (word, start, end) in enumerate(tokens):
        if word not in _DIRECTIONAL:
            continue
        for other_index in range(index + 1, min(len(tokens), index + 8)):
            other, other_start, other_end = tokens[other_index]
            if other in _DIRECTIONAL:
                if (word in _VERTICAL) == (other in _VERTICAL):
                    # A second word on the same axis marks a coordinated phrase
                    # or the start of another location; do not jump over it.
                    break
                between = [token[0] for token in tokens[index + 1 : other_index]]
                if between and not all(token in _PAIR_FILLERS for token in between):
                    break
                vertical = word if word in _VERTICAL else other
                horizontal = word if word in _HORIZONTAL else other
                zones.add(
                    _zone_index(
                        _canonical_vertical(vertical),
                        _canonical_horizontal(horizontal),
                    )
                )
                spans.append((start, other_end))
                break
            if other not in _PAIR_FILLERS:
                break

    # Coordinated, shared-axis forms that the nearest-neighbour pass
    # intentionally does not jump across.
    normalised = " ".join(token[0] for token in tokens)
    vertical = r"(top|upper|bottom|lower)"
    horizontal = r"(left|right)"
    for match in re.finditer(
        rf"\b{vertical}\s+(?:and|or)\s+{vertical}\s+{horizontal}\b",
        normalised,
    ):
        for v_word in (match.group(1), match.group(2)):
            zones.add(
                _zone_index(
                    _canonical_vertical(v_word),
                    _canonical_horizontal(match.group(3)),
                )
            )
    for match in re.finditer(
        rf"\b{horizontal}\s+(?:and|or)\s+{horizontal}\s+{vertical}\b",
        normalised,
    ):
        for h_word in (match.group(1), match.group(2)):
            zones.add(
                _zone_index(
                    _canonical_vertical(match.group(3)),
                    _canonical_horizontal(h_word),
                )
            )
    # Shared-axis forms in which the common axis comes first, e.g.
    # "left upper and lower" or "upper left and right".
    for match in re.finditer(
        rf"\b{horizontal}\s+{vertical}\s+(?:and|or)\s+{vertical}\b",
        normalised,
    ):
        for v_word in (match.group(2), match.group(3)):
            zones.add(
                _zone_index(
                    _canonical_vertical(v_word),
                    _canonical_horizontal(match.group(1)),
                )
            )
    for match in re.finditer(
        rf"\b{vertical}\s+{horizontal}\s+(?:and|or)\s+{horizontal}\b",
        normalised,
    ):
        for h_word in (match.group(2), match.group(3)):
            zones.add(
                _zone_index(
                    _canonical_vertical(match.group(1)),
                    _canonical_horizontal(h_word),
                )
            )

    return zones, spans


def parse_posmed_position(
    answer: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, bool]:
    """Parse a PosMed free-form answer into a five-dimensional PACL target.

    Rules are intentionally coarse and deterministic:

    * explicit quadrants (including reversed forms such as ``left upper``)
      map to their corresponding quadrant;
    * multiple described anatomies produce a multi-hot target;
    * a lone ``left``/``right`` expands to both quadrants on that side;
    * a lone ``top``/``upper`` or ``bottom``/``lower`` expands to both
      quadrants in that half;
    * center words map to the center class only when no directional location
      is present.  Thus "top right, close to the center" remains top-right.

    Returns:
        ``(target, valid)`` where ``target`` has shape ``(5,)``.  An answer
        without recoverable spatial language receives an all-zero target and
        ``valid=False`` so PACL and T-PatchMix can exclude it safely.
    """

    tokens = _word_tokens(answer)
    words = [token[0] for token in tokens]
    zones, _ = _explicit_quadrants(tokens)

    verticals = {_canonical_vertical(word) for word in words if word in _VERTICAL}
    horizontals = {
        _canonical_horizontal(word) for word in words if word in _HORIZONTAL
    }

    if not zones:
        if verticals and horizontals:
            # This covers shared-axis descriptions such as "lower regions,
            # one on the left and the other on the right".
            for vertical in verticals:
                for horizontal in horizontals:
                    zones.add(_zone_index(vertical, horizontal))
        elif horizontals:
            for horizontal in horizontals:
                zones.add(_zone_index("top", horizontal))
                zones.add(_zone_index("bottom", horizontal))
        elif verticals:
            for vertical in verticals:
                zones.add(_zone_index(vertical, "left"))
                zones.add(_zone_index(vertical, "right"))

    # "middle" is a center synonym in PosMed answers.  A relative center
    # phrase accompanying a quadrant is intentionally ignored.
    has_center = any(word in _CENTER or word == "middle" for word in words)
    if not zones and has_center:
        zones.add(ZONE_TO_INDEX["center"])

    target = torch.zeros(len(ZONE_NAMES), dtype=dtype)
    if zones:
        target[list(sorted(zones))] = 1
    return target, bool(zones)


# Friendly aliases for dataset code and small analysis scripts.
encode_posmed_position = parse_posmed_position
parse_position_answer = parse_posmed_position


def position_vector_to_indices(
    position: Sequence[float] | torch.Tensor,
) -> List[int]:
    """Return active zone indices from a five-dimensional position vector."""

    values = torch.as_tensor(position).flatten()
    if values.numel() != len(ZONE_NAMES):
        raise ValueError(
            f"PosMed position vector must have {len(ZONE_NAMES)} values, "
            f"got {values.numel()}"
        )
    return torch.nonzero(values > 0, as_tuple=False).flatten().tolist()


def zone_to_region(
    zone: int | str,
    height: int,
    width: int,
    *,
    margin: float = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Map a PosMed zone to ``(y0, y1, x0, x1)`` image coordinates.

    Quadrants use image halves.  The center is the middle half on both axes,
    matching the deliberately coarse five-zone encoding.
    """

    if isinstance(zone, str):
        key = zone.strip().lower().replace("_", "-").replace(" ", "-")
        if key not in ZONE_TO_INDEX:
            return None
        zone_index = ZONE_TO_INDEX[key]
    else:
        zone_index = int(zone)
    if zone_index < 0 or zone_index >= len(ZONE_NAMES):
        return None
    if height <= 0 or width <= 0:
        return None

    half_h, half_w = height // 2, width // 2
    if zone_index == ZONE_TO_INDEX["top-left"]:
        y0, y1, x0, x1 = 0, half_h, 0, half_w
    elif zone_index == ZONE_TO_INDEX["top-right"]:
        y0, y1, x0, x1 = 0, half_h, half_w, width
    elif zone_index == ZONE_TO_INDEX["bottom-left"]:
        y0, y1, x0, x1 = half_h, height, 0, half_w
    elif zone_index == ZONE_TO_INDEX["bottom-right"]:
        y0, y1, x0, x1 = half_h, height, half_w, width
    else:
        y0, y1 = height // 4, height - height // 4
        x0, x1 = width // 4, width - width // 4

    if margin > 0:
        margin = min(float(margin), 0.49)
        dy = int(round((y1 - y0) * margin))
        dx = int(round((x1 - x0) * margin))
        y0, y1 = y0 + dy, y1 - dy
        x0, x1 = x0 + dx, x1 - dx
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def canonical_zone_phrase(zone: int | str) -> Optional[str]:
    """Return a compact phrase suitable for the mixed PosMed answer."""

    if isinstance(zone, str):
        key = zone.strip().lower().replace("_", "-").replace(" ", "-")
        index = ZONE_TO_INDEX.get(key)
    else:
        index = int(zone)
    if index is None or index < 0 or index >= len(ZONE_NAMES):
        return None
    if index == ZONE_TO_INDEX["center"]:
        return "central region"
    return f"{ZONE_NAMES[index].replace('-', ' ')} region"


def merge_donor_position(answer: str, donor_zone: int | str) -> str:
    """Append the accepted donor location to an anchor answer concisely."""

    phrase = canonical_zone_phrase(donor_zone)
    base = (answer or "").strip()
    if not phrase:
        return base
    if phrase in base.lower():
        return base
    base = base.rstrip(" .;")
    if not base:
        return f"An additional target is in the {phrase} of the image."
    return f"{base}. An additional target is in the {phrase}."


def _candidate_spans(text: str) -> List[Tuple[int, int]]:
    tokens = _word_tokens(text)
    _, paired_spans = _explicit_quadrants(tokens)
    spans = list(paired_spans)
    spans.extend((match.start(), match.end()) for match in _SPATIAL_WORD_RE.finditer(text))
    if not spans:
        return []

    # Prefer a compound span to its constituent word spans, then merge any
    # overlap.  Adjacent separate locations ("upper left and lower right")
    # remain separate because the conjunction is outside both spans.
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            continue
        merged.append((start, end))
    return merged


def extract_spatial_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return all PosMed spatial spans as ``(start, end, surface)`` tuples."""

    return [(start, end, text[start:end]) for start, end in _candidate_spans(text or "")]


def posmed_position_dropout(
    text: str,
    p_mask: float = 0.4,
    mode: str = "mix",
    *,
    rng: Optional[random.Random] = None,
) -> str:
    """Apply PosAug to free-form PosMed spatial spans.

    ``mode='mask'`` uses ``[UNK_POS]``; ``mode='fuzzy'`` uses
    ``a region of the image``; and ``mode='mix'`` samples either replacement.
    Each matched span is considered independently.
    """

    if not 0.0 <= float(p_mask) <= 1.0:
        raise ValueError(f"p_mask must lie in [0, 1], got {p_mask}")
    if mode not in {"mask", "fuzzy", "mix"}:
        raise ValueError(f"Unsupported PosAug mode: {mode}")
    generator = rng if rng is not None else random
    spans = _candidate_spans(text or "")
    if not spans:
        return text

    output: List[str] = []
    cursor = 0
    for start, end in spans:
        output.append(text[cursor:start])
        surface = text[start:end]
        if generator.random() > p_mask:
            output.append(surface)
        elif mode == "mask":
            output.append("[UNK_POS]")
        elif mode == "fuzzy":
            output.append("a region of the image")
        else:
            output.append(
                "[UNK_POS]"
                if generator.random() < 0.5
                else "a region of the image"
            )
        cursor = end
    output.append(text[cursor:])
    return "".join(output)


# Match the naming used by the original dataset implementation.
pos_token_dropout = posmed_position_dropout


def _case_preserving_swap(word: str, mapping: dict[str, str]) -> str:
    replacement = mapping.get(word.lower(), word)
    if word.isupper():
        return replacement.upper()
    if word[:1].isupper():
        return replacement.capitalize()
    return replacement


def geosync_map_posmed(
    text: str,
    *,
    hflip: bool = False,
    vflip: bool = False,
) -> str:
    """Invert PosMed direction words after horizontal/vertical image flips."""

    horizontal = {"left": "right", "right": "left"}
    vertical = {
        "top": "bottom",
        "upper": "lower",
        "bottom": "top",
        "lower": "upper",
    }

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if hflip and word.lower() in horizontal:
            word = _case_preserving_swap(word, horizontal)
        if vflip and word.lower() in vertical:
            word = _case_preserving_swap(word, vertical)
        return word

    return _SPATIAL_WORD_RE.sub(replace, text or "")


def transform_position_vector(
    position: Sequence[float] | torch.Tensor,
    *,
    hflip: bool = False,
    vflip: bool = False,
) -> torch.Tensor:
    """Apply image flips to a five-zone multi-hot position target."""

    source = torch.as_tensor(position)
    if source.shape[-1] != len(ZONE_NAMES):
        raise ValueError(
            f"Expected a final dimension of {len(ZONE_NAMES)}, "
            f"got {tuple(source.shape)}"
        )
    result = source.clone()
    if hflip:
        result[..., [0, 1]] = result[..., [1, 0]]
        result[..., [2, 3]] = result[..., [3, 2]]
    if vflip:
        result[..., [0, 2]] = result[..., [2, 0]]
        result[..., [1, 3]] = result[..., [3, 1]]
    return result


def batch_transform_position_vectors(
    positions: torch.Tensor,
    hflip: bool | torch.Tensor = False,
    vflip: bool | torch.Tensor = False,
) -> torch.Tensor:
    """Vectorised per-sample counterpart of :func:`transform_position_vector`."""

    result = torch.as_tensor(positions).clone()
    if result.ndim != 2 or result.shape[1] != len(ZONE_NAMES):
        raise ValueError(f"Expected positions with shape (B, 5), got {tuple(result.shape)}")
    batch_size = result.shape[0]
    device = result.device

    def flags(value) -> torch.Tensor:
        if torch.is_tensor(value):
            value = value.to(device=device, dtype=torch.bool).flatten()
            if value.numel() == 1:
                value = value.expand(batch_size)
            if value.numel() != batch_size:
                raise ValueError("Flip flag batch size does not match position batch")
            return value
        return torch.full((batch_size,), bool(value), device=device, dtype=torch.bool)

    h_flags, v_flags = flags(hflip), flags(vflip)
    if h_flags.any():
        original = result.clone()
        result[h_flags, 0] = original[h_flags, 1]
        result[h_flags, 1] = original[h_flags, 0]
        result[h_flags, 2] = original[h_flags, 3]
        result[h_flags, 3] = original[h_flags, 2]
    if v_flags.any():
        original = result.clone()
        result[v_flags, 0] = original[v_flags, 2]
        result[v_flags, 2] = original[v_flags, 0]
        result[v_flags, 1] = original[v_flags, 3]
        result[v_flags, 3] = original[v_flags, 1]
    return result


__all__ = [
    "ZONE_NAMES",
    "ZONE_TO_INDEX",
    "parse_posmed_position",
    "encode_posmed_position",
    "parse_position_answer",
    "position_vector_to_indices",
    "zone_to_region",
    "canonical_zone_phrase",
    "merge_donor_position",
    "extract_spatial_spans",
    "posmed_position_dropout",
    "pos_token_dropout",
    "geosync_map_posmed",
    "transform_position_vector",
    "batch_transform_position_vectors",
]
