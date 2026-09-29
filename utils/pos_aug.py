"""Position-language augmentation and T-PatchMix utilities.

``pos_token_dropout`` implements PosAug from the ``Multi-modal Augmentation``
section.  The region samplers and structured span updates implement
T-PatchMix Eqs. (6)--(8): the same accepted patch is applied to the strong
image, teacher target, and referring position span.
"""

import re
import random
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Position-phrase vocabulary and canonical matcher.
SIDE = ("left", "right")
VERT = ("upper", "middle", "lower")

# Valid phrases combine vertical region(s), laterality, and ``lung``.
_COMBO = r"(?:upper\s+middle\s+lower|upper\s+middle|middle\s+lower|upper\s+lower|upper|middle|lower)"
POS_PHRASE_RE = re.compile(rf"\b{_COMBO}\s+(left|right)\s+lung\b", flags=re.IGNORECASE)

def normalize_pos_phrase(s: str) -> Optional[str]:
    """Canonicalize a valid phrase, e.g. ``Lower LEFT lung``.

    Returns ``None`` when no valid position expression is present.
    """
    m = POS_PHRASE_RE.search(s)
    if not m:
        return None
    phrase = re.sub(r"\s+", " ", m.group(0).strip().lower())
    toks = phrase.split()
    vert = [t for t in toks if t in VERT]
    side = [t for t in toks if t in SIDE]
    if not side:
        return None
    vert = [v for v in VERT if v in vert]  # Fixed anatomical ordering.
    return " ".join(vert + side + ["lung"])

def geosync_map_no_norm(text: str, hflip: bool = False, vflip: bool = False) -> str:
    """Synchronize matched position phrases with geometric image flips.

    Horizontal flips exchange left/right. Vertical flips exchange upper/lower
    while preserving middle; vertical flipping is disabled in the main recipe.
    """
    def repl(m: re.Match) -> str:
        orig = m.group(0)
        toks = orig.split()

        def swap_side(tok: str) -> str:
            lo = tok.lower()
            if lo == "left":  return "right" if tok.islower() else "Right"
            if lo == "right": return "left"  if tok.islower() else "Left"
            return tok

        def swap_vert(tok: str) -> str:
            lo = tok.lower()
            if lo == "upper":  return "lower" if tok.islower() else "Lower"
            if lo == "lower":  return "upper" if tok.islower() else "Upper"
            return tok  # ``middle`` remains unchanged.

        out = []
        for t in toks:
            t2 = t
            if hflip: t2 = swap_side(t2)
            if vflip: t2 = swap_vert(t2)
            out.append(t2)
        return " ".join(out)
    return POS_PHRASE_RE.sub(repl, text)

def pos_token_dropout(text: str, p_mask: float = 0.4, mode: str = "mix") -> str:
    """Apply PosAug to canonical position phrases in a report.

    ``mask`` replaces a phrase with ``[UNK_POS]``; ``fuzzy`` replaces it with
    ``a region of the lung``; and ``mix`` samples either operation. PosAug is
    applied only to the student's strong report view in the main protocol, as
    described in the paper's PosAug paragraph.
    """
    assert 0.0 <= p_mask <= 1.0
    def repl(m: re.Match) -> str:
        phrase = normalize_pos_phrase(m.group(0))
        if phrase is None or random.random() > p_mask:
            return phrase or m.group(0)
        if mode == "mask":
            return "[UNK_POS]"
        if mode == "fuzzy":
            return "a region of the lung"
        # mix
        return "[UNK_POS]" if random.random() < 0.5 else "a region of the lung"

    text_norm = POS_PHRASE_RE.sub(lambda m: normalize_pos_phrase(m.group(0)) or m.group(0), text)
    return POS_PHRASE_RE.sub(repl, text_norm)

# Geometric alignment helpers.
def compose_affine(A_ref: torch.Tensor, A_src: torch.Tensor) -> torch.Tensor:
    """Compose batched affine matrices to map ``src`` into ``ref`` space."""
    B = A_ref.size(0)
    device = A_ref.device
    dtype = A_ref.dtype

    # Convert 2x3 affine matrices to homogeneous 3x3 matrices.
    def to33(A):
        pad = torch.zeros((B, 3, 3), device=device, dtype=dtype)
        pad[:, 0, 0] = 1.0
        pad[:, 1, 1] = 1.0
        pad[:, 2, 2] = 1.0
        pad[:, :2, :3] = A
        return pad

    Aref33 = to33(A_ref)
    Asrc33 = to33(A_src)
    A33 = torch.bmm(Aref33, torch.inverse(Asrc33))
    return A33[:, :2, :]

def warp_to_ref(prob_src: torch.Tensor, A_src_to_ref: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Warp batched probabilities/logits into a reference image frame."""
    B = prob_src.size(0)
    H_ref, W_ref = size_hw
    theta = A_src_to_ref
    grid = F.affine_grid(theta, size=(B, 1, H_ref, W_ref), align_corners=False)
    return F.grid_sample(prob_src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

# Optional auxiliary position head retained for ablation experiments.
class PositionHead(nn.Module):
    """Predict discrete position labels from a fused feature map."""
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feat: torch.Tensor):
        # feat: [B, C, H, W]
        x = self.pool(feat).flatten(1)
        return self.fc(x)


def span_mix(a: str, b: str) -> str:
    """Replace the position phrase in ``a`` with the phrase from ``b``."""
    ma = POS_PHRASE_RE.search(a)
    mb = POS_PHRASE_RE.search(b)
    if ma and mb:
        a_new = a[:ma.start()] + mb.group(0) + a[ma.end():]
        return a_new
    if not ma and mb:
        return b
    return a

def extract_pos_phrase(text: str) -> Optional[str]:
    """Extract and canonicalize one position phrase, if present."""
    m = POS_PHRASE_RE.search(text or "")
    if not m:
        return None
    return normalize_pos_phrase(m.group(0)) or None

def span_mix_dual(a: str, b: str) -> str:
    """Merge distinct position phrases from two reports."""
    pa = extract_pos_phrase(a)
    pb = extract_pos_phrase(b)
    if pa and pb:
        if pa == pb:
            return a
        def repl(m: re.Match) -> str:
            return f"{pa} and {pb}"
        return POS_PHRASE_RE.sub(repl, a, count=1)
    if pa:
        return a
    if pb:
        return b
    return a

def pos_phrase_to_region(phrase: str, H: int, W: int, margin: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
    """Map a position phrase to ``(y0, y1, x0, x1)`` image coordinates."""
    if not phrase:
        return None
    toks = phrase.lower().split()
    side = "left" if "left" in toks else ("right" if "right" in toks else None)
    verts = [v for v in VERT if v in toks]

    x0, x1 = 0, W
    if side == "left":
        x1 = W // 2
    elif side == "right":
        x0 = W // 2

    if not verts:
        y0, y1 = 0, H
    else:
        segs = []
        for v in verts:
            if v == "upper":
                segs.append((0, H // 3))
            elif v == "middle":
                segs.append((H // 3, (2 * H) // 3))
            elif v == "lower":
                segs.append(((2 * H) // 3, H))
        y0 = min(s[0] for s in segs)
        y1 = max(s[1] for s in segs)

    if margin > 0.0:
        mx = int(W * margin)
        my = int(H * margin)
        x0 = min(max(0, x0 + mx), W)
        x1 = max(min(W, x1 - mx), 0)
        y0 = min(max(0, y0 + my), H)
        y1 = max(min(H, y1 - my), 0)

    if x1 <= x0 or y1 <= y0:
        return None
    return (y0, y1, x0, x1)

def sample_block_mask_pos(B: int, H: int, W: int, regions, block: int = 64,
                          p: float = 1.0, device=None) -> torch.Tensor:
    """Sample the position-constrained T-PatchMix mask in Eq. (6)."""
    if device is None:
        device = torch.device("cpu")
    M = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    for i in range(B):
        if torch.rand(1).item() > p:
            continue
        region = regions[i] if regions is not None else None
        if not region:
            continue
        y0, y1, x0, x1 = region
        bh = min(block, max(1, y1 - y0))
        bw = min(block, max(1, x1 - x0))
        if (y1 - y0) < 1 or (x1 - x0) < 1:
            continue
        top = torch.randint(y0, y1 - bh + 1, (1,)).item()
        left = torch.randint(x0, x1 - bw + 1, (1,)).item()
        M[i, 0, top:top+bh, left:left+bw] = 1.0
    return M

def sample_block_mask_prob(t_prob: torch.Tensor, block: int = 64, p: float = 1.0,
                           thresh: float = 0.5) -> torch.Tensor:
    """Sample the probability-driven T-PatchMix candidate for Eqs. (6)--(7)."""
    B, _, H, W = t_prob.shape
    M = torch.zeros((B, 1, H, W), device=t_prob.device, dtype=torch.float32)
    for i in range(B):
        if torch.rand(1, device=t_prob.device).item() > p:
            continue
        mask = (t_prob[i, 0] >= thresh)
        if not mask.any():
            continue
        coords = mask.nonzero(as_tuple=False)
        sel = coords[torch.randint(0, coords.size(0), (1,), device=t_prob.device)].squeeze(0)
        cy = int(sel[0].item())
        cx = int(sel[1].item())
        bh = min(block, H)
        bw = min(block, W)
        top = min(max(cy - bh // 2, 0), H - bh)
        left = min(max(cx - bw // 2, 0), W - bw)
        M[i, 0, top:top+bh, left:left+bw] = 1.0
    return M

def sample_block_mask(B: int, H: int, W: int, block: int = 64, p: float = 1.0, device=None) -> torch.Tensor:
    """Generate random binary block masks with shape ``B x 1 x H x W``."""
    if device is None:
        device = torch.device("cpu")
    M = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    for i in range(B):
        if torch.rand(1).item() > p:
            continue
        bh = min(block, H)
        bw = min(block, W)
        top = torch.randint(0, H - bh + 1, (1,)).item()
        left = torch.randint(0, W - bw + 1, (1,)).item()
        M[i, 0, top:top+bh, left:left+bw] = 1.0
    return M

def tokenize_text_list(texts: List[str], tokenizer_name: str, max_len: int = 24, device=None):
    """Tokenize a report batch into input IDs and attention masks."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, local_files_only=True)
    batch = tok(
        texts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
    )
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


_NUM_WORD2INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM_INT2WORD = {v: k for k, v in _NUM_WORD2INT.items()}

# Match count expressions such as ``three infected areas``.
_COUNT_RE = re.compile(r"\b(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+infected\s+areas?\b", re.IGNORECASE)

# Match one or more vertical descriptors followed by laterality and ``lung``.
_LOC_RE = re.compile(
    r"\b(?:(?:upper|middle|lower)\s+){0,3}(?:left|right)\s+lung\b",
    re.IGNORECASE
)

def _parse_count(text: str):
    m = _COUNT_RE.search(text)
    if not m:
        return None
    s = m.group("num").lower()
    if s.isdigit():
        return int(s)
    return _NUM_WORD2INT.get(s, None)

def _render_count(n: int) -> str:
    if n in _NUM_INT2WORD:
        return _NUM_INT2WORD[n]
    return str(int(n))

def _parse_template_text(text: str):
    """Parse the templated disease/count/location report representation."""
    t = (text or "").strip()
    if len(t) == 0:
        return None

    disease = t.split(",")[0].strip()
    count = _parse_count(t)
    locs = [re.sub(r"\s+", " ", m.group(0).strip().lower()) for m in _LOC_RE.finditer(t)]

    # Deduplicate locations while retaining report order.
    seen = set()
    locs_uniq = []
    for x in locs:
        if x not in seen:
            seen.add(x)
            locs_uniq.append(x)

    # Fall back when neither a count nor a position can be parsed.
    if (count is None) and (len(locs_uniq) == 0):
        return None

    return {
        "disease": disease,
        "count": count,
        "locs": locs_uniq
    }

def structured_text_mix(text_i: str,
                        text_j: str,
                        ratio_j: float,
                        keep_i_ratio: float = 0.2,
                        mode: str = "union"):
    """Construct the report paired with a T-PatchMix image.

    ``ratio_j`` is the image fraction contributed by source ``j``. ``union``
    combines unique locations and updates the lesion count, ``keep_i`` is an
    ablation that preserves the first report, and ``choose`` samples a source
    report according to the mixed-area ratio.  This is the synchronized text
    update following the mixed pseudo-mask definition in Eq. (8).
    """
    if mode == "keep_i":
        return text_i

    if mode == "choose":
        # A larger mixed area makes source j more likely.
        import random
        return text_j if (random.random() < ratio_j) else text_i

    # The main setting uses union semantics.
    if ratio_j < keep_i_ratio:
        return text_i

    pi = _parse_template_text(text_i)
    pj = _parse_template_text(text_j)

    # Fall back to one intact report if template parsing fails.
    if (pi is None) or (pj is None):
        return text_i if ratio_j < 0.5 else text_j

    disease = pi["disease"] if pi.get("disease") else pj.get("disease", "")
    locs = []
    for x in (pi.get("locs", []) + pj.get("locs", [])):
        if x not in locs:
            locs.append(x)

    # Prefer a location-derived count for cross-modal consistency.
    if len(locs) > 0:
        count = len(locs)
    else:
        count = pi.get("count", None) or pj.get("count", None) or 1

    # Render the merged location phrase.
    if len(locs) == 1:
        loc_str = locs[0]
    else:
        loc_str = ", ".join(locs[:-1]) + " and " + locs[-1]

    out = f"{disease}, {_render_count(count)} infected areas, {loc_str}"
    return out
