"""Pure-visual six-region position predictors for ISPG Eq. (10).

The backbone deliberately contains only the ConvNeXt ``downsample_layers`` and
``stages`` from ``MMIUNet_V2``.  It never calls a Bridger or the text encoder,
which prevents position labels from leaking through the report.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import Block, LayerNorm


REGION_NAMES = (
    "left_upper",
    "left_middle",
    "left_lower",
    "right_upper",
    "right_middle",
    "right_lower",
)


class PureConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny arranged exactly like the visual part of MMIUNet_V2."""

    def __init__(self, drop_path_rate: float = 0.0):
        """Build the four ConvNeXt stages used by the segmentation backbone."""
        super().__init__()
        depths = (3, 3, 9, 3)
        dims = (96, 192, 384, 768)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cursor = 0
        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.stages.append(
                nn.Sequential(
                    *[
                        Block(
                            dim=dims[i],
                            drop_path=rates[cursor + j],
                            layer_scale_init_value=1e-6,
                        )
                        for j in range(depth)
                    ]
                )
            )
            cursor += depth

        self.out_channels = dims

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Return the four spatial feature scales for one image batch."""
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        features = []
        x = image
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features


class GAPMLPHead(nn.Module):
    """Predict six positions from globally pooled visual features."""

    def __init__(self, dim: int = 768, hidden_dim: int = 384, dropout: float = 0.2):
        """Construct the lightweight MLP used in the reported ISPG setting."""
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, features: List[torch.Tensor], return_attention: bool = False):
        """Map the deepest visual feature to six independent logits."""
        pooled = features[-1].mean(dim=(2, 3))
        logits = self.classifier(pooled)
        return (logits, None) if return_attention else logits


class SixRegionPoolHead(nn.Module):
    """Pool fixed left/right x upper/middle/lower regions from the feature map.

    Chest radiographs are conventionally displayed as if facing the patient,
    so anatomical left is image-right by default.
    """

    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 192,
        dropout: float = 0.2,
        anatomical_left_is_image_right: bool = True,
    ):
        super().__init__()
        self.anatomical_left_is_image_right = anatomical_left_is_image_right
        self.norm = nn.LayerNorm(dim)
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(6)
            ]
        )

    @staticmethod
    def _bounds(length: int, parts: int = 3) -> List[Tuple[int, int]]:
        edges = [round(i * length / parts) for i in range(parts + 1)]
        return [(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(parts)]

    def forward(self, features: List[torch.Tensor], return_attention: bool = False):
        feat = features[-1]
        _, _, height, width = feat.shape
        vertical = self._bounds(height)
        middle = width // 2
        image_left = (0, max(1, middle))
        image_right = (middle, width)
        left_bounds, right_bounds = (
            (image_right, image_left)
            if self.anatomical_left_is_image_right
            else (image_left, image_right)
        )

        pooled = []
        masks = []
        for side in (left_bounds, right_bounds):
            x0, x1 = side
            for y0, y1 in vertical:
                region = feat[:, :, y0:y1, x0:x1]
                pooled.append(region.mean(dim=(2, 3)))
                mask = feat.new_zeros((height, width))
                mask[y0:y1, x0:x1] = 1
                masks.append(mask)

        logits = torch.cat(
            [head(self.norm(vector)) for head, vector in zip(self.classifiers, pooled)],
            dim=1,
        )
        attention = torch.stack(masks, dim=0).unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
        return (logits, attention) if return_attention else logits


def _sincos_2d(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D sine/cosine position dimension must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4)))
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pos = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pos.to(dtype=dtype).unsqueeze(0)


class SixQueryHead(nn.Module):
    """Six location queries cross-attend to pure visual tokens."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(6, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.classifier_weight = nn.Parameter(torch.empty(6, dim))
        self.classifier_bias = nn.Parameter(torch.zeros(6))
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def forward(self, features: List[torch.Tensor], return_attention: bool = False):
        feat = features[-1]
        batch, channels, height, width = feat.shape
        memory = feat.flatten(2).transpose(1, 2)
        memory = memory + _sincos_2d(height, width, channels, feat.device, feat.dtype)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.norm(self.decoder(queries, memory))
        logits = torch.einsum("bqd,qd->bq", decoded, self.classifier_weight)
        logits = logits + self.classifier_bias

        attention = None
        if return_attention:
            scores = torch.einsum("bqd,bnd->bqn", decoded, memory) / math.sqrt(channels)
            attention = scores.softmax(dim=-1).reshape(batch, 6, height, width)
        return (logits, attention) if return_attention else logits


class PositionPredictor(nn.Module):
    """Image-only six-region predictor used by the optional ISPG extension."""

    def __init__(
        self,
        head_type: str,
        hidden_dim: int = 384,
        dropout: float = 0.2,
        query_layers: int = 2,
        query_heads: int = 8,
        anatomical_left_is_image_right: bool = True,
    ):
        """Select a visual head while sharing the same ConvNeXt backbone."""
        super().__init__()
        self.head_type = head_type.lower()
        self.backbone = PureConvNeXtTiny()
        if self.head_type == "gap_mlp":
            self.head = GAPMLPHead(768, hidden_dim, dropout)
        elif self.head_type == "region_pool":
            self.head = SixRegionPoolHead(
                768,
                hidden_dim,
                dropout,
                anatomical_left_is_image_right,
            )
        elif self.head_type == "six_query":
            self.head = SixQueryHead(768, query_heads, query_layers, dropout)
        else:
            raise ValueError(
                f"Unknown head_type={head_type!r}; choose gap_mlp, region_pool, or six_query"
            )

    def forward(self, image: torch.Tensor, return_attention: bool = False):
        """Predict six report-compatible position probabilities from images."""
        return self.head(self.backbone(image), return_attention=return_attention)

    def freeze_backbone(self) -> None:
        """Freeze all ConvNeXt parameters for head-only warm-up."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self, last_n_stages: int = 4) -> None:
        """Unfreeze the requested number of deepest ConvNeXt stages."""
        last_n_stages = max(0, min(4, int(last_n_stages)))
        if last_n_stages == 0:
            return
        first = 4 - last_n_stages
        for index in range(first, 4):
            for parameter in self.backbone.downsample_layers[index].parameters():
                parameter.requires_grad = True
            for parameter in self.backbone.stages[index].parameters():
                parameter.requires_grad = True


def _checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")


def load_semimedref_backbone(
    model: PositionPredictor,
    checkpoint_path: str,
    weights: str = "student",
) -> Dict[str, int]:
    """Load only pure ConvNeXt tensors from a Semi-MedRef checkpoint."""
    state = _checkpoint_state(checkpoint_path)
    prefixes = [f"{weights}.", f"model.{weights}.", ""]
    target = model.backbone.state_dict()
    loaded = {}
    for target_key, target_value in target.items():
        candidates = []
        for prefix in prefixes:
            candidates.extend(
                (
                    prefix + target_key,
                    prefix + "backbone." + target_key,
                )
            )
        for candidate in candidates:
            value = state.get(candidate)
            if value is not None and value.shape == target_value.shape:
                loaded[target_key] = value
                break
    missing = sorted(set(target) - set(loaded))
    model.backbone.load_state_dict(loaded, strict=False)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Only loaded {len(loaded)}/{len(target)} visual tensors; "
            f"first missing keys: {preview}"
        )
    return {"loaded": len(loaded), "total": len(target)}


def load_position_checkpoint(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> Tuple[PositionPredictor, Dict]:
    """Restore a selected position predictor and its saved metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["model_config"]
    model = PositionPredictor(**config)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
