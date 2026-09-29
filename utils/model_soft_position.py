"""MMI-UNet with the differentiable ISPG position token from Eq. (10).

The original model is left untouched.  ``position_vector`` is an optional
``(B, 6)`` tensor stored in the text dictionary.  It is projected to one
language token and participates in every Bridger stage.  The same vector also
conditions the contrastive text projection, allowing segmentation and PACL
gradients to reach an image-based position predictor when probabilities are
used instead of report labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat

from utils.model import MMIUNet_V2
from utils.layers import GuideDecoder


class MMIUNetSoftPosition(MMIUNet_V2):
    """Inject the differentiable six-region ISPG token into MMI-UNet."""

    def __init__(
        self,
        bert_type: str,
        vision_type: str,
        project_dim: int = 768,
        enable_project_head: bool = False,
        position_hidden_dim: int = 192,
        position_gate_init: float = -2.0,
    ):
        """Create token and PACL projectors while retaining the base backbone."""
        super().__init__(
            bert_type=bert_type,
            vision_type=vision_type,
            project_dim=project_dim,
            enable_project_head=enable_project_head,
        )
        self.position_token_projector = nn.Sequential(
            nn.Linear(6, position_hidden_dim),
            nn.GELU(),
            nn.Linear(position_hidden_dim, 768),
            nn.LayerNorm(768),
        )
        self.position_contrast_projector = nn.Sequential(
            nn.Linear(6, position_hidden_dim),
            nn.GELU(),
            nn.Linear(position_hidden_dim, project_dim),
            nn.LayerNorm(project_dim),
        )
        self.position_token_gate = nn.Parameter(torch.tensor(float(position_gate_init)))
        self.position_contrast_gate = nn.Parameter(torch.tensor(float(position_gate_init)))

    def use_guide_decoder(self) -> None:
        """Replace the convolutional decoder with the original GuideDecoder."""
        self.decoder16 = GuideDecoder(768, 384, 7, 24, input_text_len=25)
        self.decoder8 = GuideDecoder(384, 192, 14, 12, input_text_len=25)
        self.decoder4 = GuideDecoder(192, 96, 28, 9, input_text_len=25)
        del self.decode4
        del self.decode3
        del self.decode2

    def _position_vector(self, text, batch: int, device, dtype) -> torch.Tensor:
        vector = text.get("position_vector")
        if vector is None:
            return torch.zeros((batch, 6), device=device, dtype=dtype)
        return vector.to(device=device, dtype=dtype)

    def forward(self, data, return_project: bool = False):
        """Append Eq. (10)'s projected soft token before every fusion stage."""
        encoder_feats = []
        image, text = data
        if image.shape[1] == 1:
            image = repeat(image, "b 1 h w -> b c h w", c=3)

        text_output = self.text_encoder(text["input_ids"], text["attention_mask"])
        text_embeds, text_project = text_output["feature"], text_output["project"]
        txt = text_embeds[-1]
        position = self._position_vector(text, image.shape[0], image.device, txt.dtype)
        token_gate = torch.sigmoid(self.position_token_gate)
        position_token = self.position_token_projector(position).unsqueeze(1)
        txt = torch.cat((txt, token_gate * position_token), dim=1)

        contrast_gate = torch.sigmoid(self.position_contrast_gate)
        text_project = text_project + contrast_gate * self.position_contrast_projector(position)

        x = self.downsample_layers[0](image)
        x = self.stages[0](x)
        residual = x
        vis_feat, txt_feat = self.fusion1(x, txt)
        encoder_feats.append(vis_feat)

        x = self.downsample_layers[1](vis_feat + residual)
        x = self.stages[1](x)
        residual = x
        vis_feat, txt_feat = self.fusion2(x, txt + txt_feat)
        encoder_feats.append(vis_feat)

        x = self.downsample_layers[2](vis_feat + residual)
        x = self.stages[2](x)
        residual = x
        vis_feat, txt_feat = self.fusion3(x, txt + txt_feat)
        encoder_feats.append(vis_feat)

        x = self.downsample_layers[3](vis_feat + residual)
        x = self.stages[3](x)
        vis_feat, txt_feat = self.fusion4(x, txt + txt_feat)
        encoder_feats.append(vis_feat)

        if hasattr(self, "decoder16"):
            flattened = [
                rearrange(feature, "b c h w -> b (h w) c")
                for feature in encoder_feats
            ]
            d4 = self.decoder16(flattened[3], flattened[2], txt)
            d3 = self.decoder8(d4, flattened[1], txt)
            d2 = self.decoder4(d3, flattened[0], txt)
            d2 = rearrange(d2, "b (h w) c -> b c h w", h=56, w=56)
        else:
            d4 = self.decode4(encoder_feats[3], encoder_feats[2])
            d3 = self.decode3(d4, encoder_feats[1])
            d2 = self.decode2(d3, encoder_feats[0])
        output = self.out(self.decoder1(d2))

        if return_project and self.enable_project_head:
            image_gap = vis_feat.mean(dim=(2, 3))
            image_project = self.img_project_head(image_gap)
            return output, image_project, text_project
        return output
