"""DETR — DEtection TRansformer.

Architecture:
    Backbone (ResNet-50 or MobileNetV2) → 1×1 conv projection
    → Sinusoidal 2D positional encoding
    → Transformer encoder (6 layers) → Transformer decoder (6 layers)
    → Classification head + Bounding box MLP head

Reference: "End-to-End Object Detection with Transformers" (Carion et al., 2020)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)


class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding for spatial feature maps."""

    def __init__(self, d_model: int = 256, temperature: float = 10000.0,
                 max_size: int = 200):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        # Pre-compute encodings for max spatial size
        pe = torch.zeros(d_model, max_size, max_size)
        half_d = d_model // 2

        y_pos = torch.arange(max_size).unsqueeze(1).repeat(1, max_size).float()
        x_pos = torch.arange(max_size).unsqueeze(0).repeat(max_size, 1).float()

        dim = torch.arange(half_d).float()
        div_term = temperature ** (2 * (dim // 2) / half_d)

        pe[:half_d:2, :, :] = torch.sin(x_pos.unsqueeze(0) / div_term[::2].view(-1, 1, 1))
        pe[1:half_d:2, :, :] = torch.cos(x_pos.unsqueeze(0) / div_term[1::2].view(-1, 1, 1))
        pe[half_d::2, :, :] = torch.sin(y_pos.unsqueeze(0) / div_term[::2].view(-1, 1, 1))
        pe[half_d + 1::2, :, :] = torch.cos(y_pos.unsqueeze(0) / div_term[1::2].view(-1, 1, 1))

        self.register_buffer("pe", pe)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: (B, H, W) boolean mask (True = padding)

        Returns:
            pos: (B, d_model, H, W) positional encoding
        """
        _, h, w = mask.shape
        pos = self.pe[:, :h, :w].unsqueeze(0)  # (1, d_model, H, W)
        return pos.expand(mask.shape[0], -1, -1, -1)


class MLP(nn.Module):
    """Simple multi-layer perceptron (used for bbox prediction)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


BACKBONES = ["resnet50", "mobilenet_v2"]


class DETR(nn.Module):
    """DEtection TRansformer for object detection.

    Args:
        num_classes: Number of object categories (excluding no-object)
        backbone_name: Backbone architecture ("resnet50" or "mobilenet_v2")
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Transformer encoder layers
        num_decoder_layers: Transformer decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        num_queries: Number of learned object queries
        pretrained_backbone: Use ImageNet-pretrained backbone weights
        aux_loss: Return intermediate decoder outputs for auxiliary losses
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone_name: str = "mobilenet_v2",
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 100,
        pretrained_backbone: bool = True,
        aux_loss: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # --- Backbone ---
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            backbone = resnet50(weights=weights)
            self.backbone = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            )
            backbone_out_channels = 2048
        elif backbone_name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            backbone = mobilenet_v2(weights=weights)
            self.backbone = backbone.features  # outputs 1280 channels
            backbone_out_channels = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {BACKBONES}")

        # --- Input projection ---
        self.input_proj = nn.Conv2d(backbone_out_channels, d_model, kernel_size=1)

        # --- Positional encoding ---
        self.pos_encoder = PositionalEncoding2D(d_model)

        # --- Transformer ---
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq, batch, d_model)
        )

        # --- Object queries ---
        self.query_embed = nn.Embedding(num_queries, d_model)

        # --- Prediction heads ---
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize projection and head parameters."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) batch of images
            masks:  (B, H, W) padding masks (True = padding)

        Returns:
            dict with:
                pred_logits: (B, num_queries, num_classes + 1)
                pred_boxes:  (B, num_queries, 4) in cxcywh [0,1]
        """
        B = images.shape[0]

        # Backbone features
        features = self.backbone(images)  # (B, 2048, H', W')

        # Project to d_model
        src = self.input_proj(features)  # (B, d_model, H', W')
        _, _, h, w = src.shape

        # Downsample mask to feature map size
        feat_mask = F.interpolate(
            masks.float().unsqueeze(1), size=(h, w), mode="nearest"
        ).squeeze(1).bool()  # (B, H', W')

        # Positional encoding
        pos = self.pos_encoder(feat_mask)  # (B, d_model, H', W')

        # Flatten spatial dims: (B, d_model, H'*W') → (H'*W', B, d_model)
        src_flat = src.flatten(2).permute(2, 0, 1)       # (S, B, d_model)
        pos_flat = pos.flatten(2).permute(2, 0, 1)       # (S, B, d_model)
        mask_flat = feat_mask.flatten(1)                  # (B, S)

        # Add positional encoding to source
        src_with_pos = src_flat + pos_flat

        # Object queries: (num_queries, B, d_model)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query)

        # Transformer
        hs = self.transformer(
            src=src_with_pos,
            tgt=tgt + query,
            src_key_padding_mask=mask_flat,
        )  # (num_queries, B, d_model)

        # Prediction heads
        hs = hs.permute(1, 0, 2)  # (B, num_queries, d_model)
        logits = self.class_head(hs)         # (B, num_queries, num_classes + 1)
        boxes = self.bbox_head(hs).sigmoid()  # (B, num_queries, 4)

        return {
            "pred_logits": logits,
            "pred_boxes": boxes,
        }


def build_detr(
    num_classes: int = 10,
    backbone_name: str = "mobilenet_v2",
    pretrained_backbone: bool = True,
    num_queries: int = 100,
    **kwargs,
) -> DETR:
    """Convenience builder for DETR."""
    return DETR(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        num_queries=num_queries,
        **kwargs,
    )
