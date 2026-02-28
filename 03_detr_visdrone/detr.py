"""DETR — DEtection TRansformer (Official Architecture).

Architecture matching facebookresearch/detr exactly for direct pretrained weight loading:
    FrozenBatchNorm2d ResNet-50 backbone (via IntermediateLayerGetter)
    → 1×1 conv projection
    → Sinusoidal 2D positional encoding (PositionEmbeddingSine)
    → Custom Transformer (encoder + decoder with positional embeddings in Q/K)
    → Classification head (class_embed) + Bounding box MLP head (bbox_embed)

Reference: "End-to-End Object Detection with Transformers" (Carion et al., 2020)
Key differences from nn.Transformer:
    1. Positional encodings are added to Q and K inside each attention layer,
       NOT to the input externally. This is crucial for DETR's design.
    2. Decoder returns intermediate outputs from all layers (for auxiliary loss).
    3. The encoder has NO final LayerNorm (post-norm variant).
"""

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter


# ---------------------------------------------------------------------------
# Frozen BatchNorm2d (batch stats fixed — used during fine-tuning)
# ---------------------------------------------------------------------------

class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and affine parameters are fixed.

    During fine-tuning, we don't want the backbone's batch norm stats to shift
    because the new dataset (VisDrone) has very different statistics than
    ImageNet. Freezing them keeps the backbone's learned representations stable.
    """

    def __init__(self, n: int):
        super().__init__()
        # Register as buffers (not parameters) so they're never updated
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Remove num_batches_tracked from state dict — we don't use it
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Reshape for broadcasting: (1, C, 1, 1)
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        # Fused scale + bias:  y = (x - mean) / sqrt(var + eps) * weight + bias
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# ---------------------------------------------------------------------------
# Backbone: ResNet-50 with FrozenBatchNorm2d
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """ResNet-50 backbone with frozen batch norm.

    Uses IntermediateLayerGetter to extract features from layer4 only,
    matching the official DETR structure where weights live under
    'backbone.0.body.layerN.*'.

    Freezing policy:
        - conv1, bn1, layer1: always frozen (low-level features, well-trained)
        - layer2, layer3, layer4: trainable (high-level features, adapted to new task)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load ResNet-50 with FrozenBatchNorm2d instead of regular BatchNorm
        backbone = torchvision.models.resnet50(
            replace_stride_with_dilation=[False, False, False],
            norm_layer=FrozenBatchNorm2d,
        )

        # Load ImageNet pretrained weights if requested
        if pretrained:
            state_dict = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.get_state_dict(
                progress=True
            )
            # The pretrained weights use regular BatchNorm, which has
            # 'num_batches_tracked'. FrozenBatchNorm2d handles this in
            # _load_from_state_dict by removing that key.
            backbone.load_state_dict(state_dict, strict=False)

        # Freeze early layers (conv1, bn1, layer1)
        for name, parameter in backbone.named_parameters():
            if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)

        # Only extract layer4 output (single-scale detection)
        self.body = IntermediateLayerGetter(backbone, return_layers={"layer4": "0"})
        self.num_channels = 2048  # ResNet-50 layer4 output channels

    def forward(self, images: Tensor, masks: Tensor):
        """
        Args:
            images: (B, 3, H, W) batch of images
            masks:  (B, H, W) padding masks (True = padding)

        Returns:
            features: (B, 2048, H', W') backbone features
            mask:     (B, H', W') downsampled mask
        """
        xs = self.body(images)
        features = xs["0"]

        # Downsample mask to match feature map spatial size
        mask = F.interpolate(
            masks.float().unsqueeze(1), size=features.shape[-2:]
        ).to(torch.bool).squeeze(1)

        return features, mask


# ---------------------------------------------------------------------------
# Positional Encoding: Sinusoidal 2D (matching official PositionEmbeddingSine)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D positional encoding for spatial feature maps.

    Unlike the pre-computed approach, this computes encodings dynamically
    based on the actual non-padded region (using cumsum on the mask).
    This handles variable-size inputs correctly.

    The encoding concatenates x and y sine/cosine embeddings:
        pos = [sin(y), cos(y), sin(x), cos(x)]  (each d_model/4 channels)
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0,
                 normalize: bool = True, scale: float = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor) -> Tensor:
        """
        Args:
            mask: (B, H, W) boolean mask (True = padding)

        Returns:
            pos: (B, d_model, H, W) positional encoding
        """
        not_mask = ~mask  # True = valid pixel

        # Cumulative sum gives position indices (1, 2, 3, ...)
        # Padding pixels get the same cumsum as the last valid pixel
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # (B, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # (B, H, W)

        if self.normalize:
            eps = 1e-6
            # Normalize to [0, 2π] based on the max position in each dimension
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Frequency bands: temperature^(2i/d) for i in [0, num_pos_feats)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # (B, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Interleave sin and cos: [sin(f0), cos(f0), sin(f1), cos(f1), ...]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # Concatenate y and x: (B, H, W, d_model) → (B, d_model, H, W)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# MLP (used for bounding box prediction)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron (used for bbox prediction head)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        )
        self.num_layers = num_layers

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Components (matching official DETR exactly)
# ---------------------------------------------------------------------------

def _get_clones(module, n):
    """Create n deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer: self-attention + FFN with residual connections.

    Key DETR difference from standard transformer:
        Positional encoding is added to Q and K only (not V).
        This means attention patterns are position-aware, but the value
        stream carries pure content features without position bias.

    Architecture (post-norm variant, normalize_before=False):
        1. q = k = src + pos_embed  (add position to queries and keys)
           v = src                  (values are pure content)
        2. src = src + dropout(self_attn(q, k, v))
        3. src = norm1(src)
        4. src = src + dropout(FFN(src))
        5. src = norm2(src)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """Add positional encoding to tensor (used for Q and K)."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src: Tensor,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None) -> Tensor:
        """Post-norm: attention → add & norm → FFN → add & norm."""
        # Self-attention with positional encoding on Q and K
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src: Tensor,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None) -> Tensor:
        """Pre-norm: norm → attention → add → norm → FFN → add."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer: self-attention + cross-attention + FFN.

    Has TWO attention mechanisms:
        1. Self-attention: queries attend to each other
           (query_pos added to Q and K, value = tgt content)
        2. Cross-attention: queries attend to encoder memory
           (query_pos added to Q, pos_embed added to K of memory)

    Architecture (post-norm):
        1. Self-attn(tgt + query_pos, tgt + query_pos, tgt) → norm1
        2. Cross-attn(tgt + query_pos, memory + pos, memory) → norm2
        3. FFN → norm3
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt: Tensor, memory: Tensor,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None) -> Tensor:
        # 1. Self-attention among object queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross-attention: queries attend to encoder memory
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt: Tensor, memory: Tensor,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


class TransformerEncoder(nn.Module):
    """Stack of N encoder layers.

    Note: The official DETR post-norm encoder has NO final LayerNorm
    (encoder_norm is None). This is different from the original
    "Attention is All You Need" paper which has a final norm.
    """

    def __init__(self, encoder_layer, num_layers: int, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Stack of N decoder layers with optional intermediate output collection.

    When return_intermediate=True, collects the output after each decoder
    layer (after norm). This enables auxiliary losses at every layer,
    which significantly helps DETR training convergence.
    """

    def __init__(self, decoder_layer, num_layers: int, norm=None,
                 return_intermediate: bool = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)  # (num_layers, Q, B, d_model)

        return output.unsqueeze(0)  # (1, Q, B, d_model)


class Transformer(nn.Module):
    """DETR Transformer: encoder + decoder.

    Key design: The Transformer handles all the reshaping internally.
    It receives spatial feature maps (B, C, H, W) and returns decoder
    outputs (num_layers, B, Q, d_model).

    This is different from nn.Transformer which expects pre-flattened
    sequence inputs and doesn't handle positional encodings.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False,
                 return_intermediate_dec: bool = False):
        super().__init__()

        # Build encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Build decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor,
                query_embed: Tensor, pos_embed: Tensor):
        """
        Args:
            src:         (B, C, H, W) projected backbone features
            mask:        (B, H, W)    padding mask (True = padding)
            query_embed: (Q, d_model) learned object query embeddings
            pos_embed:   (B, C, H, W) positional encoding

        Returns:
            hs:     (num_dec_layers, B, Q, d_model) decoder outputs
            memory: (B, C, H, W) encoder output reshaped back to spatial
        """
        bs, c, h, w = src.shape

        # Flatten spatial dims: (B, C, H, W) → (H*W, B, C)
        src = src.flatten(2).permute(2, 0, 1)        # (S, B, C) where S = H*W
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (S, B, C)
        mask = mask.flatten(1)                        # (B, S)

        # Expand query embeddings for batch: (Q, d_model) → (Q, B, d_model)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # Initialize decoder target as zeros (will be refined layer by layer)
        tgt = torch.zeros_like(query_embed)

        # Encode: spatial features → contextual memory
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # Decode: object queries attend to memory
        hs = self.decoder(tgt, memory,
                          memory_key_padding_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)

        # Reshape outputs:
        # hs: (num_layers, Q, B, C) → (num_layers, B, Q, C)
        # memory: (S, B, C) → (B, C, H, W)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


# ---------------------------------------------------------------------------
# DETR Model (top-level, matching official facebookresearch/detr)
# ---------------------------------------------------------------------------

class DETR(nn.Module):
    """DEtection TRansformer for object detection.

    This matches the official facebookresearch/detr architecture exactly,
    so pretrained weights can be loaded directly.

    State dict key structure:
        backbone.0.body.{conv1,bn1,layer1-4}.*  — ResNet-50 backbone
        input_proj.{weight,bias}                — 2048 → d_model projection
        query_embed.weight                      — learned object queries
        transformer.encoder.layers.N.*          — encoder weights
        transformer.decoder.layers.N.*          — decoder weights
        transformer.decoder.norm.*              — decoder output LayerNorm
        class_embed.{weight,bias}               — classification head
        bbox_embed.layers.N.{weight,bias}       — bbox regression MLP
    """

    def __init__(
        self,
        num_classes: int = 10,
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

        # --- Backbone (matches 'backbone.0.body.*' in official weights) ---
        backbone = Backbone(pretrained=pretrained_backbone)
        self.backbone = nn.Sequential(backbone)  # Wrapped in Sequential to match key prefix

        # --- Input projection ---
        self.input_proj = nn.Conv2d(backbone.num_channels, d_model, kernel_size=1)

        # --- Positional encoding (matches 'backbone.1.*' but has no learned params) ---
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

        # --- Transformer ---
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate_dec=aux_loss,
        )

        # --- Object queries ---
        self.query_embed = nn.Embedding(num_queries, d_model)

        # --- Prediction heads (names match official DETR) ---
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

    def forward(
        self,
        images: Tensor,
        masks: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            images: (B, 3, H, W) batch of images
            masks:  (B, H, W) padding masks (True = padding)

        Returns:
            dict with:
                pred_logits: (B, num_queries, num_classes + 1)
                pred_boxes:  (B, num_queries, 4) in cxcywh [0,1]
        """
        # Backbone features + downsampled mask
        features, mask = self.backbone[0](images, masks)

        # Positional encoding
        pos = self.pos_encoder(mask)

        # Project backbone features to transformer dimension
        src = self.input_proj(features)

        # Transformer: encode + decode
        hs, _ = self.transformer(src, mask, self.query_embed.weight, pos)
        # hs shape: (num_dec_layers, B, Q, d_model)

        # Prediction heads (applied to last decoder layer output)
        outputs_class = self.class_embed(hs)          # (L, B, Q, num_classes+1)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # (L, B, Q, 4)

        out = {
            "pred_logits": outputs_class[-1],  # (B, Q, num_classes+1)
            "pred_boxes": outputs_coord[-1],   # (B, Q, 4)
        }

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return out


def build_detr(
    num_classes: int = 10,
    pretrained_backbone: bool = True,
    num_queries: int = 100,
    aux_loss: bool = False,
    **kwargs,
) -> DETR:
    """Convenience builder for DETR."""
    return DETR(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        num_queries=num_queries,
        aux_loss=aux_loss,
        **kwargs,
    )
