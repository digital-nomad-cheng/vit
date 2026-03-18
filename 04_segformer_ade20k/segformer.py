"""
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
(NeurIPS 2021, Xie et al.)

Implementation of SegFormer-B0 from scratch.

Architecture:
  - Mix Transformer (MiT-B0) hierarchical encoder with 4 stages
  - Lightweight All-MLP decoder head
  - No positional encoding — replaced by Mix-FFN with depth-wise conv

Usage:
    from segformer import segformer_b0
    model = segformer_b0(num_classes=150)
    out = model(torch.randn(1, 3, 512, 512))  # (1, 150, 128, 128)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Overlapping patch embedding
# ---------------------------------------------------------------------------

class OverlapPatchEmbedding(nn.Module):
    """Overlapping patch embedding using Conv2d with kernel_size > stride.

    Produces 2D feature maps (B, C, H', W') from input (B, C_in, H, W).
    Preserves local continuity between patches.
    """

    def __init__(self, in_channels: int, embed_dim: int,
                 patch_size: int = 7, stride: int = 4):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, N, C') where N = H'*W'
            H', W': spatial dims after patch embedding
        """
        x = self.proj(x)                      # (B, C', H', W')
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C')
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Efficient self-attention
# ---------------------------------------------------------------------------

class EfficientSelfAttention(nn.Module):
    """Multi-head self-attention with sequence reduction.

    Reduces the K and V sequence length by factor `sr_ratio` using a
    strided Conv2d + LayerNorm, bringing complexity from O(N^2) down to
    O(N * N/R^2).
    """

    def __init__(self, dim: int, num_heads: int = 1, sr_ratio: int = 8,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Sequence reduction for K, V
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) where N = H * W
            H, W: spatial dimensions
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape

        # Query: (B, heads, N, head_dim)
        q = self.q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        # Reduce K, V sequence length
        if self.sr_ratio > 1:
            # Reshape to spatial, apply strided conv, flatten back
            x_sr = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            x_sr = self.sr(x_sr)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x

        k = rearrange(self.k(x_sr), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v(x_sr), 'b n (h d) -> b h n d', h=self.num_heads)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# Mix-FFN (replaces positional encoding)
# ---------------------------------------------------------------------------

class MixFFN(nn.Module):
    """Feed-forward network with 3x3 depth-wise convolution.

    The depth-wise conv acts as an implicit positional encoding,
    allowing the model to work at arbitrary resolutions without
    explicit positional embeddings.
    """

    def __init__(self, dim: int, expansion_ratio: int = 4, drop: float = 0.0):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) where N = H * W
            H, W: spatial dimensions
        Returns:
            (B, N, C)
        """
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Drop path (stochastic depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Create random tensor with shape (batch, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block: Attention + Mix-FFN with residuals."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int,
                 expansion_ratio: int = 4, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion_ratio=expansion_ratio, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


# ---------------------------------------------------------------------------
# Mix Transformer (MiT) encoder
# ---------------------------------------------------------------------------

class MiTEncoder(nn.Module):
    """Mix Transformer encoder with 4 hierarchical stages.

    Each stage: OverlapPatchEmbedding → N × TransformerBlock → LayerNorm
    Produces multi-scale features at 1/4, 1/8, 1/16, 1/32 of input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, ...] = (32, 64, 160, 256),
        num_heads: tuple[int, ...] = (1, 2, 5, 8),
        depths: tuple[int, ...] = (2, 2, 2, 2),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
        expansion_ratio: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_stages = len(embed_dims)

        # Stochastic depth: linearly increasing drop path rate
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        block_idx = 0
        for i in range(self.num_stages):
            # Patch embedding
            ch_in = in_channels if i == 0 else embed_dims[i - 1]
            patch_size = 7 if i == 0 else 3
            stride = 4 if i == 0 else 2

            self.patch_embeds.append(
                OverlapPatchEmbedding(ch_in, embed_dims[i], patch_size, stride)
            )

            # Transformer blocks for this stage
            stage_blocks = nn.ModuleList()
            for j in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        sr_ratio=sr_ratios[i],
                        expansion_ratio=expansion_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[block_idx],
                    )
                )
                block_idx += 1
            self.blocks.append(stage_blocks)
            self.norms.append(nn.LayerNorm(embed_dims[i]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            List of 4 feature maps: [(B, C_i, H_i, W_i) for i in 0..3]
            where H_i, W_i = H/(4*2^i), W/(4*2^i)
        """
        features = []

        for i in range(self.num_stages):
            x, H, W = self.patch_embeds[i](x)

            for block in self.blocks[i]:
                x = block(x, H, W)

            x = self.norms[i](x)

            # Reshape back to spatial for next stage or output
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            features.append(x)

        return features


# ---------------------------------------------------------------------------
# Lightweight All-MLP decoder head
# ---------------------------------------------------------------------------

class SegFormerHead(nn.Module):
    """Lightweight All-MLP Decoder.

    1. Projects each encoder feature to a common `decoder_dim`
    2. Upsamples all features to the highest resolution (1/4 of input)
    3. Concatenates along channel dimension
    4. Fuses with an MLP and produces per-pixel class logits
    """

    def __init__(
        self,
        encoder_dims: tuple[int, ...] = (32, 64, 160, 256),
        decoder_dim: int = 256,
        num_classes: int = 150,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = len(encoder_dims)

        # Per-stage linear projection (implemented as 1x1 conv)
        self.linear_projs = nn.ModuleList([
            nn.Conv2d(encoder_dims[i], decoder_dim, kernel_size=1)
            for i in range(self.num_stages)
        ])

        # Fusion: concat → reduce → classify
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * self.num_stages, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of 4 tensors from MiT encoder
                      [(B, C_i, H_i, W_i) for i in 0..3]
        Returns:
            logits: (B, num_classes, H_0, W_0) at 1/4 input resolution
        """
        target_size = features[0].shape[2:]  # highest resolution (1/4)

        projected = []
        for i, feat in enumerate(features):
            # Project to decoder_dim
            x = self.linear_projs[i](feat)
            # Upsample to target resolution
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear',
                                  align_corners=False)
            projected.append(x)

        # Concatenate and fuse
        x = torch.cat(projected, dim=1)    # (B, decoder_dim * 4, H_0, W_0)
        x = self.fuse(x)                   # (B, decoder_dim, H_0, W_0)
        x = self.dropout(x)
        x = self.classifier(x)             # (B, num_classes, H_0, W_0)
        return x


# ---------------------------------------------------------------------------
# Full SegFormer model
# ---------------------------------------------------------------------------

class SegFormer(nn.Module):
    """SegFormer: Hierarchical Transformer encoder + All-MLP decoder.

    Produces logits at 1/4 of input resolution. Use F.interpolate to upsample
    to full resolution for final predictions.
    """

    def __init__(
        self,
        encoder: MiTEncoder,
        decoder: SegFormerHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes, H/4, W/4)
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def segformer_b0(num_classes: int = 150, drop_path_rate: float = 0.1) -> SegFormer:
    """Build SegFormer-B0 (smallest variant, ~3.8M params)."""
    embed_dims = (32, 64, 160, 256)
    encoder = MiTEncoder(
        in_channels=3,
        embed_dims=embed_dims,
        num_heads=(1, 2, 5, 8),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        expansion_ratio=4,
        drop_path_rate=drop_path_rate,
    )
    decoder = SegFormerHead(
        encoder_dims=embed_dims,
        decoder_dim=256,
        num_classes=num_classes,
    )
    return SegFormer(encoder, decoder)


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

def load_pretrained_segformer_b0(
    model: SegFormer,
    checkpoint_path: str | None = None,
    num_classes: int = 150,
) -> SegFormer:
    """Load pretrained weights from HuggingFace SegFormer-B0 checkpoint.

    Downloads nvidia/segformer-b0-finetuned-ade-512-512 if no path given.
    Maps HuggingFace key names to our architecture.
    """
    if checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id="nvidia/segformer-b0-finetuned-ade-512-512",
                filename="pytorch_model.bin",
            )
        except ImportError:
            raise RuntimeError(
                "Install huggingface_hub: pip install huggingface_hub"
            )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Map HuggingFace keys → our keys
    mapped = {}
    for hf_key, value in state_dict.items():
        key = _map_hf_key(hf_key)
        if key is not None:
            mapped[key] = value

    # Reshape 2D linear weights → 4D conv weights for decoder projections
    # HF uses nn.Linear (shape [out, in]), we use nn.Conv2d 1x1 (shape [out, in, 1, 1])
    for key in mapped:
        if "decoder.linear_projs" in key and "weight" in key and mapped[key].ndim == 2:
            mapped[key] = mapped[key].unsqueeze(-1).unsqueeze(-1)

    # Load with strict=False to handle potential mismatches in classifier
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"[load_pretrained] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[load_pretrained] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    return model


def _map_hf_key(hf_key: str) -> str | None:
    """Map a HuggingFace SegFormer key to our model's key naming."""
    # Skip segformer. prefix
    if hf_key.startswith("segformer."):
        key = hf_key[len("segformer."):]
    elif hf_key.startswith("decode_head."):
        key = hf_key
    else:
        return None

    # --- Encoder mappings ---
    # HF: encoder.patch_embeddings.{i}.proj.weight
    # Ours: encoder.patch_embeds.{i}.proj.weight
    key = key.replace("encoder.patch_embeddings.", "encoder.patch_embeds.")

    # HF: encoder.patch_embeddings.{i}.layer_norm.weight
    # Ours: encoder.patch_embeds.{i}.norm.weight
    # NOTE: must apply SR norm mapping BEFORE the generic layer_norm→norm replacement
    key = key.replace(".attention.self.layer_norm.", ".attn.sr_norm.")
    key = key.replace("layer_norm.weight", "norm.weight")
    key = key.replace("layer_norm.bias", "norm.bias")

    # HF: encoder.block.{i}.{j}.attention.self.query / key / value
    # Ours: encoder.blocks.{i}.{j}.attn.q / k / v
    key = key.replace("encoder.block.", "encoder.blocks.")
    key = key.replace(".attention.self.query.", ".attn.q.")
    key = key.replace(".attention.self.key.", ".attn.k.")
    key = key.replace(".attention.self.value.", ".attn.v.")
    key = key.replace(".attention.self.sr.", ".attn.sr.")
    key = key.replace(".attention.output.dense.", ".attn.proj.")

    # HF: encoder.block.{i}.{j}.layer_norm_1
    # Ours: encoder.blocks.{i}.{j}.norm1
    key = key.replace(".layer_norm_1.", ".norm1.")
    key = key.replace(".layer_norm_2.", ".norm2.")

    # HF: encoder.block.{i}.{j}.mlp.dense1/dense2/dwconv
    # Ours: encoder.blocks.{i}.{j}.ffn.fc1/fc2/dwconv
    key = key.replace(".mlp.dense1.", ".ffn.fc1.")
    key = key.replace(".mlp.dense2.", ".ffn.fc2.")
    key = key.replace(".mlp.dwconv.dwconv.", ".ffn.dwconv.")

    # HF: encoder.layer_norm.{i}
    # Ours: encoder.norms.{i}
    key = key.replace("encoder.layer_norm.", "encoder.norms.")

    # --- Decoder mappings ---
    # HF: decode_head.linear_c.{i}.proj.weight → decoder.linear_projs.{i}.weight
    # Only strip ".proj" for decoder linear projections (not encoder patch embeds!)
    key = key.replace("decode_head.linear_c.", "decoder.linear_projs.")
    if key.startswith("decoder.linear_projs."):
        key = key.replace(".proj.weight", ".weight")
        key = key.replace(".proj.bias", ".bias")

    # HF: decode_head.linear_fuse → decoder.fuse.0 (Conv2d)
    key = key.replace("decode_head.linear_fuse.", "decoder.fuse.0.")
    # HF: decode_head.batch_norm → decoder.fuse.1 (BatchNorm2d)
    key = key.replace("decode_head.batch_norm.", "decoder.fuse.1.")

    # HF: decode_head.classifier → decoder.classifier
    key = key.replace("decode_head.classifier.", "decoder.classifier.")

    return key


if __name__ == "__main__":
    # Quick sanity check
    model = segformer_b0(num_classes=150)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Encoder params: {n_enc:,}")
    print(f"Decoder params: {n_dec:,}")
    print(f"Total params:   {n_total:,}")
