import torch
import torch.nn as nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # or dots = torch.einsum('bhit,bhjt->bhij', q, k) * self.scale
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        # or out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, num_heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x) -> torch.Tensor:
        x = self.norm(x)
        for att, ff in self.layers:
            x = att(x) + x
            x = ff(x) + x 
        return self.norm(x)

class PatchEmbedding(Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.dim = dim

        assert self.image_size[0] % self.patch_size[0] == 0 and self.image_size[1] % self.patch_size[1] == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        patch_dim = self.patch_size[0] * self.patch_size[1] * channels
        self.projection = nn.Linear(patch_dim, dim)
        
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size[0], p2=self.patch_size[1]),
            nn.LayerNorm(patch_dim),
            self.projection,
            nn.LayerNorm(dim)
        )

    def forward(self, x) -> torch.Tensor:
        return self.embedding(x)



class ViT(Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64,dropout=0.1, pool="cls"):
        super().__init__()
        assert pool in {'cls', 'mean'}, "Pool type must be either 'cls' or 'mean'"
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        assert self.image_size[0] % self.patch_size[0] == 0 and self.image_size[1] % self.patch_size[1] == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool

        num_cls_tokens = 1 if pool == 'cls' else 0
        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))
        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, dim, channels)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        if self.pool == 'cls':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        return self.mlp_head(x)