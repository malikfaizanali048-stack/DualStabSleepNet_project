"""
Vision Transformer backbone, Section III (Fig. 1b / Fig. 3).

Paper spec (Table II): ViT dim=256, depth=4, heads=4.
Fig. 3 shows the transformer stages producing multi-level features
F1, F2, F3, F4 that feed into per-level feature-domain diffusion (EDM)
modules -- with K=4 (Table II: num_levels=4). The natural reading, since
depth=4 exactly equals K=4, is that F_k = output of the k-th transformer
block (i.e. one feature level per encoder depth), which is what this
module implements.

ASSUMPTIONS (not specified in paper, documented):
  - patch_size=8 for the Conv2d patch embedding (paper doesn't give a
    patch/window size for STFT-map patchification).
  - Learned absolute positional embeddings (standard ViT choice).
  - Pre-norm transformer blocks (standard, matches Fig. 3's
    Norm -> MultiHeadAttention -> Norm -> MLP ordering shown in the figure).
  - F_k tokens include the CLS token; downstream classification uses CLS
    from F4 (the final level) unless fusion logic says otherwise (see
    fusion module, next milestone).
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T_frames) -> already padded to multiple of patch_size
        x = self.proj(x)                      # (B, embed_dim, F', T')
        x = x.flatten(2).transpose(1, 2)       # (B, N_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Matches Fig. 3: Norm -> MultiHeadAttention -> (+residual) -> Norm -> MLP -> (+residual)."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 256, depth: int = 4,
                 heads: int = 4, patch_size: int = 8, max_patches: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_channels, dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, F, T_frames) log-STFT stack, already padded to a multiple
           of patch_size (see stft_transform.pad_to_multiple).
        Returns: list of `depth` tensors [F1, F2, F3, F4], each (B, N+1, dim)
                 -- the hierarchical features consumed by feature-domain
                 diffusion modules.
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)                     # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, N+1, dim)
        n_tok = tokens.shape[1]
        if n_tok > self.pos_embed.shape[1]:
            raise ValueError(
                f"Number of patches+cls ({n_tok}) exceeds max_patches+1 "
                f"({self.pos_embed.shape[1]}); increase max_patches in config."
            )
        tokens = tokens + self.pos_embed[:, :n_tok, :]

        features = []
        h = tokens
        for block in self.blocks:
            h = block(h)
            features.append(h)
        return features  # [F1, F2, F3, F4]
