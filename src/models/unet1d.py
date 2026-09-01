"""
1D U-Net backbone (F_theta in EDM's Eq. 5) for the DATA-DOMAIN diffusion
stabilization module described in paper Section III-B.

Paper's description (Section III-B.b) that this must satisfy:
  - "EDM-preconditioned one-dimensional U-Net architecture with a
     multi-scale encoder-decoder topology and skip connections"
  - "Residual modeling is incorporated within each scale to stabilize
     optimization"
  - noise-scale-conditioned (c_noise input from EDM preconditioning)
  - operates on raw multi-channel PSG waveform segments (EEG, EOG, EMG
     stacked as channels) -- NOT yet transformed to time-frequency; that
     happens after this module, per Fig. 1(a).

The paper does not give explicit channel counts / depth for this U-Net, so
those are documented assumptions below (first tuning knobs if data-domain
ablation numbers -- Table V/VI -- don't match).
"""
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of the (scalar, per-sample) c_noise value."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    args = x[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class FiLMResidualBlock1D(nn.Module):
    """
    Residual conv block with FiLM-style noise-scale conditioning.
    "Residual modeling incorporated within each scale" (paper III-B.b).
    """
    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)  # scale + shift (FiLM)

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None]
        shift = shift[:, :, None]

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = nn.functional.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nn.functional.silu(h)
        h = self.conv2(h)
        return x + h


class Down1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class DataDomainUNet1D(nn.Module):
    """
    F_theta for the data-domain EDM module. Input/output shape: (B, C, T)
    where C = num_pgs_channels (EEG, EOG, EMG -> 3 per paper) and T = the
    number of raw waveform samples in one input segment.

    ASSUMPTIONS (undocumented in paper, log & treat as tuning knobs):
      - base_channels=32, channel_mults=(1,2,4,4) -> 4 scales, matching the
        "multi-scale encoder-decoder" description qualitatively.
      - 2 residual blocks per scale.
      - kernel_size=9 (wide temporal receptive field appropriate for
        30s-epoch physiological waveforms).
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 channel_mults=(1, 2, 4, 4), num_res_blocks: int = 2,
                 cond_dim: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = base_channels
        enc_channels = [ch]
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = nn.ModuleList(
                [FiLMResidualBlock1D(out_ch, cond_dim) for _ in range(num_res_blocks)]
            )
            # project channel count at the first block of the stage if needed
            self.down_blocks.append(blocks)
            if ch != out_ch:
                self.down_blocks.append(nn.Conv1d(ch, out_ch, kernel_size=1))
            else:
                self.down_blocks.append(nn.Identity())
            self.downsamplers.append(Down1D(out_ch, out_ch))
            ch = out_ch
            enc_channels.append(ch)

        self.mid_block1 = FiLMResidualBlock1D(ch, cond_dim)
        self.mid_block2 = FiLMResidualBlock1D(ch, cond_dim)

        # Decoder (mirrors encoder, with skip connections)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        rev_mults = list(reversed(channel_mults))
        for idx, mult in enumerate(rev_mults):
            out_ch = base_channels * mult
            self.upsamplers.append(Up1D(ch, out_ch))
            skip_ch = enc_channels[-(idx + 1)]
            self.skip_proj.append(nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=1))
            blocks = nn.ModuleList([FiLMResidualBlock1D(out_ch, cond_dim) for _ in range(num_res_blocks)])
            self.up_blocks.append(blocks)
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, in_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) already scaled by c_in (EDM preconditioning handles that
           outside this module -- see EDMPrecond.precondition).
        c_noise: (B,) scalar noise-conditioning value from EDM preconditioning.
        """
        cond = self.cond_mlp(sinusoidal_embedding(c_noise, self.cond_dim))

        h = self.in_conv(x)
        skips = [h]
        i = 0
        for stage in range(len(self.downsamplers)):
            blocks = self.down_blocks[i]
            proj = self.down_blocks[i + 1]
            i += 2
            h = proj(h)
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            h = self.downsamplers[stage](h)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        for stage in range(len(self.upsamplers)):
            h = self.upsamplers[stage](h)
            skip = skips[-(stage + 1)]
            if h.shape[-1] != skip.shape[-1]:
                h = nn.functional.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = self.skip_proj[stage](h)
            for block in self.up_blocks[stage]:
                h = block(h, cond)

        h = self.out_norm(h)
        h = nn.functional.silu(h)
        return self.out_conv(h)