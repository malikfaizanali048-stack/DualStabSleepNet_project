"""
STFT(H x W) transform, Fig. 1(a): converts the diffusion-stabilized raw
waveform (B, C, T) into per-channel time-frequency maps (B, C, F, T_frames)
fed into the ViT patch embedding.

ASSUMPTION (not specified in paper): win_seconds/hop_seconds/window come
from configs/base.yaml -> data.stft (win=2s, hop=1s, hann). Log-magnitude
spectrogram is the standard choice for CNN/ViT-friendly time-frequency
input and matches common sleep-staging literature (e.g. AttnSleep,
SleepTransformer use similar STFT front-ends).
"""
import torch


def compute_log_stft(x: torch.Tensor, sample_rate: int, win_seconds: float,
                      hop_seconds: float, window: str = "hann") -> torch.Tensor:
    """
    x: (B, C, T) raw waveform, already diffusion-stabilized.
    Returns: (B, C, F, T_frames) log-magnitude spectrogram, one per channel.
    """
    B, C, T = x.shape
    win_length = int(win_seconds * sample_rate)
    hop_length = int(hop_seconds * sample_rate)
    n_fft = win_length

    if window == "hann":
        win = torch.hann_window(win_length, device=x.device)
    else:
        raise ValueError(f"Unsupported window: {window}")

    x_flat = x.reshape(B * C, T)
    spec = torch.stft(
        x_flat, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=win, return_complex=True, center=True, pad_mode="reflect",
    )  # (B*C, F, T_frames)
    mag = spec.abs()
    log_mag = torch.log1p(mag)  # log(1+|X|), stable at 0 and monotonic

    F_bins, T_frames = log_mag.shape[-2], log_mag.shape[-1]
    return log_mag.reshape(B, C, F_bins, T_frames)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Reflect-pads the last two spatial dims (F, T_frames) up to a multiple
    of `multiple`, so patch embedding with stride=multiple divides evenly."""
    F_bins, T_frames = x.shape[-2], x.shape[-1]
    pad_f = (-F_bins) % multiple
    pad_t = (-T_frames) % multiple
    # pad order: (left, right, top, bottom) on last two dims
    return torch.nn.functional.pad(x, (0, pad_t, 0, pad_f), mode="reflect")