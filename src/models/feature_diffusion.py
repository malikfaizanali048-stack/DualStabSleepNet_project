"""
Feature-domain stabilization, Section III-C (Eq. 8-13), Fig. 3.

Implements:
  Eq 8-9: student features F_k, EMA teacher features F_t_k
  Eq 10:  F_tilde_k = F_t_k + sigma*eps          (perturb around teacher)
  Eq 11:  F_hat_k = D_psi^k(F_tilde_k, sigma)     (per-level correction net)
  Eq 12:  L_feat^(k) = E[w(sigma)||F_hat_k - F_t_k||^2]
  Eq 13:  L_total = L_cls + lambda * sum_k L_feat^(k)   (assembled in dssnet.py)

Training-strategy reading (paper III-C.c is compressed; documented here):
  Stage 1: freeze the ViT student -> teacher (EMA of a static student) is
           also static -> train ONLY the K correction nets D_psi^k on Eq 10-12.
  Stage 2: unfreeze student+fusion+classifier. Per batch:
           - classification path uses `project_features` (student features
             pushed through D_psi^k at sigma->0, i.e. the INFERENCE-TIME
             deterministic projection described in III-C.c) -> fusion -> L_cls.
           - regularization path keeps computing Eq 10-12 (`feature_diffusion_loss`)
             against the (now slowly-moving) EMA teacher, to keep D_psi^k
             calibrated as the manifold drifts during joint fine-tuning.
  This is our best-supported reading of the paper's two-stage description,
  not an explicit paper algorithm box -- flagged for you to sanity check.

ASSUMPTIONS (undocumented in paper):
  - sigma_data for feature domain reused at 0.5 (same EDM constant as data
    domain, since paper frames this as "the same modeling formulation...
    instantiated at different levels", III-A).
  - sigma_min for feature domain not given (only sigma_feat_max=0.2 in
    Table II); using 1e-3 as a small default floor.
  - Correction net D_psi^k is a lightweight per-token FiLM-conditioned
    residual MLP (matches paper's "lightweight projection operators"
    description, III-C.c) rather than a full U-Net -- appropriate since
    it operates on (B, N_tokens, dim) feature sequences, not images/waveforms.
"""
import copy
import math

import torch
import torch.nn as nn

from .edm_utils import EDMPrecond


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class FeatureCorrectionNet(nn.Module):
    """D_psi^k: lightweight FiLM-conditioned residual MLP, operates per-token
    on (B, N, dim) feature sequences, broadcasting the noise condition
    across all tokens."""
    def __init__(self, dim: int = 256, hidden_mult: int = 2, cond_dim: int = 64):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.cond_dim = cond_dim
        hidden = dim * hidden_mult
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.film = nn.Linear(cond_dim, hidden * 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, f: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        # f: (B, N, dim), c_noise: (B,)
        cond = self.cond_mlp(sinusoidal_embedding(c_noise, self.cond_dim))  # (B, cond_dim)
        h = self.norm1(f)
        h = self.fc1(h)
        scale, shift = self.film(cond).chunk(2, dim=-1)   # (B, hidden) each
        h = h * (1 + scale[:, None, :]) + shift[:, None, :]
        h = nn.functional.silu(self.norm2(h))
        h = self.fc2(h)
        return f + h  # residual correction


def make_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, mu: float = 0.999):
    """Eq 9: theta_bar <- mu*theta_bar + (1-mu)*theta"""
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.mul_(mu).add_(p_s.detach(), alpha=(1 - mu))
    for b_t, b_s in zip(teacher.buffers(), student.buffers()):
        b_t.copy_(b_s)


def sample_log_uniform_sigma(batch_size, sigma_min, sigma_max, device):
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    u = torch.rand(batch_size, device=device)
    return torch.exp(log_min + u * (log_max - log_min))


def feature_diffusion_loss(module: FeatureCorrectionNet, precond: EDMPrecond,
                            teacher_feat: torch.Tensor, sigma_min: float,
                            sigma_max: float) -> torch.Tensor:
    """Eq 10-12: perturb the (detached) teacher feature, denoise, weighted MSE."""
    teacher_feat = teacher_feat.detach()
    B = teacher_feat.shape[0]
    sigma = sample_log_uniform_sigma(B, sigma_min, sigma_max, teacher_feat.device)
    eps = torch.randn_like(teacher_feat)
    sigma_b = sigma.reshape(-1, 1, 1)
    f_tilde = teacher_feat + sigma_b * eps                     # Eq 10

    f_hat = precond.precondition(module, f_tilde, sigma)       # Eq 11

    w = precond.loss_weight(sigma).reshape(-1, 1, 1)
    loss = (w * (f_hat - teacher_feat) ** 2).mean()             # Eq 12
    return loss


@torch.no_grad()
def project_features(module: FeatureCorrectionNet, precond: EDMPrecond,
                      student_feat: torch.Tensor, sigma_eps: float = 1e-3) -> torch.Tensor:
    """
    Inference-time deterministic projection (III-C.c): apply D_psi^k to the
    STUDENT feature at an infinitesimal noise scale sigma->0, acting as a
    lightweight stabilizing correction (no multi-step sampling needed --
    the paper explicitly says this adds no inference complexity).
    """
    B = student_feat.shape[0]
    sigma = torch.full((B,), sigma_eps, device=student_feat.device)
    return precond.precondition(module, student_feat, sigma)


def project_features_train(module: FeatureCorrectionNet, precond: EDMPrecond,
                            student_feat: torch.Tensor, sigma_eps: float = 1e-3) -> torch.Tensor:
    """Same as project_features but WITH gradient (used in Stage 2 joint training,
    since L_cls must backprop through the projected features into the backbone)."""
    B = student_feat.shape[0]
    sigma = torch.full((B,), sigma_eps, device=student_feat.device)
    return precond.precondition(module, student_feat, sigma)