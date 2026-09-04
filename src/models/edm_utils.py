"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models,
Karras et al. 2022 [31]) preconditioning, used identically by BOTH the
data-domain and feature-domain diffusion modules per the paper's Section
III-A ("unified continuous-scale diffusion framework").

Paper equations implemented here:
  Eq. 1: x = x_clean + sigma * eps                (forward perturbation)
  Eq. 2: D_theta(x, sigma) ~= x_clean              (data-prediction target)
  Eq. 3: L(theta) = E[ w(sigma) * ||D_theta(x,sigma) - x_clean||^2 ]
  Eq. 5: D_theta(x,sigma) = c_skip(x)*x + c_out(sigma)*F_theta(c_in(sigma)*x, c_noise(sigma))

c_skip/c_out/c_in/c_noise/w below are the standard EDM closed forms (Karras
et al., Table 1). The paper cites [31] for this and does not redefine the
coefficients, so we use the canonical EDM formulas rather than inventing
alternatives -- this is the "same architecture as the paper" choice.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class EDMPrecond:
    sigma_data: float = 0.5   # matches paper's Table II: sigma_data = 0.5

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        # w(sigma) that makes c_out(sigma) * sqrt(w(sigma)) == 1, so the
        # network's effective training target has unit variance regardless
        # of sigma (Karras et al. Eq. 8). This is what Eq. 3's w(sigma) is.
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def precondition(self, F_theta, x: torch.Tensor, sigma: torch.Tensor, **f_kwargs):
        """
        Implements Eq. 5. `F_theta` is the raw backbone (the 1D U-Net for
        data-domain, or the small per-level correction net for
        feature-domain). `sigma` is expected shape (B,) or (B,1,...,1)
        broadcastable to x.
        """
        sigma = sigma.reshape(-1, *([1] * (x.dim() - 1)))
        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma).reshape(-1)  # network noise-conditioning input, shape (B,)

        F_out = F_theta(c_in * x, c_noise, **f_kwargs)
        return c_skip * x + c_out * F_out


def sample_log_uniform_sigma(batch_size, sigma_min, sigma_max, device):
    """
    Continuous-scale sigma sampling per Eq. 1: sigma in [sigma_min, sigma_max].
    Sampled log-uniformly (standard EDM training-time distribution choice;
    the paper specifies the RANGE [0.05, 0.2] in Table II but not the exact
    sampling density within it -- log-uniform is the EDM-paper convention
    we inherit since this framework is explicitly built on [31]).
    """
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    u = torch.rand(batch_size, device=device)
    return torch.exp(log_min + u * (log_max - log_min))


def edm_loss(F_theta, x_clean: torch.Tensor, sigma_min: float, sigma_max: float,
             precond: EDMPrecond, **f_kwargs) -> torch.Tensor:
    """
    Full Eq. 1-3 training step: sample sigma and noise, perturb x_clean,
    denoise, compute scale-weighted MSE against x_clean.
    """
    B = x_clean.shape[0]
    sigma = sample_log_uniform_sigma(B, sigma_min, sigma_max, x_clean.device)
    eps = torch.randn_like(x_clean)
    sigma_bcast = sigma.reshape(-1, *([1] * (x_clean.dim() - 1)))
    x_noisy = x_clean + sigma_bcast * eps

    x_pred = precond.precondition(F_theta, x_noisy, sigma, **f_kwargs)

    w = precond.loss_weight(sigma).reshape(-1, *([1] * (x_clean.dim() - 1)))
    loss = (w * (x_pred - x_clean) ** 2).mean()
    return loss


@torch.no_grad()
def heun_deterministic_sample(F_theta, x_init: torch.Tensor, sigmas: torch.Tensor,
                               precond: EDMPrecond, **f_kwargs) -> torch.Tensor:
    """
    Deterministic reverse-process sampler along a monotonically decreasing
    noise schedule `sigmas` (e.g. torch.linspace(sigma_max, sigma_min, N)).

    The paper names this 'solve++' [31] without giving explicit pseudocode.
    [31] (Karras et al.) is the EDM paper, whose reference deterministic
    sampler is 2nd-order Heun -- so we implement that as the best-supported
    reading of 'solve++'. FLAG: if you find the exact solve++ definition
    elsewhere, swap this out; note this assumption in your writeup.
    """
    x = x_init
    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        sigma_cur_b = sigma_cur.expand(x.shape[0])
        d_cur = (x - precond.precondition(F_theta, x, sigma_cur_b, **f_kwargs)) / sigma_cur
        x_euler = x + (sigma_next - sigma_cur) * d_cur

        if sigma_next > 0:
            sigma_next_b = sigma_next.expand(x.shape[0])
            d_next = (x_euler - precond.precondition(F_theta, x_euler, sigma_next_b, **f_kwargs)) / sigma_next
            x = x + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)
        else:
            x = x_euler
    return x
