"""
Full DSSNet assembly (Fig. 1 end-to-end), tying together:
  - Data-domain diffusion (frozen, Section III-B)
  - STFT -> ViT student backbone (Section III-C, Eq 8)
  - EMA teacher (Eq 9)
  - K=4 feature-domain diffusion modules (Eq 10-13)
  - Fusion + classification head (Fig 3: "FN layer" x2 -> result)

Fusion head design (paper doesn't give exact fusion architecture beyond
"FN layer" x2 in Fig 3): we concatenate the CLS token from each of the 4
stabilized feature levels (after feature-domain projection), giving the
classifier visibility into all hierarchy levels, then two linear layers
down to num_classes=5. This is a documented assumption -- if per-class
numbers (esp. N1, the hardest stage) don't match Table IV, this fusion
head is one of the first places to revisit.

PERFORMANCE NOTE: forward_stage1 / forward_stage2 / forward_infer run the
frozen data-domain diffusion module's 12-step Heun sampler every call.
Since that module is frozen (never updated during Stage 1/2), its output
for a given input never changes -- re-running it every batch of every
epoch is wasteful. src/training/precompute_stabilized.py runs it ONCE per
sample ahead of time and caches the resulting spectrogram; the
`_from_spec` variants below (forward_stage1_from_spec,
forward_stage2_from_spec, forward_infer_from_spec) consume those cached
spectrograms directly, skipping stabilize_waveform + to_spectrogram
entirely. Use these in train_dssnet_two_stage.py once cached data exists.
The original x_raw-based methods are kept unchanged for cases where cached
spectrograms aren't available.
"""
import torch
import torch.nn as nn

from .unet1d import DataDomainUNet1D
from .edm_utils import EDMPrecond, heun_deterministic_sample
from .vit_backbone import ViTBackbone
from .feature_diffusion import (
    FeatureCorrectionNet, make_ema_teacher, update_ema,
    feature_diffusion_loss, project_features, project_features_train,
)
from ..preprocessing.stft_transform import compute_log_stft, pad_to_multiple


NUM_CLASSES = 5  # W, N1, N2, N3, REM


class FusionClassifier(nn.Module):
    def __init__(self, dim: int = 256, num_levels: int = 4, num_classes: int = NUM_CLASSES,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fn1 = nn.Sequential(nn.Linear(dim * num_levels, hidden), nn.GELU(), nn.Dropout(dropout))
        self.fn2 = nn.Linear(hidden, num_classes)

    def forward(self, stabilized_features: list) -> torch.Tensor:
        cls_tokens = [f[:, 0, :] for f in stabilized_features]  # CLS token per level, (B, dim) each
        fused = torch.cat(cls_tokens, dim=-1)                    # (B, dim*num_levels)
        h = self.fn1(fused)
        return self.fn2(h)                                       # (B, num_classes) logits


class DSSNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        dm = cfg["diffusion"]["data_domain"]
        fm = cfg["diffusion"]["feature_domain"]
        d = cfg["data"]

        self.sample_rate = d.get("sampling_rate_hz", 100)
        self.stft_cfg = d["stft"]
        self.patch_size = m.get("patch_size", 8)

        # --- Data-domain diffusion (trained + frozen separately, see
        # src/training/train_data_diffusion.py -- loaded via load_state_dict
        # and never updated here) ---
        self.data_denoiser = DataDomainUNet1D(in_channels=m["num_channels"])
        self.data_precond = EDMPrecond(sigma_data=dm["sigma_data"])
        self.data_sigma_min = dm["sigma_min"]
        self.data_sigma_max = dm["sigma_max"]
        self.data_n_steps = dm["n_steps"]
        for p in self.data_denoiser.parameters():
            p.requires_grad_(False)

        # --- ViT student + EMA teacher ---
        self.vit_student = ViTBackbone(
            in_channels=m["num_channels"], dim=m["vit_dim"], depth=m["vit_depth"],
            heads=m["vit_heads"], patch_size=self.patch_size,
        )
        self.vit_teacher = make_ema_teacher(self.vit_student)
        self.ema_mu = fm["ema_momentum"]

        # --- Feature-domain diffusion modules, one per level (K) ---
        self.num_levels = fm["num_levels"]
        self.feat_modules = nn.ModuleList([
            FeatureCorrectionNet(dim=m["vit_dim"]) for _ in range(self.num_levels)
        ])
        self.feat_precond = EDMPrecond(sigma_data=0.5)  # see module docstring assumption
        self.feat_sigma_min = fm.get("sigma_feat_min", 1e-3)  # not given in paper; documented default
        self.feat_sigma_max = fm["sigma_feat_max"]
        self.lambda_feat = fm["lambda_feat"]

        self.classifier = FusionClassifier(dim=m["vit_dim"], num_levels=self.num_levels)

    def load_frozen_data_denoiser(self, state_dict):
        self.data_denoiser.load_state_dict(state_dict)
        self.data_denoiser.eval()

    @torch.no_grad()
    def stabilize_waveform(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Frozen data-domain diffusion, deterministic solve++ (Heun) sampling."""
        sigmas = torch.linspace(self.data_sigma_max, 1e-4, self.data_n_steps, device=x_raw.device)
        return heun_deterministic_sample(self.data_denoiser, x_raw, sigmas, self.data_precond)

    def to_spectrogram(self, x_stable: torch.Tensor) -> torch.Tensor:
        spec = compute_log_stft(
            x_stable, sample_rate=self.sample_rate,
            win_seconds=self.stft_cfg["win_seconds"], hop_seconds=self.stft_cfg["hop_seconds"],
            window=self.stft_cfg["window"],
        )
        return pad_to_multiple(spec, multiple=self.patch_size)

    def forward_stage1(self, x_raw: torch.Tensor):
        """
        Stage 1 (backbone frozen): train ONLY feature diffusion modules on
        Eq 10-12 against the (static, since backbone frozen) teacher.
        Returns the scalar loss sum_k L_feat^(k).

        NOTE: runs the frozen 12-step diffusion sampler every call. Prefer
        forward_stage1_from_spec() with precomputed spectrograms for
        training loops (see precompute_stabilized.py).
        """
        with torch.no_grad():
            x_stable = self.stabilize_waveform(x_raw)
            spec = self.to_spectrogram(x_stable)
            teacher_features = self.vit_teacher(spec)  # list of K tensors, no grad needed

        total_loss = 0.0
        for k in range(self.num_levels):
            total_loss = total_loss + feature_diffusion_loss(
                self.feat_modules[k], self.feat_precond, teacher_features[k],
                self.feat_sigma_min, self.feat_sigma_max,
            )
        return total_loss

    def forward_stage1_from_spec(self, spec: torch.Tensor):
        """
        Same as forward_stage1, but takes an already-stabilized,
        already-STFT'd spectrogram (from precompute_stabilized.py) instead
        of raw waveform -- skips the frozen diffusion sampler entirely.
        """
        with torch.no_grad():
            teacher_features = self.vit_teacher(spec)

        total_loss = 0.0
        for k in range(self.num_levels):
            total_loss = total_loss + feature_diffusion_loss(
                self.feat_modules[k], self.feat_precond, teacher_features[k],
                self.feat_sigma_min, self.feat_sigma_max,
            )
        return total_loss

    def forward_stage2(self, x_raw: torch.Tensor):
        """
        Stage 2 (joint fine-tune): returns (logits, feat_reg_loss).
        Caller combines: L_total = CE(logits, y) + lambda * feat_reg_loss  (Eq 13)

        NOTE: runs the frozen 12-step diffusion sampler every call. Prefer
        forward_stage2_from_spec() with precomputed spectrograms for
        training loops (see precompute_stabilized.py).
        """
        with torch.no_grad():
            x_stable = self.stabilize_waveform(x_raw)
        spec = self.to_spectrogram(x_stable)

        student_features = self.vit_student(spec)          # grad flows (Eq 8)
        with torch.no_grad():
            teacher_features = self.vit_teacher(spec)       # Eq 9, no grad

        stabilized_features = []
        feat_reg_loss = 0.0
        for k in range(self.num_levels):
            stabilized = project_features_train(
                self.feat_modules[k], self.feat_precond, student_features[k],
            )
            stabilized_features.append(stabilized)
            feat_reg_loss = feat_reg_loss + feature_diffusion_loss(
                self.feat_modules[k], self.feat_precond, teacher_features[k],
                self.feat_sigma_min, self.feat_sigma_max,
            )

        logits = self.classifier(stabilized_features)
        return logits, feat_reg_loss

    def forward_stage2_from_spec(self, spec: torch.Tensor):
        """
        Same as forward_stage2, but takes an already-stabilized,
        already-STFT'd spectrogram instead of raw waveform -- skips the
        frozen diffusion sampler entirely.
        """
        student_features = self.vit_student(spec)
        with torch.no_grad():
            teacher_features = self.vit_teacher(spec)

        stabilized_features = []
        feat_reg_loss = 0.0
        for k in range(self.num_levels):
            stabilized = project_features_train(
                self.feat_modules[k], self.feat_precond, student_features[k],
            )
            stabilized_features.append(stabilized)
            feat_reg_loss = feat_reg_loss + feature_diffusion_loss(
                self.feat_modules[k], self.feat_precond, teacher_features[k],
                self.feat_sigma_min, self.feat_sigma_max,
            )

        logits = self.classifier(stabilized_features)
        return logits, feat_reg_loss

    @torch.no_grad()
    def forward_infer(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Inference: student backbone + deterministic feature projection, no teacher needed."""
        x_stable = self.stabilize_waveform(x_raw)
        spec = self.to_spectrogram(x_stable)
        student_features = self.vit_student(spec)
        stabilized_features = [
            project_features(self.feat_modules[k], self.feat_precond, student_features[k])
            for k in range(self.num_levels)
        ]
        return self.classifier(stabilized_features)

    @torch.no_grad()
    def forward_infer_from_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """Same as forward_infer, but takes a precomputed spectrogram directly."""
        student_features = self.vit_student(spec)
        stabilized_features = [
            project_features(self.feat_modules[k], self.feat_precond, student_features[k])
            for k in range(self.num_levels)
        ]
        return self.classifier(stabilized_features)

    def ema_step(self):
        update_ema(self.vit_teacher, self.vit_student, self.ema_mu)

    def set_stage1_trainable(self):
        """Freeze everything except the K feature diffusion modules."""
        for p in self.vit_student.parameters():
            p.requires_grad_(False)
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        for m in self.feat_modules:
            for p in m.parameters():
                p.requires_grad_(True)

    def set_stage2_trainable(self):
        """Unfreeze backbone + fusion + classifier + feature diffusion modules."""
        for p in self.vit_student.parameters():
            p.requires_grad_(True)
        for p in self.classifier.parameters():
            p.requires_grad_(True)
        for m in self.feat_modules:
            for p in m.parameters():
                p.requires_grad_(True)
