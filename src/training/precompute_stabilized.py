"""
Precompute the frozen data-domain diffusion stabilization + STFT
spectrogram ONCE for every sample in train/val/test, and cache to disk.

WHY: forward_stage1/forward_stage2 in dssnet.py previously re-ran the
12-step Heun sampler (paper's "solve++", Section III-B.c -- ~23 U-Net
passes per batch) on EVERY batch of EVERY epoch, even though the
data-domain module is frozen (paper: "trained independently and frozen
after convergence") and always produces the same output for the same
input. That meant the same expensive computation was redone 30+ times in
Stage 1 and up to 100 times in Stage 2. This script runs it exactly ONCE
per sample and caches the resulting spectrogram, so Stage 1/2 training
loads cheap precomputed tensors instead of running the sampler at all.

Usage:
    python precompute_stabilized.py --config ../../configs/sleepedf78.yaml \
        --data_diffusion_ckpt ../../data_diffusion_best.pt \
        --out_dir ../../cached_spectrograms
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_utils import load_config
from src.models.dssnet import DSSNet
from src.training.dataset import load_manifest, subject_wise_split, SleepEpochDataset


@torch.no_grad()
def precompute_split(model, loader, device, desc):
    specs, labels = [], []
    n_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        x_stable = model.stabilize_waveform(x)      # frozen 12-step Heun sampler -- runs ONCE here
        spec = model.to_spectrogram(x_stable)
        specs.append(spec.cpu())
        labels.append(y)
        if i % 20 == 0:
            print(f"  [{desc}] batch {i}/{n_batches}", flush=True)
    return torch.cat(specs, dim=0), torch.cat(labels, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_diffusion_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    subjects = load_manifest(ds_cfg["processed_dir"])
    train_subj, val_subj, test_subj = subject_wise_split(subjects, seed=42)
    print(f"Subjects -- train:{len(train_subj)} val:{len(val_subj)} test:{len(test_subj)}")

    train_ds = SleepEpochDataset(ds_cfg["processed_dir"], train_subj)
    val_ds = SleepEpochDataset(ds_cfg["processed_dir"], val_subj)
    test_ds = SleepEpochDataset(ds_cfg["processed_dir"], test_subj)

    ckpt = torch.load(args.data_diffusion_ckpt, map_location="cpu", weights_only=False)
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    for ds in (train_ds, val_ds, test_ds):
        ds.set_normalization(mean, std)
    print(f"Epochs -- train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    model = DSSNet(cfg).to(device)
    model.load_frozen_data_denoiser(ckpt["state_dict"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
        print(f"\n=== Precomputing {split_name} ({len(ds)} samples, {len(loader)} batches) ===")
        specs, labels = precompute_split(model, loader, device, split_name)
        out_path = os.path.join(args.out_dir, f"{split_name}_spectrograms.pt")
        torch.save({"spec": specs, "labels": labels}, out_path)
        print(f"Saved {split_name} -> {out_path}  (spec shape: {tuple(specs.shape)})")

    print("\nDone. All splits cached -- Stage 1/2 will now load these instead of "
          "running the diffusion sampler.")


if __name__ == "__main__":
    main()
