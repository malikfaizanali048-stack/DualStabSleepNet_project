"""
Stage 0: train the data-domain diffusion module INDEPENDENTLY of everything
else (paper III-B: "trained independently and frozen after convergence...
No classification loss is back-propagated through this module").

This is unsupervised -- pure EDM denoising objective (Eq 1-3) on raw
waveform epochs. Labels aren't used at all. Train split only (still
respecting the person-level subject-wise split so this module never sees
test-subject waveforms either -- test-set leakage isn't just about labels).

Usage:
    python train_data_diffusion.py --config ../../configs/sleepedf78.yaml \
        --epochs 50 --batch_size 128
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_utils import load_config
from src.models.unet1d import DataDomainUNet1D
from src.models.edm_utils import EDMPrecond, edm_loss
from src.training.dataset import (
    load_manifest, subject_wise_split, SleepEpochDataset, compute_normalization_stats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--out", default=None, help="checkpoint output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    dm_cfg = cfg["diffusion"]["data_domain"]
    batch_size = args.batch_size or cfg["optim"]["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    subjects = load_manifest(ds_cfg["processed_dir"])
    train_subj, val_subj, _ = subject_wise_split(subjects, seed=42)
    print(f"Train subjects: {len(train_subj)}  Val subjects: {len(val_subj)}")

    train_ds = SleepEpochDataset(ds_cfg["processed_dir"], train_subj)
    val_ds = SleepEpochDataset(ds_cfg["processed_dir"], val_subj)
    mean, std = compute_normalization_stats(train_ds)
    train_ds.set_normalization(mean, std)
    val_ds.set_normalization(mean, std)
    print(f"Train epochs: {len(train_ds)}  Val epochs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, drop_last=True, pin_memory=True,
                               persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    net = DataDomainUNet1D(in_channels=cfg["model"]["num_channels"]).to(device)
    precond = EDMPrecond(sigma_data=dm_cfg["sigma_data"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["optim"]["lr"],
                             weight_decay=cfg["optim"].get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["optim"].get("early_stopping_patience", 15)
    out_path = args.out or os.path.join(ds_cfg["processed_dir"], "..", "data_diffusion_best.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(args.epochs):
        net.train()
        train_losses = []
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = edm_loss(net, x, dm_cfg["sigma_min"], dm_cfg["sigma_max"], precond)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()

        net.eval()
        val_losses = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    loss = edm_loss(net, x, dm_cfg["sigma_min"], dm_cfg["sigma_max"], precond)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "state_dict": net.state_dict(),
                "norm_mean": mean, "norm_std": std,
                "epoch": epoch, "val_loss": val_loss,
            }, out_path)
            print(f"  -> saved new best checkpoint to {out_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> early stopping: no val_loss improvement for {patience} epochs")
                break

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. Checkpoint: {out_path}")


if __name__ == "__main__":
    main()