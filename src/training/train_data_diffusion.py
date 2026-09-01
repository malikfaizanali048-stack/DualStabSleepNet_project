"""
Stage 0: train the data-domain EDM denoiser (Section III-B) independently
and unsupervised. No labels used. Output checkpoint is loaded once via
DSSNet.load_frozen_data_denoiser() and never backprop'd through again.
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from ..models.unet1d import DataDomainUNet1D
from ..models.edm_utils import EDMPrecond, edm_loss
from .dataset import (
    SleepEpochDataset, subject_wise_split, compute_normalization_stats,
    load_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True, help="checkpoint output path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dm_cfg = cfg["diffusion"]["data_domain"]
    m_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = load_manifest(args.data_dir)
    train_s, val_s, _ = subject_wise_split(subjects, val_frac=0.1, test_frac=0.1, seed=42)

    train_ds = SleepEpochDataset(args.data_dir, train_s)
    val_ds = SleepEpochDataset(args.data_dir, val_s)

    mean, std = compute_normalization_stats(train_ds)
    train_ds.set_normalization(mean, std)
    val_ds.set_normalization(mean, std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    model = DataDomainUNet1D(in_channels=m_cfg["num_channels"]).to(device)
    precond = EDMPrecond(sigma_data=dm_cfg["sigma_data"])
    sigma_min, sigma_max = dm_cfg["sigma_min"], dm_cfg["sigma_max"]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, _ in train_loader:            # labels unused -- unsupervised
            x = x.to(device)
            opt.zero_grad()
            loss = edm_loss(model, x, sigma_min, sigma_max, precond)
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for x, _ in val_loader:
                x = x.to(device)
                val_running += edm_loss(model, x, sigma_min, sigma_max, precond).item()
            val_loss = val_running / len(val_loader)

        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "mean": mean, "std": std},
                       args.out)
            print(f"  saved best checkpoint (val_loss={val_loss:.5f}) -> {args.out}")


if __name__ == "__main__":
    main()