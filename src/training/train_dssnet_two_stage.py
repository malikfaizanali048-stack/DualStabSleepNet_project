"""
Stage 1 + Stage 2 training loop, using DSSNet.forward_stage1/forward_stage2
and set_stage1_trainable/set_stage2_trainable (already verified end-to-end).
Requires: frozen data-domain checkpoint from train_data_diffusion.py already
saved (Stage 0 must run first).
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.dssnet import DSSNet
from .dataset import (
    SleepEpochDataset, subject_wise_split, compute_normalization_stats,
    make_balanced_sampler, load_manifest,
)


def run_epoch_stage1(model, loader, opt, device):
    model.train()
    running = 0.0
    for x, _ in loader:                       # unsupervised at this stage
        x = x.to(device)
        opt.zero_grad()
        loss = model.forward_stage1(x)
        loss.backward()
        opt.step()
        model.ema_step()                       # Eq 9, update teacher after student-side... 
        # NOTE: backbone is frozen in stage 1, so EMA teacher==student here;
        # ema_step() is a no-op in effect but kept for interface consistency.
        running += loss.item()
    return running / len(loader)


def run_epoch_stage2(model, loader, opt, device, lambda_feat, criterion):
    model.train()
    running_cls, running_feat, correct, total = 0.0, 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits, feat_loss = model.forward_stage2(x)
        cls_loss = criterion(logits, y)
        loss = cls_loss + lambda_feat * feat_loss          # Eq 13
        loss.backward()
        opt.step()
        model.ema_step()                                    # Eq 9, teacher tracks student now

        running_cls += cls_loss.item()
        running_feat += feat_loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return running_cls / len(loader), running_feat / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, running = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model.forward_infer(x)
        loss = criterion(logits, y)
        running += loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return running / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_denoiser_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = load_manifest(args.data_dir)
    train_s, val_s, test_s = subject_wise_split(subjects, val_frac=0.1, test_frac=0.1, seed=42)

    train_ds = SleepEpochDataset(args.data_dir, train_s)
    val_ds = SleepEpochDataset(args.data_dir, val_s)

    mean, std = compute_normalization_stats(train_ds)
    train_ds.set_normalization(mean, std)
    val_ds.set_normalization(mean, std)

    train_sampler = make_balanced_sampler(train_ds)        # N1 balancing
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=train_sampler, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    model = DSSNet(cfg).to(device)

    ckpt = torch.load(args.data_denoiser_ckpt, map_location=device)
    model.load_frozen_data_denoiser(ckpt["model"])
    print(f"loaded frozen data-domain denoiser from {args.data_denoiser_ckpt}")

    criterion = nn.CrossEntropyLoss()
    lambda_feat = cfg["diffusion"]["feature_domain"]["lambda_feat"]

    # ---- Stage 1: freeze everything except feature-diffusion modules ----
    model.set_stage1_trainable()
    stage1_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(stage1_params, lr=1e-4, weight_decay=1e-4)

    print(f"Stage 1: {sum(p.numel() for p in stage1_params):,} trainable params")
    for epoch in range(args.stage1_epochs):
        loss = run_epoch_stage1(model, train_loader, opt1, device)
        print(f"[stage1 epoch {epoch+1}/{args.stage1_epochs}] feat_loss={loss:.5f}")

    # ---- Stage 2: unfreeze backbone + classifier + feature-diffusion ----
    model.set_stage2_trainable()
    stage2_params = [p for p in model.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(stage2_params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.stage2_epochs)

    print(f"Stage 2: {sum(p.numel() for p in stage2_params):,} trainable params")
    best_val_acc = 0.0
    patience, bad_epochs = 10, 0
    for epoch in range(args.stage2_epochs):
        cls_loss, feat_loss, train_acc = run_epoch_stage2(
            model, train_loader, opt2, device, lambda_feat, criterion
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"[stage2 epoch {epoch+1}/{args.stage2_epochs}] "
              f"cls_loss={cls_loss:.4f} feat_loss={feat_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "val_acc": val_acc}, args.out)
            print(f"  saved best checkpoint (val_acc={val_acc:.4f}) -> {args.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch+1} (patience={patience})")
                break


if __name__ == "__main__":
    main()