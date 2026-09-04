"""
Stage 2 ONLY (Section III-C.c): loads the Stage 1 checkpoint (feature
diffusion modules pretrained against the frozen teacher), then unfreezes
everything and jointly optimizes L_total = L_cls + lambda * L_feat (Eq 13),
with the EMA teacher updated every step (Eq 9).

Requires:
  - precomputed spectrograms from precompute_stabilized.py
  - a Stage 1 checkpoint from train_stage1.py

Usage:
    python train_stage2.py --config ../../configs/sleepedf78.yaml \
        --cache_dir ../../cached_spectrograms \
        --stage1_ckpt /kaggle/working/stage1_checkpoint.pt \
        --out /kaggle/working/dssnet_best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.dssnet import DSSNet
from src.training.dataset import CachedSpectrogramDataset, make_balanced_sampler
from src.eval.metrics import compute_metrics, format_metrics
from src.config_utils import load_config


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for spec, y in loader:
        spec = spec.to(device)
        logits = model.forward_infer_from_spec(spec)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--stage1_ckpt", required=True,
                         help="checkpoint produced by train_stage1.py")
    parser.add_argument("--out", default="/kaggle/working/dssnet_best.pt")
    parser.add_argument("--resume", action="store_true",
                         help="resume Stage 2 from --out if it already exists "
                              "(instead of starting fresh from --stage1_ckpt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optim_cfg = cfg["optim"]
    fm_cfg = cfg["diffusion"]["feature_domain"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = CachedSpectrogramDataset(os.path.join(args.cache_dir, "train_spectrograms.pt"))
    val_ds = CachedSpectrogramDataset(os.path.join(args.cache_dir, "val_spectrograms.pt"))
    test_ds = CachedSpectrogramDataset(os.path.join(args.cache_dir, "test_spectrograms.pt"))
    print(f"Epochs -- train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    batch_size = optim_cfg["batch_size"]
    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = DSSNet(cfg).to(device)

    start_epoch = 0
    best_val_macro_f1 = -1.0

    if args.resume and os.path.exists(args.out):
        # Continuing an interrupted Stage 2 run.
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_macro_f1 = ckpt["val_metrics"]["macro_f1"]
        print(f"Resumed Stage 2 from epoch {start_epoch} "
              f"(best val macro-F1={best_val_macro_f1*100:.2f}%)")
    else:
        # Fresh Stage 2 start: load Stage 1's trained feature-diffusion weights.
        stage1_ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stage1_ckpt["model_state_dict"])
        print(f"Loaded Stage 1 checkpoint from {args.stage1_ckpt} "
              f"(stage1 loss={stage1_ckpt.get('loss', 'n/a')})")

    print("\n=== STAGE 2: joint fine-tuning ===")
    model.set_stage2_trainable()
    opt2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=optim_cfg["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=optim_cfg["stage2_epochs"])
    scaler2 = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    patience_counter = 0
    total_epochs = optim_cfg["stage2_epochs"]
    for epoch in range(start_epoch, total_epochs):
        model.train()
        losses = []
        for spec, y in train_loader:
            spec, y = spec.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt2.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits, feat_reg_loss = model.forward_stage2_from_spec(spec)
                cls_loss = torch.nn.functional.cross_entropy(logits, y)
                total_loss = cls_loss + fm_cfg["lambda_feat"] * feat_reg_loss

            scaler2.scale(total_loss).backward()
            scaler2.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler2.step(opt2)
            scaler2.update()
            model.ema_step()  # Eq 9, every step
            losses.append(total_loss.item())
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        print(f"[Stage2] Epoch {epoch+1}/{total_epochs}  "
              f"train_loss={sum(losses)/len(losses):.4f}  "
              f"val: {format_metrics(val_metrics)}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, args.out)
            print(f"  -> new best (val macro-F1={best_val_macro_f1*100:.2f}%), saved to {args.out}")
        else:
            patience_counter += 1
            if patience_counter >= optim_cfg["early_stopping_patience"]:
                print(f"  -> early stopping (no improvement for {patience_counter} epochs)")
                break

    print("\n=== Loading best checkpoint for final test evaluation ===")
    best = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print("\nFINAL TEST METRICS:")
    print(format_metrics(test_metrics))
    for name, m in test_metrics["per_class"].items():
        print(f"  {name}: P={m['precision']*100:.1f}% R={m['recall']*100:.1f}% "
              f"F1={m['f1']*100:.1f}% Spec={m['specificity']*100:.1f}%")


if __name__ == "__main__":
    main()
