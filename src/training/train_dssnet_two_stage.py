"""
Stage 1 + Stage 2 training for the full DSSNet (Section III-C.c):
  Stage 1: backbone frozen, train ONLY the K feature-domain diffusion
           modules against the (static) EMA teacher (Eq 10-12).
  Stage 2: unfreeze everything, joint objective L_total = L_cls + lambda*L_feat
           (Eq 13), EMA teacher updated every step.

Requires a data-domain diffusion checkpoint already trained via
train_data_diffusion.py (loaded frozen, never updated here -- paper III-B).

Checkpoint selection: best validation macro-F1 (documented assumption --
paper doesn't state its selection rule explicitly; best-val is standard
practice and avoids reporting a lucky final-epoch number).

Usage:
    python train_dssnet_two_stage.py --config ../../configs/sleepedf78.yaml \
        --data_diffusion_ckpt ../../data_diffusion_best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.dssnet import DSSNet
from src.training.dataset import (
    load_manifest, subject_wise_split, SleepEpochDataset,
    compute_normalization_stats, make_balanced_sampler,
)
from src.eval.metrics import compute_metrics, format_metrics
from src.config_utils import load_config


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model.forward_infer(x)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_diffusion_ckpt", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    optim_cfg = cfg["optim"]
    fm_cfg = cfg["diffusion"]["feature_domain"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    subjects = load_manifest(ds_cfg["processed_dir"])
    train_subj, val_subj, test_subj = subject_wise_split(subjects, seed=42)
    print(f"Subjects -- train:{len(train_subj)} val:{len(val_subj)} test:{len(test_subj)}")

    train_ds = SleepEpochDataset(ds_cfg["processed_dir"], train_subj)
    val_ds = SleepEpochDataset(ds_cfg["processed_dir"], val_subj)
    test_ds = SleepEpochDataset(ds_cfg["processed_dir"], test_subj)

    # Normalization stats: reuse the SAME stats the data-domain diffusion
    # module was trained with (loaded from its checkpoint), so the frozen
    # denoiser sees inputs on the scale it was trained on.
    ddm_ckpt = torch.load(args.data_diffusion_ckpt, map_location="cpu", weights_only=False)
    mean, std = ddm_ckpt["norm_mean"], ddm_ckpt["norm_std"]
    for ds in (train_ds, val_ds, test_ds):
        ds.set_normalization(mean, std)
    print(f"Epochs -- train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    batch_size = optim_cfg["batch_size"]
    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=4, drop_last=True, pin_memory=True,
                               persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    # --- Model ---
    model = DSSNet(cfg).to(device)
    model.load_frozen_data_denoiser(ddm_ckpt["state_dict"])
    print("Loaded frozen data-domain diffusion checkpoint "
          f"(val_loss={ddm_ckpt.get('val_loss', 'n/a')}).")

    out_dir = args.out or os.path.join(ds_cfg["processed_dir"], "..")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "dssnet_best.pt")

    # ============================ STAGE 1 ============================
    print("\n=== STAGE 1: training feature-diffusion modules only ===")
    model.set_stage1_trainable()
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=optim_cfg["lr"]
    )
    scaler1 = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    for epoch in range(optim_cfg["stage1_epochs"]):
        model.train()
        # keep frozen submodules in eval() so e.g. dropout/BN in the
        # (frozen) student backbone doesn't add train-time noise
        model.vit_student.eval()
        losses = []
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            opt1.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = model.forward_stage1(x)
            scaler1.scale(loss).backward()
            scaler1.step(opt1)
            scaler1.update()
            losses.append(loss.item())
        print(f"[Stage1] Epoch {epoch+1}/{optim_cfg['stage1_epochs']}  "
              f"loss={sum(losses)/len(losses):.4f}")

    # ============================ STAGE 2 ============================
    print("\n=== STAGE 2: joint fine-tuning ===")
    model.set_stage2_trainable()
    opt2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=optim_cfg["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=optim_cfg["stage2_epochs"])
    scaler2 = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_macro_f1 = -1.0
    patience_counter = 0
    for epoch in range(optim_cfg["stage2_epochs"]):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt2.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits, feat_reg_loss = model.forward_stage2(x)
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
        print(f"[Stage2] Epoch {epoch+1}/{optim_cfg['stage2_epochs']}  "
              f"train_loss={sum(losses)/len(losses):.4f}  "
              f"val: {format_metrics(val_metrics)}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> new best (val macro-F1={best_val_macro_f1*100:.2f}%), saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= optim_cfg["early_stopping_patience"]:
                print(f"  -> early stopping (no improvement for {patience_counter} epochs)")
                break

    # --- Final test-set evaluation using BEST checkpoint (not last epoch) ---
    print("\n=== Loading best checkpoint for final test evaluation ===")
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print("\nFINAL TEST METRICS:")
    print(format_metrics(test_metrics))
    for name, m in test_metrics["per_class"].items():
        print(f"  {name}: P={m['precision']*100:.1f}% R={m['recall']*100:.1f}% "
              f"F1={m['f1']*100:.1f}% Spec={m['specificity']*100:.1f}%")


if __name__ == "__main__":
    main()