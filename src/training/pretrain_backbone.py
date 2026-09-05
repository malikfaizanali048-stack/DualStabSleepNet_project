"""
Backbone pretraining -- the missing piece diagnosed as the likely cause
of very low Stage 2 accuracy (54.6% test ACC vs paper's 88%).

Trains ONLY the ViT student + classifier via plain cross-entropy, with
NO feature-domain diffusion involved. This MUST run before Stage 1,
because Stage 1 trains feature-diffusion modules to project features
"toward the teacher-defined manifold" (paper III-C.a) -- which is
meaningless if the teacher (EMA of the student) is still random,
untrained noise, as it was in the original run. This step gives the
backbone -- and therefore the teacher -- something real to represent.

After this, train_stage1.py loads this checkpoint via --backbone_ckpt,
which also calls sync_teacher_to_student() so the teacher mirrors the
now-pretrained backbone instead of starting from random init.

Usage:
    python pretrain_backbone.py --config ../../configs/sleepedf78.yaml \
        --cache_dir ../../cached_spectrograms \
        --out /kaggle/working/backbone_pretrained.pt
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
        logits = model.forward_pretrain_from_spec(spec)
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
    parser.add_argument("--out", default="/kaggle/working/backbone_pretrained.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true",
                         help="resume from --out if it already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optim_cfg = cfg["optim"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = CachedSpectrogramDataset(os.path.join(args.cache_dir, "train_spectrograms.pt"))
    val_ds = CachedSpectrogramDataset(os.path.join(args.cache_dir, "val_spectrograms.pt"))
    print(f"Train epochs: {len(train_ds)}  Val epochs: {len(val_ds)}")

    batch_size = optim_cfg["batch_size"]
    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = DSSNet(cfg).to(device)
    model.set_pretrain_trainable()

    start_epoch = 0
    best_val_macro_f1 = -1.0
    patience_counter = 0
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_macro_f1 = ckpt["val_metrics"]["macro_f1"]
        print(f"Resumed backbone pretraining from epoch {start_epoch} "
              f"(best val macro-F1={best_val_macro_f1*100:.2f}%)")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=optim_cfg["lr"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    patience = optim_cfg.get("early_stopping_patience", 15)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for spec, y in train_loader:
            spec, y = spec.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model.forward_pretrain_from_spec(spec)
                loss = torch.nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        print(f"[Pretrain] Epoch {epoch+1}/{args.epochs}  "
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
            if patience_counter >= patience:
                print(f"  -> early stopping (no improvement for {patience_counter} epochs)")
                break

    print(f"\nBackbone pretraining complete. Best val macro-F1={best_val_macro_f1*100:.2f}%. "
          f"Checkpoint: {args.out}")
    print("Run train_stage1.py next, pointing --backbone_ckpt at this file.")


if __name__ == "__main__":
    main()