"""
Stage 1 ONLY (Section III-C.c): backbone frozen, train ONLY the K
feature-domain diffusion modules against the (static) EMA teacher
(Eq 10-12).

Split out from the combined two-stage script so Stage 1 can be run,
checkpointed, and the session closed -- Stage 2 picks up later from the
saved checkpoint via train_stage2.py.

Requires precomputed spectrograms from precompute_stabilized.py.

Usage:
    python train_stage1.py --config ../../configs/sleepedf78.yaml \
        --cache_dir ../../cached_spectrograms \
        --out /kaggle/working/stage1_checkpoint.pt
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.dssnet import DSSNet
from src.training.dataset import CachedSpectrogramDataset, make_balanced_sampler
from src.config_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out", default="/kaggle/working/stage1_checkpoint.pt")
    parser.add_argument("--resume", action="store_true",
                         help="resume Stage 1 from --out if it already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optim_cfg = cfg["optim"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = CachedSpectrogramDataset(os.path.join(args.cache_dir, "train_spectrograms.pt"))
    print(f"Train epochs (cached spectrograms): {len(train_ds)}")

    batch_size = optim_cfg["batch_size"]
    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=0, drop_last=True, pin_memory=True)

    model = DSSNet(cfg).to(device)

    start_epoch = 0
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed Stage 1 from epoch {start_epoch} (loss={ckpt.get('loss', 'n/a')})")

    print("\n=== STAGE 1: training feature-diffusion modules only ===")
    model.set_stage1_trainable()
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=optim_cfg["lr"]
    )
    scaler1 = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    total_epochs = optim_cfg["stage1_epochs"]
    for epoch in range(start_epoch, total_epochs):
        model.train()
        model.vit_student.eval()
        losses = []
        for spec, _ in train_loader:
            spec = spec.to(device, non_blocking=True)
            opt1.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = model.forward_stage1_from_spec(spec)
            scaler1.scale(loss).backward()
            scaler1.step(opt1)
            scaler1.update()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"[Stage1] Epoch {epoch+1}/{total_epochs}  loss={avg_loss:.4f}")

        # Save after EVERY epoch so an interruption never loses more than
        # one epoch's progress, and --resume always has something to load.
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
        }, args.out)
        print(f"  -> saved Stage 1 checkpoint to {args.out}")

    print(f"\nStage 1 complete. Final checkpoint: {args.out}")
    print("Run train_stage2.py next, pointing --stage1_ckpt at this file.")


if __name__ == "__main__":
    main()
