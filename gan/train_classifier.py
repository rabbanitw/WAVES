"""Train C_eval: ResNet-18 binary classifier on (Open Images orig=0, NB edit=1).

This is the EXTERNAL evaluator for the GAN. It is trained and frozen
before any GAN training; the GAN never sees its weights.

Run:
    python gan/train_classifier.py
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import (
    DatasetConfig, DatasetConfigV2,
    PicoBananaSingle, PicoBananaSingleV2,
    make_splits, make_splits_v2,
)
from models import BinaryClassifier


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/c_eval")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--v2", action="store_true",
                    help="use the larger v2 dataset (preprocessed v1 + photoreal_v2 download)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.v2:
        cfg = DatasetConfigV2()
        train_items, val_items, test_items, _ = make_splits_v2(cfg)
        SingleDS = PicoBananaSingleV2
        print(f"using V2 dataset (combined v1-resized + photoreal_v2 download)")
    else:
        cfg = DatasetConfig()
        train_items, val_items, test_items, _ = make_splits(cfg)
        SingleDS = PicoBananaSingle
    print(f"splits: train={len(train_items)}  val={len(val_items)}  test={len(test_items)} (pairs)")

    train_ds = SingleDS(train_items, cfg)
    val_ds = SingleDS(val_items, cfg)
    test_ds = SingleDS(test_items, cfg)
    print(f"single examples: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    model = BinaryClassifier(pretrained=not args.no_pretrained,
                             backbone=args.backbone).to(device)
    print(f"backbone: {args.backbone}  pretrained: {not args.no_pretrained}")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    log_path = os.path.join(args.out_dir, "log.txt")
    with open(log_path, "w") as flog:
        flog.write("epoch\tstep\ttrain_loss\tval_acc\tval_loss\ttime\n")
        t0 = time.time()
        for epoch in range(args.epochs):
            running = 0.0
            for step, batch in enumerate(train_dl):
                x = batch["img"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                logits = model(x)
                loss = crit(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                running += loss.item()
                if step % 20 == 0:
                    avg = running / max(step + 1, 1)
                    print(f"  epoch {epoch:2d} step {step:4d}/{len(train_dl)}  loss={avg:.4f}", flush=True)
            val_acc, val_loss = evaluate(model, val_dl, device)
            elapsed = time.time() - t0
            print(f"epoch {epoch:2d}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            flog.write(f"{epoch}\t{(epoch+1)*len(train_dl)}\t{running/len(train_dl):.4f}\t{val_acc:.4f}\t{val_loss:.4f}\t{elapsed:.0f}\n")
            flog.flush()
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch},
                           os.path.join(args.out_dir, "best.pt"))
                print(f"  saved best to {args.out_dir}/best.pt", flush=True)

        # final test
        ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_acc, test_loss = evaluate(model, test_dl, device)
        print(f"\ntest_acc={test_acc:.4f}  test_loss={test_loss:.4f}  (best epoch={ckpt['epoch']}  val={ckpt['val_acc']:.4f})")
        flog.write(f"# test_acc={test_acc:.4f}  test_loss={test_loss:.4f}\n")


if __name__ == "__main__":
    main()
