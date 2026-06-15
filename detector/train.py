"""Train the SynthID-shaped binary detector.

The classifier learns to distinguish:
   label 0 = real photograph  (Open Images source)
   label 1 = Nano-Banana edit

on the Pico-Banana-400K SFT split (filtered to photoreal edit_types only),
with matched JPEG q=95 codec preprocessing applied to both classes
inside the dataloader.

Run:
   python detector/train.py --epochs 20 --batch 64 --lr 1e-4

Reads the dataset locations from DatasetConfig in detector/data.py;
edit those to point at your local copy of Pico-Banana-400K.
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

from data import DatasetConfig, PicoBananaSingle, make_splits
from models import BinaryClassifier, count_params


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
    ap.add_argument("--out-dir", default="runs/detector")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DatasetConfig()
    train_items, val_items, test_items, _ = make_splits(cfg)
    print(f"splits: train={len(train_items)}  val={len(val_items)}  test={len(test_items)} (pairs)")

    train_ds = PicoBananaSingle(train_items, cfg)
    val_ds = PicoBananaSingle(val_items, cfg)
    test_ds = PicoBananaSingle(test_items, cfg)
    print(f"single examples: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    model = BinaryClassifier(pretrained=True, backbone=args.backbone).to(device)
    print(f"backbone={args.backbone}  params={count_params(model)/1e6:.2f}M")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            val_acc, val_loss = evaluate(model, val_dl, device)
            elapsed = time.time() - t0
            print(f"epoch {epoch:2d}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)",
                  flush=True)
            flog.write(f"{epoch}\t{(epoch+1)*len(train_dl)}\t{running/len(train_dl):.4f}\t"
                       f"{val_acc:.4f}\t{val_loss:.4f}\t{elapsed:.0f}\n")
            flog.flush()
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch,
                            "args": vars(args)},
                           os.path.join(args.out_dir, "best.pt"))
                print(f"  -> saved best.pt at val_acc={val_acc:.4f}", flush=True)

        ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_acc, test_loss = evaluate(model, test_dl, device)
        print(f"\ntest_acc={test_acc:.4f}  test_loss={test_loss:.4f}  "
              f"(best epoch={ckpt['epoch']}  val={ckpt['val_acc']:.4f})")


if __name__ == "__main__":
    main()
