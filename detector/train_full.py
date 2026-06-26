"""Train the SynthID-shaped binary discriminator on the full
photoreal_v2/ harvest (~110k Pico-Banana pairs and counting).

   label 0 = Open Images source photo
   label 1 = Nano-Banana edit

Split: deterministic 75/25 train/test by slot. The training script
holds out the last 5 % of train as a tiny val for picking best ckpt.

Two evaluations at the end:
  * in-distribution test (held-out 25 % of photoreal_v2)
  * OOD synthetic (1,904 Gemini images from DiffusionDB prompts,
    all label=1 – reported as detection rate)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T2

sys.path.insert(0, str(Path(__file__).parent))
from data import (
    DatasetConfig,
    PicoBananaSingle,
    _load_and_match_codec,
    is_photoreal,
    load_metadata,
    to_tensor_neg1_1,
)
from models import BinaryClassifier, count_params

PHOTOREAL_ROOT = "/mnt/data/pico-banana-400k/photoreal_v2"
PHOTOREAL_META = "/mnt/data/pico-banana-400k/photoreal_v2/metadata.jsonl"
OOD_DIR = "/mnt/data/synthid_ood/jpg384"


def split_75_25(items, seed=17):
    items = sorted(items, key=lambda d: d["slot"])
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_test = len(shuffled) // 4
    test = shuffled[:n_test]
    train_all = shuffled[n_test:]
    n_val = max(500, len(train_all) // 20)  # 5 % of train, min 500
    val = train_all[:n_val]
    train = train_all[n_val:]
    return train, val, test


class AugmentedDataset(Dataset):
    """Applies torchvision v2 augmentations to the [-1, 1] tensor from
    PicoBananaSingle. Augs that don't touch the synthetic-vs-natural
    signature: hflip, random crop (scale 0.8-1.0), modest color jitter."""

    def __init__(self, base, image_size):
        self.base = base
        # Geometric augs are tensor-safe in any range. ColorJitter assumes
        # [0, 1] inputs (hue conversion is undefined for negatives), so we
        # split into geom + photometric and normalize for the latter.
        self.geom = T2.Compose([
            T2.RandomHorizontalFlip(p=0.5),
            T2.RandomResizedCrop(size=image_size, scale=(0.8, 1.0),
                                 antialias=True),
        ])
        self.photo = T2.ColorJitter(brightness=0.1, contrast=0.1,
                                    saturation=0.1, hue=0.02)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        d = self.base[k]
        x = self.geom(d["img"])               # [-1, 1] tensor, geom-safe
        x01 = (x + 1.0) / 2.0                  # to [0, 1]
        x01 = x01.clamp_(0.0, 1.0)
        x01 = self.photo(x01)                  # ColorJitter wants [0, 1]
        x01 = x01.clamp_(0.0, 1.0)
        d["img"] = x01 * 2.0 - 1.0             # back to [-1, 1]
        return d


class OODSingle(Dataset):
    """Directory-of-JPEGs dataset, all label=1. Uses the same matched-codec
    preprocessing as PicoBananaSingle so the model sees identically-shaped
    inputs at eval."""

    def __init__(self, dir_, image_size=256, jpeg_q=95):
        self.paths = sorted(p for p in os.listdir(dir_) if p.endswith(".jpg"))
        self.dir = dir_
        self.image_size = image_size
        self.jpeg_q = jpeg_q

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        path = os.path.join(self.dir, self.paths[k])
        img = _load_and_match_codec(path, self.image_size, self.jpeg_q)
        return {"img": to_tensor_neg1_1(img), "label": 1, "fname": self.paths[k]}


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    pos_correct = pos_total = 0
    neg_correct = neg_total = 0
    loss_sum = 0.0
    for batch in loader:
        x = batch["img"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        pos_mask = (y == 1)
        pos_total += pos_mask.sum().item()
        pos_correct += ((pred == y) & pos_mask).sum().item()
        neg_mask = (y == 0)
        neg_total += neg_mask.sum().item()
        neg_correct += ((pred == y) & neg_mask).sum().item()
    model.train()
    return {
        "acc": correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "tpr": pos_correct / pos_total if pos_total else 0.0,  # NB-detection
        "tnr": neg_correct / neg_total if neg_total else 0.0,  # natural-photo correct-reject
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/full_75_25")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--aug", action="store_true",
                    help="enable hflip+crop+color-jitter augmentation")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--warmup-pct", type=float, default=0.0,
                    help="linear warmup over this fraction of total steps")
    ap.add_argument("--constant-lr", action="store_true",
                    help="skip cosine decay; constant LR throughout (still respects warmup)")
    ap.add_argument("--eval-every-steps", type=int, default=0,
                    help="if >0, run val/test/OOD eval every N optimizer steps "
                         "(in addition to the end-of-epoch eval) and append to "
                         "trajectory.csv in out-dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DatasetConfig(
        root=PHOTOREAL_ROOT,
        metadata=PHOTOREAL_META,
        image_size=args.image_size,
        jpeg_q=95,
        photoreal_only=True,
    )

    items = load_metadata(cfg.metadata, photoreal_only=cfg.photoreal_only)
    print(f"loaded {len(items)} photoreal_v2 pairs", flush=True)
    train_items, val_items, test_items = split_75_25(items, seed=args.seed)
    print(f"splits (pairs): train={len(train_items)}  val={len(val_items)}  "
          f"test={len(test_items)}", flush=True)

    train_ds_raw = PicoBananaSingle(train_items, cfg)
    train_ds = AugmentedDataset(train_ds_raw, cfg.image_size) if args.aug else train_ds_raw
    val_ds = PicoBananaSingle(val_items, cfg)
    test_ds = PicoBananaSingle(test_items, cfg)
    ood_ds = OODSingle(OOD_DIR, image_size=cfg.image_size, jpeg_q=cfg.jpeg_q)
    print(f"single examples: train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  ood={len(ood_ds)}  aug={args.aug}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    ood_dl = DataLoader(ood_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = BinaryClassifier(pretrained=True, backbone=args.backbone).to(device)
    print(f"backbone={args.backbone}  params={count_params(model)/1e6:.2f}M", flush=True)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    warmup_steps = max(1, int(args.warmup_pct * total_steps))
    if args.constant_lr:
        if warmup_steps > 1:
            sched = LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / warmup_steps))
        else:
            sched = LambdaLR(opt, lr_lambda=lambda s: 1.0)
    elif warmup_steps > 1:
        warmup = LambdaLR(opt, lr_lambda=lambda s: (s + 1) / warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
        sched = SequentialLR(opt, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])
    else:
        sched = CosineAnnealingLR(opt, T_max=total_steps)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"opt: lr={args.lr}  wd={args.weight_decay}  ls={args.label_smoothing}  "
          f"warmup_steps={warmup_steps}/{total_steps}  aug={args.aug}", flush=True)

    best_val = 0.0
    log_path = os.path.join(args.out_dir, "log.txt")
    flog = open(log_path, "w")
    flog.write("epoch\tstep\ttrain_loss\tval_acc\tval_loss\ttest_acc\ttest_tpr\ttest_tnr\tood_detect\ttime\n")
    # Mid-epoch trajectory log
    traj_path = os.path.join(args.out_dir, "trajectory.csv")
    ftraj = None
    if args.eval_every_steps > 0:
        ftraj = open(traj_path, "w")
        ftraj.write("global_step,samples_seen,train_loss_recent,"
                    "val_acc,test_acc,test_tpr,test_tnr,ood_detect,elapsed_s\n")
        ftraj.flush()
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        n_seen = 0
        recent_loss = 0.0
        recent_n = 0
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
            recent_loss += loss.item()
            n_seen += 1
            recent_n += 1
            global_step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} step {step:5d}/{len(train_dl)}  "
                      f"loss={loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"({el:.0f}s)", flush=True)
            if ftraj is not None and global_step % args.eval_every_steps == 0:
                vt = evaluate_acc(model, val_dl, device)
                tt = evaluate_acc(model, test_dl, device)
                ot = evaluate_acc(model, ood_dl, device)
                elapsed = time.time() - t0
                samples_seen = global_step * args.batch
                ftraj.write(f"{global_step},{samples_seen},"
                            f"{recent_loss/max(1,recent_n):.4f},"
                            f"{vt['acc']:.4f},{tt['acc']:.4f},"
                            f"{tt['tpr']:.4f},{tt['tnr']:.4f},"
                            f"{ot['tpr']:.4f},{elapsed:.0f}\n")
                ftraj.flush()
                print(f"  [TRAJ] step {global_step}  samples={samples_seen}  "
                      f"val={vt['acc']:.4f}  test={tt['acc']:.4f}  "
                      f"TPR={tt['tpr']:.4f}  TNR={tt['tnr']:.4f}  "
                      f"OOD={ot['tpr']:.4f}  ({elapsed:.0f}s)", flush=True)
                recent_loss, recent_n = 0.0, 0
        val_m = evaluate_acc(model, val_dl, device)
        test_m = evaluate_acc(model, test_dl, device)
        ood_m = evaluate_acc(model, ood_dl, device)
        elapsed = time.time() - t0
        print(f"epoch {epoch:2d}  train_loss={running/n_seen:.4f}  "
              f"val_acc={val_m['acc']:.4f}  "
              f"test_acc={test_m['acc']:.4f}  (TPR={test_m['tpr']:.4f}, TNR={test_m['tnr']:.4f})  "
              f"OOD_detect={ood_m['tpr']:.4f}  ({elapsed:.0f}s)",
              flush=True)
        flog.write(f"{epoch}\t{(epoch+1)*len(train_dl)}\t{running/n_seen:.4f}\t"
                   f"{val_m['acc']:.4f}\t{val_m['loss']:.4f}\t"
                   f"{test_m['acc']:.4f}\t{test_m['tpr']:.4f}\t{test_m['tnr']:.4f}\t"
                   f"{ood_m['tpr']:.4f}\t{elapsed:.0f}\n")
        flog.flush()
        if val_m["acc"] > best_val:
            best_val = val_m["acc"]
            torch.save({
                "state_dict": model.state_dict(),
                "val_acc": val_m["acc"], "epoch": epoch, "args": vars(args),
                "test_id": test_m, "test_ood": ood_m,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best.pt at val_acc={val_m['acc']:.4f}", flush=True)
    flog.close()
    if ftraj is not None:
        ftraj.close()

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\nLoaded best.pt (epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f})", flush=True)

    print("\n--- in-distribution test ---", flush=True)
    test_m = evaluate_acc(model, test_dl, device)
    print(f"  n={test_m['n']}  acc={test_m['acc']:.4f}  loss={test_m['loss']:.4f}  "
          f"NB-detection (TPR)={test_m['tpr']:.4f}  "
          f"natural-correct-reject (TNR)={test_m['tnr']:.4f}", flush=True)

    print("\n--- OOD synthetic test (all label=1) ---", flush=True)
    ood_m = evaluate_acc(model, ood_dl, device)
    print(f"  n={ood_m['n']}  OOD-NB-detection rate={ood_m['tpr']:.4f}  "
          f"loss={ood_m['loss']:.4f}", flush=True)

    summary = {
        "best_epoch": ckpt["epoch"],
        "best_val_acc": ckpt["val_acc"],
        "n_train_pairs": len(train_items),
        "n_val_pairs": len(val_items),
        "n_test_pairs": len(test_items),
        "test_id": test_m,
        "test_ood": ood_m,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
