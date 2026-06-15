"""Score a trained detector against a directory of images.

Use cases (all reported in the kit's README):

  # accuracy on the Pico-Banana held-out test split
  python detector/eval.py --ckpt runs/detector/best.pt --split test

  # detection rate on the SynthID test set
  python detector/eval.py --ckpt runs/detector/best.pt --image-dir images/

  # detection rate on regen-attacked versions
  python detector/eval.py --ckpt runs/detector/best.pt --image-dir images/regen_20/

Reports:
   - fraction predicted as label 1 (Nano-Banana / AI-generated)
   - mean & median p(label=1)
   - per-class accuracy (when --split is used and ground truth is known)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (DatasetConfig, PicoBananaSingle, make_splits,
                  _load_and_match_codec, to_tensor_neg1_1)
from models import BinaryClassifier


def score_dir(model, image_dir, size, jpeg_q, device, batch_size=32, glob_pat="*.jpg"):
    paths = sorted(glob.glob(os.path.join(image_dir, glob_pat)))
    if not paths:
        sys.exit(f"no {glob_pat} in {image_dir}")
    tensors = torch.stack([
        to_tensor_neg1_1(_load_and_match_codec(p, size, jpeg_q)) for p in paths
    ])
    preds, probs1 = [], []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            xb = tensors[i:i + batch_size].to(device)
            logits = model(xb)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            preds.append(logits.argmax(dim=1).cpu())
            probs1.append(p1.cpu())
    return paths, torch.cat(preds).numpy(), torch.cat(probs1).numpy()


def score_split(model, split_name, cfg, device, batch_size=32, workers=8):
    train_items, val_items, test_items, _ = make_splits(cfg)
    items = {"train": train_items, "val": val_items, "test": test_items}[split_name]
    ds = PicoBananaSingle(items, cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)
    labels, preds, probs1 = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"]
            logits = model(x)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            labels.append(y); preds.append(logits.argmax(dim=1).cpu()); probs1.append(p1.cpu())
    return (torch.cat(labels).numpy(), torch.cat(preds).numpy(), torch.cat(probs1).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/detector/best.pt")
    ap.add_argument("--image-dir", default=None,
                    help="directory of images to score (no ground truth)")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="evaluate on Pico-Banana split (uses ground truth)")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    args = ap.parse_args()

    if (args.image_dir is None) == (args.split is None):
        sys.exit("specify exactly one of --image-dir or --split")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=device)
    model = BinaryClassifier(pretrained=False, backbone=args.backbone).to(device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}  (val_acc={sd.get('val_acc', '?')}, "
          f"epoch={sd.get('epoch', '?')})\n")

    cfg = DatasetConfig()
    if args.split is not None:
        labels, preds, probs1 = score_split(model, args.split, cfg, device,
                                            args.batch, args.workers)
        n = len(labels)
        acc = (preds == labels).mean()
        for cls, name in enumerate(["real (Open Images)", "edit (NB)"]):
            m = labels == cls
            if m.sum() == 0: continue
            c = (preds[m] == labels[m]).mean()
            print(f"  class {cls} ({name}):  {int(m.sum())} examples,  acc={c:.4f}")
        print(f"  overall: {n} examples,  acc={acc:.4f}")
        print(f"  mean p(NB)={probs1.mean():.4f}  median p(NB)={np.median(probs1):.4f}")
    else:
        paths, preds, probs1 = score_dir(model, args.image_dir,
                                         cfg.image_size, cfg.jpeg_q,
                                         device, args.batch, args.glob)
        n = len(paths)
        n_nb = (preds == 1).sum()
        n_high = (probs1 > 0.9).sum()
        print(f"  {n} images scored from {args.image_dir}")
        print(f"  predicted label=1 (Nano-Banana):  {int(n_nb)}/{n} = {100*n_nb/n:.1f}%")
        print(f"  high-confidence (p>0.9):          {int(n_high)}")
        print(f"  mean p(NB)={probs1.mean():.4f}  median p(NB)={np.median(probs1):.4f}")


if __name__ == "__main__":
    main()
