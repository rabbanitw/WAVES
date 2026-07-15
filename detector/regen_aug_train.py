"""Regen-augmented training: mix in pre-generated SDXL 1x10 (strength=0.01)
regens of a training subset. Labels inherit from source (regen'd natural
photos = label 0, regen'd Nano-Banana edits = label 1). Standard CE training
on the concatenated dataset. No PGD, no fancy losses.

The point: force the detector to learn features that survive an SDXL
DDIM roundtrip, so at inference a regen-attacked image still classifies
correctly.

Eval:
  - In-dist test (Pico-Banana photoreal)
  - Clean OOD-DiffDB (synth)
  - Clean OOD-COCO-edit (synth)
  - Clean OOD-COCO-nat  (nat)
  - Regen'd OOD-COCO-edit (synth, attacked)
  - Regen'd OOD-COCO-nat  (nat, attacked)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from data import (DatasetConfig, PicoBananaSingle, load_metadata,
                  to_tensor_neg1_1)
from models import BinaryClassifier, count_params
from train_full import (COCO_NAT_DIR, OOD2_DIR, OOD_DIR, OODSingle,
                        PHOTOREAL_META, PHOTOREAL_ROOT, evaluate_acc,
                        split_80_20)
evaluate_clean = evaluate_acc

REGEN_ROOT = Path("/mnt/data/pico_regen_10step_v1")


class GeomAugWrapper(Dataset):
    """Wraps a Dataset and applies random geometric augmentation to the
    'img' tensor with probability `p`. Ops: random-center crop (60-100%),
    random rotation (up to +/- max_deg), random downsample+upsample.
    Fills a gap in regen-aug training which is geometry-invariant."""

    def __init__(self, base, p=0.5, max_rot_deg=10, min_crop_keep=0.6,
                 min_scale=0.3, image_size=384):
        self.base = base
        self.p = p
        self.max_rot_deg = max_rot_deg
        self.min_crop_keep = min_crop_keep
        self.min_scale = min_scale
        self.image_size = image_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        d = self.base[k]
        if random.random() >= self.p:
            return d
        # Convert [-1, 1] tensor back to PIL for geometric ops
        x = d["img"]  # C, H, W in [-1, 1]
        import numpy as np
        arr = ((x + 1.0) / 2.0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(arr)
        # Pick one of the three ops uniformly
        op = random.choice(["crop", "rotate", "down_up"])
        if op == "crop":
            keep = random.uniform(self.min_crop_keep, 1.0)
            w, h = pil.size
            kw, kh = int(w * keep), int(h * keep)
            left = random.randint(0, w - kw)
            top  = random.randint(0, h - kh)
            pil = pil.crop((left, top, left + kw, top + kh)).resize(
                (w, h), Image.LANCZOS)
        elif op == "rotate":
            deg = random.uniform(-self.max_rot_deg, self.max_rot_deg)
            pil = pil.rotate(deg, resample=Image.BILINEAR, expand=False)
        else:  # down_up
            f = random.uniform(self.min_scale, 1.0)
            w, h = pil.size
            nw, nh = max(1, int(w * f)), max(1, int(h * f))
            pil = pil.resize((nw, nh), Image.LANCZOS).resize((w, h), Image.LANCZOS)
        d = dict(d)
        d["img"] = to_tensor_neg1_1(pil)
        return d


class _StripSlot(Dataset):
    """Wrap a Dataset that returns {'img','label','slot',...} and drop
    everything except img+label so ConcatDataset works with siblings that
    only have img+label."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, k):
        d = self.base[k]
        return {"img": d["img"], "label": d["label"]}


class RegenDirDataset(Dataset):
    """Loads pre-generated regen images from a directory with a fixed label."""

    def __init__(self, dir_, label, image_size=384, jpeg_q=95):
        self.dir = Path(dir_)
        self.paths = sorted(p for p in self.dir.iterdir() if p.suffix == ".jpg")
        self.label = label
        self.image_size = image_size
        self.jpeg_q = jpeg_q

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        # Pre-gen'd images are already at 384x384 q95, but pass through the
        # same matched-codec pipeline for safety.
        import io as _io
        pil = Image.open(self.paths[k]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC)
        buf = _io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_q)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        return {"img": to_tensor_neg1_1(pil), "label": self.label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/regen_aug")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--constant-lr", action="store_true")
    ap.add_argument("--geom-aug-p", type=float, default=0.0,
                    help="If >0, wrap the training set with GeomAugWrapper "
                         "and apply crop/rotate/downsample at this per-sample "
                         "probability. Closes the geometric-transform "
                         "generalization gap left by regen-only augmentation.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda"

    cfg = DatasetConfig(
        root=PHOTOREAL_ROOT, metadata=PHOTOREAL_META,
        image_size=args.image_size, jpeg_q=95, photoreal_only=True,
    )
    items = load_metadata(cfg.metadata, photoreal_only=cfg.photoreal_only)
    train_items, test_items = split_80_20(items, seed=args.seed)
    print(f"loaded {len(items)} pairs  split: train={len(train_items)} "
          f"test={len(test_items)}", flush=True)

    # Original in-distribution training data
    train_ds_orig = _StripSlot(PicoBananaSingle(train_items, cfg))
    test_ds = _StripSlot(PicoBananaSingle(test_items, cfg))

    # Regen'd training samples (both label sides)
    train_regen_nat = RegenDirDataset(REGEN_ROOT / "train_natural", label=0,
                                      image_size=args.image_size)
    train_regen_syn = RegenDirDataset(REGEN_ROOT / "train_synth", label=1,
                                      image_size=args.image_size)
    train_ds = ConcatDataset([train_ds_orig, train_regen_nat, train_regen_syn])
    print(f"train mix: orig={len(train_ds_orig)}  "
          f"regen_natural={len(train_regen_nat)}  "
          f"regen_synth={len(train_regen_syn)}  total={len(train_ds)}",
          flush=True)
    if args.geom_aug_p > 0:
        train_ds = GeomAugWrapper(train_ds, p=args.geom_aug_p,
                                  image_size=args.image_size)
        print(f"WRAPPING TRAIN with GeomAugWrapper p={args.geom_aug_p} "
              f"(random crop/rotate/downsample)", flush=True)

    # Standard OOD eval sets
    ood_ds = OODSingle(OOD_DIR, image_size=cfg.image_size,
                       jpeg_q=cfg.jpeg_q, label=1)
    ood2_ds = OODSingle(OOD2_DIR, image_size=cfg.image_size,
                        jpeg_q=cfg.jpeg_q, label=1)
    coco_nat_ds = OODSingle(COCO_NAT_DIR, image_size=cfg.image_size,
                            jpeg_q=cfg.jpeg_q, label=0)

    # Regen'd OOD eval sets
    ood2_regen_ds = RegenDirDataset(REGEN_ROOT / "ood_coco_edit", label=1,
                                    image_size=args.image_size)
    coco_nat_regen_ds = RegenDirDataset(REGEN_ROOT / "ood_coco_nat", label=0,
                                        image_size=args.image_size)
    print(f"eval: test={len(test_ds)}  DiffDB={len(ood_ds)}  "
          f"OOD-COCO-edit clean={len(ood2_ds)} regen'd={len(ood2_regen_ds)}  "
          f"OOD-COCO-nat  clean={len(coco_nat_ds)} regen'd={len(coco_nat_regen_ds)}",
          flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    def _dl(ds):
        return DataLoader(ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    test_dl = _dl(test_ds)
    ood_dl  = _dl(ood_ds)
    ood2_dl = _dl(ood2_ds)
    coco_nat_dl = _dl(coco_nat_ds)
    ood2_regen_dl = _dl(ood2_regen_ds)
    coco_nat_regen_dl = _dl(coco_nat_regen_ds)

    model = BinaryClassifier(pretrained=True, backbone=args.backbone).to(device)
    print(f"backbone={args.backbone}  params={count_params(model)/1e6:.2f}M",
          flush=True)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    sched = (LambdaLR(opt, lr_lambda=lambda s: 1.0) if args.constant_lr
             else CosineAnnealingLR(opt, T_max=total_steps))
    crit = nn.CrossEntropyLoss()

    best_score = -2.0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_seen = 0
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
            n_seen += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} step {step:5d}/{len(train_dl)}  "
                      f"loss={loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"({el:.0f}s)", flush=True)

        # ---- eval ----
        model.eval()
        test_m = evaluate_clean(model, test_dl, device)
        ood_m  = evaluate_clean(model, ood_dl, device)
        ood2_m = evaluate_clean(model, ood2_dl, device)
        coconat_m = evaluate_clean(model, coco_nat_dl, device)
        ood2_regen_m = evaluate_clean(model, ood2_regen_dl, device)
        coconat_regen_m = evaluate_clean(model, coco_nat_regen_dl, device)
        bal_clean = ood2_m["tpr"] + coconat_m["tnr"] - 1.0
        bal_regen = ood2_regen_m["tpr"] + coconat_regen_m["tnr"] - 1.0
        score = (bal_clean + bal_regen) / 2.0
        elapsed = time.time() - t0
        print(f"\nepoch {epoch}: test={test_m['acc']:.4f} "
              f"(TPR={test_m['tpr']:.4f}, TNR={test_m['tnr']:.4f})  "
              f"DiffDB={ood_m['tpr']:.4f}",
              flush=True)
        print(f"  CLEAN  OOD-COCO-edit={ood2_m['tpr']:.4f}  "
              f"COCO-nat-TNR={coconat_m['tnr']:.4f}  bal={bal_clean:+.4f}",
              flush=True)
        print(f"  REGEN  OOD-COCO-edit={ood2_regen_m['tpr']:.4f}  "
              f"COCO-nat-TNR={coconat_regen_m['tnr']:.4f}  bal={bal_regen:+.4f}  "
              f"combined={score:+.4f}  ({elapsed:.0f}s)", flush=True)

        if score > best_score:
            best_score = score
            torch.save({
                "state_dict": model.state_dict(),
                "score": score, "bal_clean": bal_clean, "bal_regen": bal_regen,
                "epoch": epoch, "args": vars(args),
                "test_id": test_m, "diffdb": ood_m,
                "coco_edit_clean": ood2_m, "coco_nat_clean": coconat_m,
                "coco_edit_regen": ood2_regen_m, "coco_nat_regen": coconat_regen_m,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best.pt at combined score={score:+.4f}", flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "seed": args.seed, "backbone": args.backbone,
            "best_score": best_score,
            "n_train_pairs": len(train_items),
            "regen_train_samples": len(train_regen_nat) + len(train_regen_syn),
        }, f, indent=2)
    print(f"\ndone. best combined score = {best_score:+.4f}", flush=True)


if __name__ == "__main__":
    main()
