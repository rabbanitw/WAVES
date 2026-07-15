"""Train the detector on PSNR-filtered pairs only. High-PSNR pairs are edits
where the natural photo and the NB output are nearly pixel-identical, so any
learnable signal has to come from the diffusion fingerprint substrate itself,
not semantic content differences.

Test hypothesis: does filtered training produce a detector that (a) still
generalizes to OOD-COCO edits, and (b) shifts what feature the model uses?"""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data import DatasetConfig, PicoBananaSingle, load_metadata
from models import BinaryClassifier, count_params
from train_full import (COCO_NAT_DIR, OOD2_DIR, OOD_DIR, OODSingle,
                        PHOTOREAL_META, PHOTOREAL_ROOT, evaluate_acc,
                        split_80_20)

evaluate_clean = evaluate_acc
PSNR_JSON = "/mnt/data/pico_pair_psnr_photoreal_v2.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/psnr_filtered")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--constant-lr", action="store_true")
    ap.add_argument("--min-psnr", type=float, required=True,
                    help="Only train on pairs with PSNR(orig, edit) >= this")
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

    # Filter by PSNR
    psnr_map = {str(k): float(v) for k, v in json.load(open(PSNR_JSON)).items()}
    kept = [d for d in items if psnr_map.get(str(d["slot"]), -1) >= args.min_psnr]
    print(f"pairs total={len(items)}  after PSNR>={args.min_psnr}: {len(kept)}",
          flush=True)
    train_items, test_items = split_80_20(kept, seed=args.seed)
    print(f"train={len(train_items)}  test={len(test_items)}", flush=True)

    train_ds = PicoBananaSingle(train_items, cfg)
    test_ds = PicoBananaSingle(test_items, cfg)
    # OOD eval sets (unfiltered)
    ood_ds = OODSingle(OOD_DIR, image_size=cfg.image_size,
                       jpeg_q=cfg.jpeg_q, label=1)
    ood2_ds = OODSingle(OOD2_DIR, image_size=cfg.image_size,
                        jpeg_q=cfg.jpeg_q, label=1)
    coco_nat_ds = OODSingle(COCO_NAT_DIR, image_size=cfg.image_size,
                            jpeg_q=cfg.jpeg_q, label=0)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)

    def _dl(ds):
        return DataLoader(ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    test_dl = _dl(test_ds)
    ood_dl = _dl(ood_ds)
    ood2_dl = _dl(ood2_ds)
    coco_nat_dl = _dl(coco_nat_ds)

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
        for step, batch in enumerate(train_dl):
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} step {step:5d}/{len(train_dl)}  "
                      f"loss={loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"({el:.0f}s)", flush=True)

        # eval
        model.eval()
        test_m = evaluate_clean(model, test_dl, device)
        ood_m = evaluate_clean(model, ood_dl, device)
        ood2_m = evaluate_clean(model, ood2_dl, device)
        coconat_m = evaluate_clean(model, coco_nat_dl, device)
        bal = ood2_m["tpr"] + coconat_m["tnr"] - 1.0
        elapsed = time.time() - t0
        print(f"\nepoch {epoch}: test={test_m['acc']:.4f} "
              f"(TPR={test_m['tpr']:.4f}, TNR={test_m['tnr']:.4f})", flush=True)
        print(f"  DiffDB={ood_m['tpr']:.4f}  COCO-edit={ood2_m['tpr']:.4f}  "
              f"COCO-nat-TNR={coconat_m['tnr']:.4f}  bal={bal:+.4f}  "
              f"({elapsed:.0f}s)", flush=True)

        if bal > best_score:
            best_score = bal
            torch.save({
                "state_dict": model.state_dict(),
                "score": bal, "epoch": epoch, "args": vars(args),
                "min_psnr": args.min_psnr,
                "n_train_pairs": len(train_items),
                "test_id": test_m, "diffdb": ood_m,
                "coco_edit": ood2_m, "coco_nat": coconat_m,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best.pt at bal={bal:+.4f}", flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "seed": args.seed, "backbone": args.backbone,
            "min_psnr": args.min_psnr,
            "best_score": best_score,
            "n_train_pairs": len(train_items),
        }, f, indent=2)
    print(f"\ndone. best bal = {best_score:+.4f}", flush=True)


if __name__ == "__main__":
    main()
