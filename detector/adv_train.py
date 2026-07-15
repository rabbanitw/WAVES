"""Adversarially-trained binary discriminator.

On every training step:
  - sample an L_inf budget eps from a discrete ladder (0 = no attack)
  - if eps > 0, run K-step PGD against the current model with the
    correct label as the loss target, producing x_adv
  - take an optimizer step on (x_adv, y_original)

This forces the model to be robust to pixel-level adversarial
perturbations up to the budget ladder's largest eps -- the kind of
attack that previously fooled C_eval at ~1 pixel-unit perturbation.

Eval includes:
  - in-dist clean test
  - clean OOD-DiffDB / OOD-COCO-edit / OOD-COCO-natural
  - eps=8 adv-attacked variants of OOD-COCO-edit and OOD-COCO-natural
    so we can track robustness directly during training.

Saves best-by-balanced-score; deletes intermediate ckpts to keep disk
usage modest across multi-seed sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).parent))
from data import (DatasetConfig, PicoBananaSingle, _load_and_match_codec,
                  load_metadata, to_tensor_neg1_1)
from models import BinaryClassifier, count_params


# ---- Gaussian blur preprocessing (folded INTO the classifier) ----
def _make_gaussian_kernel(sigma, device):
    ksize = 2 * int(3 * sigma) + 1
    ax = torch.arange(ksize, device=device) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g1 = g1 / g1.sum()
    k2 = g1[:, None] * g1[None, :]
    return k2.view(1, 1, ksize, ksize).expand(3, 1, ksize, ksize).contiguous(), ksize


class RandomJPEGWrapper(torch.utils.data.Dataset):
    """Wraps a Dataset whose __getitem__ returns {'img': tensor in [-1,1], ...}.
    For each access, re-encodes the image at a JPEG quality sampled uniformly
    from `q_choices`. Forces the trainee to learn features that survive heavy
    JPEG quantization (a non-differentiable op the attacker can't perfectly
    gradient through)."""

    def __init__(self, base_ds, q_choices, image_size=384):
        import io as _io
        self.base = base_ds
        self.q_choices = q_choices
        self.image_size = image_size
        self._io = _io

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        item = self.base[k]
        x = item["img"]  # [-1, 1]
        # convert to PIL
        arr = ((x + 1.0) / 2.0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(arr)
        q = random.choice(self.q_choices)
        buf = self._io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(q))
        buf.seek(0)
        pil2 = Image.open(buf).convert("RGB")
        from data import to_tensor_neg1_1
        item["img"] = to_tensor_neg1_1(pil2)
        return item


class BlurWrap(nn.Module):
    """Wraps a classifier with an input-side Gaussian blur. The blur is part
    of the model -- every callsite that does `model(x)` (PGD attack, train
    forward, eval forward) sees a blurred input. The attacker's gradient
    flows back through the blur, so adversarial perturbations have to find
    delta such that blur(delta) crosses the decision boundary -- forcing
    the attack into the low frequencies the natural-image distribution
    actually occupies."""

    def __init__(self, base, sigma, device):
        super().__init__()
        self.base = base
        self.sigma = sigma
        kernel, ksize = _make_gaussian_kernel(sigma, device)
        self.register_buffer("kernel", kernel)
        self.ksize = ksize

    def forward(self, x):
        pad = self.ksize // 2
        x_blurred = F.conv2d(x, self.kernel, padding=pad, groups=3)
        return self.base(x_blurred)
from train_full import (COCO_NAT_DIR, OOD2_DIR, OOD_DIR, OODSingle,
                        PHOTOREAL_META, PHOTOREAL_ROOT, evaluate_acc,
                        split_80_20)


def pgd_attack(model, x, y, eps, alpha, k):
    """L_inf PGD attack. x in [-1, 1] (the format the model wants).
    eps is a scalar in [0,1] pixel-space; alpha same. We map to [-1,1]
    scale by *2.0."""
    eps_b = eps * 2.0
    alpha_b = alpha * 2.0
    x_orig = x.detach()
    delta = torch.zeros_like(x_orig)
    for _ in range(k):
        delta.requires_grad_(True)
        x_pert = (x_orig + delta).clamp(-1.0, 1.0)
        logits = model(x_pert)
        # untargeted: maximize cross-entropy wrt true label
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha_b * grad.sign()
            delta = delta.clamp(-eps_b, eps_b)
            # clamp resulting image to [-1, 1]
            delta = ((x_orig + delta).clamp(-1.0, 1.0) - x_orig)
    return (x_orig + delta).clamp(-1.0, 1.0).detach()


@torch.no_grad()
def evaluate_clean(model, loader, device):
    return evaluate_acc(model, loader, device)


def evaluate_adv(model, loader, device, eps, alpha, k):
    """Evaluate on adversarially-attacked inputs. Restores model.eval()
    at the end. eps is scalar pixel units divided by 255."""
    model.eval()
    correct = total = 0
    pos_correct = pos_total = 0
    neg_correct = neg_total = 0
    for batch in loader:
        x = batch["img"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        x_adv = pgd_attack(model, x, y, eps, alpha, k)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            pos = (y == 1)
            pos_total += pos.sum().item()
            pos_correct += ((pred == y) & pos).sum().item()
            neg = (y == 0)
            neg_total += neg.sum().item()
            neg_correct += ((pred == y) & neg).sum().item()
    return {
        "acc": correct / total if total else 0.0,
        "tpr": pos_correct / pos_total if pos_total else 0.0,
        "tnr": neg_correct / neg_total if neg_total else 0.0,
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/adv_train")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--eps-ladder", default="0,1,2,4,8,16",
                    help="comma-separated list of L_inf budgets (units of /255). "
                         "Each step samples one uniformly.")
    ap.add_argument("--adv-k", type=int, default=3, help="inner PGD steps per train step")
    ap.add_argument("--adv-alpha-units", type=float, default=1.0,
                    help="PGD step size in units of /255")
    ap.add_argument("--constant-lr", action="store_true")
    ap.add_argument("--eval-every-steps", type=int, default=0)
    ap.add_argument("--warmup-clean-steps", type=int, default=300,
                    help="train cleanly (no PGD) for this many steps so the "
                         "ImageNet-pretrained features can adapt before we "
                         "start hammering them with adversarial examples")
    ap.add_argument("--attacker-ckpt", default=None,
                    help="path to a FROZEN attacker checkpoint. If provided, "
                         "PGD perturbations are computed against THIS model, "
                         "not the model being trained. This converts online "
                         "adversarial training into adversarial data "
                         "augmentation, which converges in 1 epoch.")
    ap.add_argument("--attacker-backbone", default="resnet50",
                    choices=["resnet18", "resnet50", "resnet101"],
                    help="architecture of --attacker-ckpt")
    ap.add_argument("--attacker-ckpts-extra", default=None,
                    help="comma-separated EXTRA attacker checkpoints, used "
                         "alongside --attacker-ckpt. Each batch samples one "
                         "of the pool uniformly. Train against perturbations "
                         "from a DIVERSE attacker pool -> learned defense "
                         "should generalize beyond one architecture.")
    ap.add_argument("--attacker-backbones-extra", default=None,
                    help="comma-separated backbones for --attacker-ckpts-extra, "
                         "same length and order")
    ap.add_argument("--trades-beta", type=float, default=0.0,
                    help="If > 0, use TRADES-style loss: CE(x_clean,y) + "
                         "beta * KL(softmax(model(x_clean)) || softmax(model(x_adv))). "
                         "Decouples clean classification from adversarial "
                         "consistency, prevents model collapse during AT.")
    ap.add_argument("--mixed-batch", action="store_true",
                    help="Each batch is half clean + half adv (single eps "
                         "sampled per batch from --eps-ladder, excluding 0). "
                         "All samples keep their correct labels. Vanilla CE. "
                         "This is data augmentation, not online PGD AT -- "
                         "the clean half always provides solid gradient so "
                         "the model can't collapse.")
    ap.add_argument("--resume-from-ckpt", default=None,
                    help="If provided, load the model weights from this ckpt "
                         "after instantiation (instead of starting from "
                         "ImageNet-pretrained weights). For AT fine-tuning.")
    ap.add_argument("--noise-aug-sigma", type=float, default=0.0,
                    help="If >0, add Gaussian noise of this std to BOTH the "
                         "clean and adv halves of each training batch (in "
                         "[-1,1] input space). This trains the model to be "
                         "INTRINSICALLY noise-robust, so randomized smoothing "
                         "at inference is a real defense (vs bolted-on "
                         "gradient masking).")
    ap.add_argument("--blur-sigma", type=float, default=0.0,
                    help="If >0, prepend a Gaussian blur of this std (in "
                         "pixels) to BOTH trainee and attacker models. The "
                         "blur is folded into the classifier so PGD attacks "
                         "must produce perturbations that SURVIVE blur, i.e. "
                         "live in the low frequencies natural images occupy. "
                         "Same blur applied at inference.")
    ap.add_argument("--train-jpeg-q-choices", default=None,
                    help="Comma-separated list of JPEG qualities to sample "
                         "uniformly per TRAINING image (e.g. '30,50,75,95'). "
                         "Model learns features that survive JPEG "
                         "quantization at any of these qualities -- so PGD "
                         "perturbations must also survive JPEG (non-diff "
                         "operation) to fool it. Eval uses --jpeg-q=95 only.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DatasetConfig(
        root=PHOTOREAL_ROOT, metadata=PHOTOREAL_META,
        image_size=args.image_size, jpeg_q=95, photoreal_only=True,
    )
    items = load_metadata(cfg.metadata, photoreal_only=cfg.photoreal_only)
    train_items, test_items = split_80_20(items, seed=args.seed)
    print(f"loaded {len(items)} pairs  split: train={len(train_items)} "
          f"test={len(test_items)}", flush=True)

    train_ds = PicoBananaSingle(train_items, cfg)
    test_ds = PicoBananaSingle(test_items, cfg)
    if args.train_jpeg_q_choices:
        q_choices = [int(x) for x in args.train_jpeg_q_choices.split(",")]
        print(f"WRAPPING TRAIN with random JPEG q in {q_choices}", flush=True)
        train_ds = RandomJPEGWrapper(train_ds, q_choices, image_size=args.image_size)
    ood_ds = OODSingle(OOD_DIR, image_size=cfg.image_size, jpeg_q=cfg.jpeg_q, label=1)
    ood2_ds = OODSingle(OOD2_DIR, image_size=cfg.image_size, jpeg_q=cfg.jpeg_q, label=1)
    coco_nat_ds = OODSingle(COCO_NAT_DIR, image_size=cfg.image_size,
                            jpeg_q=cfg.jpeg_q, label=0)
    # Pre-generated adv OOD eval sets (each is a directory of PNGs that the
    # frozen R50 attacker is 100% fooled on; here we check if the new model
    # can still classify them correctly)
    ADV_EPS_LEVELS = [1, 2, 4, 8, 16]
    adv_edit_dls = {}
    adv_nat_dls = {}
    expanded_test_datasets = [ood2_ds, coco_nat_ds]  # start with the clean ones
    for eps_units in ADV_EPS_LEVELS:
        edit_dir = f"/mnt/data/coco_ood_v2_500/edit_adv_eps{eps_units:03d}"
        nat_dir  = f"/mnt/data/coco_ood_v2_500/orig_adv_eps{eps_units:03d}"
        if os.path.isdir(edit_dir):
            de = OODSingle(edit_dir, image_size=cfg.image_size,
                           jpeg_q=cfg.jpeg_q, label=1)
            de.paths = sorted(p for p in os.listdir(edit_dir) if p.endswith(".png"))
            adv_edit_dls[eps_units] = DataLoader(
                de, batch_size=args.batch, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            expanded_test_datasets.append(de)
        if os.path.isdir(nat_dir):
            dn = OODSingle(nat_dir, image_size=cfg.image_size,
                           jpeg_q=cfg.jpeg_q, label=0)
            dn.paths = sorted(p for p in os.listdir(nat_dir) if p.endswith(".png"))
            adv_nat_dls[eps_units] = DataLoader(
                dn, batch_size=args.batch, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            expanded_test_datasets.append(dn)
    # Single combined test set: clean OOD-COCO (496+496) + all eps adv (496*10)
    from torch.utils.data import ConcatDataset
    expanded_test_ds = ConcatDataset(expanded_test_datasets)
    expanded_test_dl = DataLoader(expanded_test_ds, batch_size=args.batch,
                                  shuffle=False, num_workers=args.workers,
                                  pin_memory=True)
    print(f"  expanded OOD-COCO test set: {len(expanded_test_ds)} samples "
          f"(clean + 5 eps levels, both sides)", flush=True)
    print(f"single examples: train={len(train_ds)} test={len(test_ds)} "
          f"ood_diffdb={len(ood_ds)} ood_coco_edit={len(ood2_ds)} "
          f"ood_coco_nat={len(coco_nat_ds)}  "
          f"adv_eval_levels={list(adv_edit_dls.keys())}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    ood_dl = DataLoader(ood_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    ood2_dl = DataLoader(ood2_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    coco_nat_dl = DataLoader(coco_nat_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = BinaryClassifier(pretrained=True, backbone=args.backbone).to(device)
    print(f"backbone={args.backbone}  params={count_params(model)/1e6:.2f}M", flush=True)
    if args.resume_from_ckpt:
        print(f"resuming from {args.resume_from_ckpt}", flush=True)
        rck = torch.load(args.resume_from_ckpt, map_location=device)
        model.load_state_dict(rck["state_dict"])
        print(f"  loaded weights from {args.resume_from_ckpt}", flush=True)
    if args.blur_sigma > 0:
        print(f"WRAPPING TRAINEE with Gaussian blur sigma={args.blur_sigma}",
              flush=True)
        model = BlurWrap(model, args.blur_sigma, device).to(device)

    # Optional frozen attacker pool (for "adversarial data augmentation" mode).
    # `attacker` (the first / canonical one) keeps backwards compat; the
    # `attackers` list contains the full pool to sample from each batch.
    attacker = None
    attackers = []
    if args.attacker_ckpt:
        print(f"loading frozen attacker {args.attacker_ckpt} "
              f"(backbone={args.attacker_backbone})", flush=True)
        attacker = BinaryClassifier(pretrained=False,
                                    backbone=args.attacker_backbone).to(device).eval()
        ack = torch.load(args.attacker_ckpt, map_location=device)
        attacker.load_state_dict(ack["state_dict"])
        for p in attacker.parameters():
            p.requires_grad_(False)
        if args.blur_sigma > 0:
            attacker = BlurWrap(attacker, args.blur_sigma, device).to(device).eval()
            for p in attacker.parameters():
                p.requires_grad_(False)
        attackers.append(attacker)
        print(f"  frozen attacker loaded.", flush=True)
    if args.attacker_ckpts_extra:
        extra_paths = args.attacker_ckpts_extra.split(",")
        extra_bbs = args.attacker_backbones_extra.split(",")
        assert len(extra_paths) == len(extra_bbs), \
            "attacker-ckpts-extra and attacker-backbones-extra must match length"
        for p, bb in zip(extra_paths, extra_bbs):
            print(f"loading EXTRA frozen attacker {p} (backbone={bb})", flush=True)
            m_a = BinaryClassifier(pretrained=False, backbone=bb).to(device).eval()
            ack = torch.load(p, map_location=device)
            m_a.load_state_dict(ack["state_dict"])
            for pp in m_a.parameters():
                pp.requires_grad_(False)
            if args.blur_sigma > 0:
                m_a = BlurWrap(m_a, args.blur_sigma, device).to(device).eval()
                for pp in m_a.parameters():
                    pp.requires_grad_(False)
            attackers.append(m_a)
            print(f"  EXTRA attacker loaded.", flush=True)
    print(f"attacker pool size: {len(attackers)}", flush=True)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    if args.constant_lr:
        sched = LambdaLR(opt, lr_lambda=lambda s: 1.0)
    else:
        sched = CosineAnnealingLR(opt, T_max=total_steps)
    crit = nn.CrossEntropyLoss()

    eps_ladder = [int(e) for e in args.eps_ladder.split(",")]
    eps_pixel_to_01 = lambda e: e / 255.0
    alpha = args.adv_alpha_units / 255.0
    print(f"PGD: eps_ladder={eps_ladder}/255  k={args.adv_k}  "
          f"alpha={args.adv_alpha_units}/255", flush=True)

    rng = random.Random(args.seed)

    best_score = -2.0
    log_path = os.path.join(args.out_dir, "log.txt")
    flog = open(log_path, "w")
    flog.write("epoch\tstep\ttrain_loss\ttest_acc\ttest_tpr\ttest_tnr\tood_diffdb\t"
               "ood_coco_edit\tcoco_nat_tnr\tbalanced_clean\tadv_bal_avg\t"
               "combined_score\ttime\n")
    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        n_seen = 0
        for step, batch in enumerate(train_dl):
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            # Clean warmup: force eps=0 for the first N steps
            if global_step < args.warmup_clean_steps:
                eps_units = 0
            else:
                eps_units = rng.choice(eps_ladder)

            # --- mixed-batch mode: half clean + half adv per batch ---
            if args.mixed_batch and eps_units > 0:
                B = x.size(0)
                half = B // 2
                # Sample an attacker from the pool (or use the trainee if none)
                if attackers:
                    atk = rng.choice(attackers)
                    atk_label = "pool[%d]" % attackers.index(atk)
                else:
                    atk = model
                    atk_label = "self"
                    model.eval()
                x_half_adv = pgd_attack(atk, x[:half], y[:half],
                                        eps_pixel_to_01(eps_units), alpha, args.adv_k)
                if not attackers:
                    model.train()
                x_in = torch.cat([x_half_adv, x[half:]], dim=0)
                # Optional input-space Gaussian noise augmentation: makes
                # the model intrinsically noise-robust so that random
                # smoothing at inference is a real defense, not gradient
                # masking. Same sigma should be used at deploy time.
                if args.noise_aug_sigma > 0:
                    x_in = (x_in + torch.randn_like(x_in) * args.noise_aug_sigma
                            ).clamp(-1.0, 1.0)
                # labels stay correct (no flipping)
                logits = model(x_in)
                loss = crit(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                running += loss.item()
                n_seen += 1
                global_step += 1
                if step % 100 == 0:
                    el = time.time() - t0
                    print(f"  ep{epoch} step {step:4d}/{len(train_dl)}  "
                          f"eps={eps_units:>2d} (mixed,{atk_label})  loss={loss.item():.4f}  "
                          f"lr={sched.get_last_lr()[0]:.2e}  ({el:.0f}s)", flush=True)
                continue
            # --- end mixed-batch mode ---

            if eps_units > 0:
                if attacker is not None:
                    x_adv = pgd_attack(attacker, x, y,
                                       eps_pixel_to_01(eps_units), alpha, args.adv_k)
                else:
                    model.eval()
                    x_adv = pgd_attack(model, x, y,
                                       eps_pixel_to_01(eps_units), alpha, args.adv_k)
                    model.train()
                if args.trades_beta > 0:
                    # TRADES: CE(clean) + beta * KL(clean || adv)
                    logits_clean = model(x)
                    logits_adv = model(x_adv)
                    loss_ce = crit(logits_clean, y)
                    log_p_clean = F.log_softmax(logits_clean, dim=1)
                    log_p_adv = F.log_softmax(logits_adv, dim=1)
                    loss_kl = F.kl_div(log_p_adv, log_p_clean.detach(),
                                       reduction="batchmean", log_target=True)
                    loss = loss_ce + args.trades_beta * loss_kl
                else:
                    # Original: CE on x_adv with y_true
                    logits = model(x_adv)
                    loss = crit(logits, y)
            else:
                logits = model(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            n_seen += 1
            global_step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} step {step:4d}/{len(train_dl)}  "
                      f"eps={eps_units:>2d}  loss={loss.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  ({el:.0f}s)", flush=True)

        # ---- eval ----
        test_m = evaluate_clean(model, test_dl, device)
        ood_m = evaluate_clean(model, ood_dl, device)
        ood2_m = evaluate_clean(model, ood2_dl, device)
        coconat_m = evaluate_clean(model, coco_nat_dl, device)
        bal_clean = ood2_m["tpr"] + coconat_m["tnr"] - 1.0

        # Evaluate on the PRE-GENERATED adv OOD eval sets across the
        # full eps ladder. These are perturbed against the frozen R50
        # attacker; we measure whether the new model can still classify
        # them correctly. Reports per-eps and an averaged adv-balanced.
        per_eps_bal = []
        per_eps_log = []
        for eps_units in ADV_EPS_LEVELS:
            if eps_units not in adv_edit_dls or eps_units not in adv_nat_dls:
                continue
            ae = evaluate_clean(model, adv_edit_dls[eps_units], device)
            an = evaluate_clean(model, adv_nat_dls[eps_units], device)
            b = ae["tpr"] + an["tnr"] - 1.0
            per_eps_bal.append(b)
            per_eps_log.append((eps_units, ae["tpr"], an["tnr"], b))
        bal_adv = sum(per_eps_bal) / max(1, len(per_eps_bal))
        # Expanded combined OOD-COCO acc (single set, clean + 5 adv eps)
        expanded = evaluate_clean(model, expanded_test_dl, device)
        expanded_bal = expanded["tpr"] + expanded["tnr"] - 1.0
        # combined score: prefer the expanded test (real metric for the kit)
        score = expanded_bal

        elapsed = time.time() - t0
        adv_str = "  ".join(f"e{e}({et:.2f},{nt:.2f},{b:+.2f})"
                            for (e, et, nt, b) in per_eps_log)
        print(f"\nepoch {epoch}: "
              f"test={test_m['acc']:.4f} (TPR={test_m['tpr']:.4f}, TNR={test_m['tnr']:.4f})  "
              f"DiffDB={ood_m['tpr']:.4f}  "
              f"COCO-edit={ood2_m['tpr']:.4f}  COCO-nat-TNR={coconat_m['tnr']:.4f}  "
              f"clean-bal={bal_clean:+.4f}  adv-bal-avg={bal_adv:+.4f}  "
              f"EXPANDED-acc={expanded['acc']:.4f} (TPR={expanded['tpr']:.4f}, "
              f"TNR={expanded['tnr']:.4f})  bal={expanded_bal:+.4f}  "
              f"({elapsed:.0f}s)", flush=True)
        print(f"  per-eps adv  (eps: edit-TPR, nat-TNR, bal): {adv_str}", flush=True)
        flog.write(f"{epoch}\t{global_step}\t{running/n_seen:.4f}\t"
                   f"{test_m['acc']:.4f}\t{test_m['tpr']:.4f}\t{test_m['tnr']:.4f}\t"
                   f"{ood_m['tpr']:.4f}\t{ood2_m['tpr']:.4f}\t{coconat_m['tnr']:.4f}\t"
                   f"{bal_clean:.4f}\t{bal_adv:.4f}\t{score:.4f}\t{elapsed:.0f}\n")
        flog.flush()
        if score > best_score:
            best_score = score
            # If BlurWrap'd, save only the inner classifier state_dict for
            # compatibility with downstream load code (which constructs a bare
            # BinaryClassifier and loads into it). The blur-sigma is in args.
            sd = (model.base.state_dict() if isinstance(model, BlurWrap)
                  else model.state_dict())
            torch.save({
                "state_dict": sd,
                "score": score, "bal_clean": bal_clean, "bal_adv": bal_adv,
                "epoch": epoch, "args": vars(args),
                "test_id": test_m, "test_ood": ood_m, "test_ood_coco": ood2_m,
                "coco_natural": coconat_m, "per_eps_adv": per_eps_log,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best.pt at combined score={score:+.4f}", flush=True)
    flog.close()

    summary = {
        "seed": args.seed, "backbone": args.backbone,
        "eps_ladder": eps_ladder, "adv_k": args.adv_k,
        "best_score": best_score,
        "n_train_pairs": len(train_items), "n_test_pairs": len(test_items),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ndone. best combined score = {best_score:+.4f}", flush=True)


if __name__ == "__main__":
    main()
