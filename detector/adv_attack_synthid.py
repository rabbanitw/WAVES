"""Adversarial attack against best_ood_balanced_r50.pt on the
coco_ood_examples/v2 carrot/steak NB edit (00000.jpg).

Untargeted-style PGD with target = class 0 (natural). Operates on the
native-resolution image; the model's forward includes Lanczos resize
to 384 -> matched-codec JPEG round-trip is skipped during attack (the
re-encode is non-differentiable) but applied at final verification to
match the deployed eval pipeline exactly.

Saves:
- adv_attack/orig_native.png      : copy of the source image
- adv_attack/adv_native.png       : attacked image at native res (PNG, lossless)
- adv_attack/delta_native.png     : |adv - orig| amplified for inspection
- adv_attack/trajectory.csv       : P(natural) at each PGD step
- adv_attack/results.json         : pre/post P(natural), detector verdicts
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import _load_and_match_codec, to_tensor_neg1_1
from models import BinaryClassifier

# ---- config ----
# 00004 = traffic-light street scene "overcast -> clear blue sky" edit.
# At baseline the detector calls it synthetic with P(synth) = 1.0000, so
# this is the strongest possible test of whether an attack on C_eval can
# also degrade the underlying SynthID watermark. (00000.jpg was already a
# false negative -- P(natural)=0.73 at baseline -- so attacking it told us
# nothing.)
SRC = "/home/trabbani/WAVES/coco_ood_examples/v2/edit/00004.jpg"
CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack")

EPS = 32.0 / 255.0       # L_inf budget in [0,1] pixel space
ALPHA = 1.0 / 255.0      # PGD step size
N_STEPS = 300
IMAGE_SIZE = 384         # detector's native input size
TARGET_LABEL = 0         # we want the model to call this "natural"
P_THRESHOLD = 0.99       # stop early when P(natural) >= this

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    device = "cuda"
    print(f"loading {CKPT} ...", flush=True)
    model = BinaryClassifier(pretrained=False, backbone="resnet50").to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded best.pt with balanced score = {ckpt['balanced']:.4f}", flush=True)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    src_pil.save(OUT_DIR / "orig_native.png", "PNG", optimize=True)

    # native [0, 1] tensor, will be perturbed in place
    x_orig = pil_to_chw01(src_pil).to(device)  # (1, 3, H, W) in [0, 1]
    print(f"native tensor shape: {tuple(x_orig.shape)}", flush=True)

    # ---- baseline P(natural) on the unattacked image (matched-codec eval) ----
    base_pil_384 = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_neg1_1 = to_tensor_neg1_1(base_pil_384).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(base_neg1_1)
        baseline_p_natural = float(F.softmax(logits, dim=1)[0, 0].item())
        baseline_p_synth = float(F.softmax(logits, dim=1)[0, 1].item())
    print(f"\nBASELINE (unattacked, matched-codec):", flush=True)
    print(f"  P(natural) = {baseline_p_natural:.4f}", flush=True)
    print(f"  P(synth)   = {baseline_p_synth:.4f}", flush=True)

    # ---- PGD attack in [0, 1] native pixel space ----
    # The model expects [-1, 1] input AFTER Lanczos resize to 384.
    # Forward: x01 -> resize(384) -> *2-1 -> model.
    delta = torch.zeros_like(x_orig, requires_grad=True)
    target = torch.tensor([TARGET_LABEL], device=device, dtype=torch.long)

    traj = []
    print(f"\nPGD attack: eps={EPS:.4f}  alpha={ALPHA:.4f}  steps={N_STEPS}", flush=True)
    for step in range(N_STEPS):
        delta.requires_grad_(True)
        x_adv = (x_orig + delta).clamp(0.0, 1.0)
        # Differentiable resize to detector input size
        x_resized = F.interpolate(x_adv, size=(IMAGE_SIZE, IMAGE_SIZE),
                                  mode="bilinear", align_corners=False, antialias=True)
        x_neg1_1 = x_resized * 2.0 - 1.0
        logits = model(x_neg1_1)
        # CE with label 0 -> minimize to push toward "natural"
        loss = F.cross_entropy(logits, target)
        loss.backward()

        with torch.no_grad():
            # PGD step: descend on loss (= ascend on P(natural))
            delta = delta - ALPHA * delta.grad.sign()
            # Project to L_inf ball
            delta = delta.clamp(-EPS, EPS)
            # Keep adversarial image in [0, 1]
            delta = ((x_orig + delta).clamp(0.0, 1.0) - x_orig)

            probs = F.softmax(logits, dim=1)[0]
            p_nat = float(probs[0].item())
            p_syn = float(probs[1].item())
            traj.append({"step": step, "p_natural": p_nat, "p_synth": p_syn,
                         "loss": float(loss.item())})
            if step % 20 == 0 or step == N_STEPS - 1:
                print(f"  step {step:3d}  loss={loss.item():.4f}  "
                      f"P(natural)={p_nat:.4f}  P(synth)={p_syn:.4f}", flush=True)
            if p_nat >= P_THRESHOLD:
                print(f"  -> P(natural) = {p_nat:.4f} >= {P_THRESHOLD}, stopping early at step {step}",
                      flush=True)
                break
        delta.requires_grad_(True)

    # ---- finalize adversarial image ----
    with torch.no_grad():
        x_adv_final = (x_orig + delta).clamp(0.0, 1.0)
        x_adv_np = (x_adv_final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        adv_pil = Image.fromarray(x_adv_np)
    adv_path = OUT_DIR / "adv_native.png"
    adv_pil.save(adv_path, "PNG", optimize=True)
    print(f"\nwrote {adv_path}  ({os.path.getsize(adv_path)/1024:.0f} KB)", flush=True)

    # Delta visualization (amplified)
    with torch.no_grad():
        d_abs = (delta[0].abs().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
        # Stretch to full range for inspection
        d_max = d_abs.max()
        if d_max > 0:
            d_stretch = (d_abs / d_max * 255).astype(np.uint8)
        else:
            d_stretch = d_abs.astype(np.uint8)
        Image.fromarray(d_stretch).save(OUT_DIR / "delta_amplified.png", "PNG")
        # Per-pixel max-abs deviation in native [0, 255]
        print(f"  delta range: [{d_abs.min():.1f}, {d_abs.max():.1f}] (units: 0-255 scale)",
              flush=True)
        print(f"  mean|delta|: {d_abs.mean():.2f}", flush=True)

    # ---- final verification through the full deployed eval pipeline ----
    # 1. save adv as JPEG q=95 (the format the deployed eval uses)
    adv_jpg = OUT_DIR / "adv_native.jpg"
    adv_pil.save(adv_jpg, "JPEG", quality=95, optimize=True)
    # 2. matched-codec preprocessing (Lanczos to 384 + JPEG q=95 + decode)
    final_pil_384 = _load_and_match_codec(str(adv_jpg), IMAGE_SIZE, jpeg_q=95)
    final_neg1_1 = to_tensor_neg1_1(final_pil_384).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(final_neg1_1)
        final_p_natural = float(F.softmax(logits, dim=1)[0, 0].item())
        final_p_synth = float(F.softmax(logits, dim=1)[0, 1].item())

    print(f"\nFINAL (attacked, matched-codec deployed eval):", flush=True)
    print(f"  P(natural) = {final_p_natural:.4f}  "
          f"(baseline was {baseline_p_natural:.4f})", flush=True)
    print(f"  P(synth)   = {final_p_synth:.4f}  "
          f"(baseline was {baseline_p_synth:.4f})", flush=True)
    verdict_was = "synthetic" if baseline_p_synth >= 0.5 else "natural"
    verdict_now = "synthetic" if final_p_synth >= 0.5 else "natural"
    print(f"  detector verdict: {verdict_was} -> {verdict_now}", flush=True)

    # Write trajectory CSV
    with open(OUT_DIR / "trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "p_natural", "p_synth", "ce_loss_target0"])
        for r in traj:
            w.writerow([r["step"], f"{r['p_natural']:.6f}",
                        f"{r['p_synth']:.6f}", f"{r['loss']:.6f}"])

    # Results JSON
    results = {
        "src": SRC,
        "ckpt": CKPT,
        "ckpt_balanced": ckpt["balanced"],
        "eps_linf": EPS,
        "alpha": ALPHA,
        "n_steps_run": len(traj),
        "image_size_for_detector": IMAGE_SIZE,
        "target_label": TARGET_LABEL,
        "native_size": list(src_pil.size),
        "baseline": {"p_natural": baseline_p_natural, "p_synth": baseline_p_synth,
                     "verdict": verdict_was},
        "final": {"p_natural": final_p_natural, "p_synth": final_p_synth,
                  "verdict": verdict_now},
        "delta_native_units": {
            "min_abs": float(d_abs.min()),
            "max_abs": float(d_abs.max()),
            "mean_abs": float(d_abs.mean()),
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
