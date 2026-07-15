"""Re-run the imperceptible PGD attack against the adv-trained
s48 R18 (`runs/adv_mixed_r18/s48/best.pt`). Same image and PGD setup
as adv_attack_synthid.py so we can directly compare robustness:
  - baseline best_ood_balanced_r50: P(synth)=1.0 -> P(natural)=1.0 in 2 steps,
    max|delta|=2/255.
  - s48 adv-trained: how many steps + how much eps does it take here?

No L_inf cap; logit-margin loss + step size 1/255; stop when P(natural) >= 0.999."""

from __future__ import annotations

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

SRC = "/home/trabbani/WAVES/coco_ood_examples/v2/edit/00004.jpg"
CKPT = "/home/trabbani/WAVES/detector/runs/adv_mixed_r18/s48/best.pt"
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack_vs_s48")

ALPHA = 1.0 / 255.0
MAX_STEPS = 1000
EPS_CAP = 96.0 / 255.0   # generous cap, we expect the attack may need it
IMAGE_SIZE = 384
P_THRESHOLD = 0.999

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    device = "cuda"
    print(f"loading {CKPT} ...", flush=True)
    model = BinaryClassifier(pretrained=False, backbone="resnet18").to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded s48 R18 (expanded-bal score = {ckpt['score']:+.4f})", flush=True)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    src_pil.save(OUT_DIR / "orig_native.png", "PNG", optimize=True)
    x_orig = pil_to_chw01(src_pil).to(device)

    # baseline (matched-codec)
    base_pil = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_t = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        p = F.softmax(model(base_t), dim=1)[0]
        baseline = {"p_natural": float(p[0]), "p_synth": float(p[1])}
    print(f"baseline (matched-codec): "
          f"P(natural)={baseline['p_natural']:.4f}  P(synth)={baseline['p_synth']:.4f}",
          flush=True)
    x_orig_neg1_1 = (x_orig * 2.0 - 1.0)

    # PGD with logit-margin loss
    delta = torch.zeros_like(x_orig, requires_grad=True)
    traj = []
    steps_to_threshold = None
    delta_at_threshold = None
    print(f"\nPGD: alpha={ALPHA:.5f}  eps_cap={EPS_CAP:.4f}  threshold P(nat)>={P_THRESHOLD}",
          flush=True)
    for step in range(MAX_STEPS):
        delta.requires_grad_(True)
        x_adv = (x_orig + delta).clamp(0.0, 1.0)
        x_resized = F.interpolate(x_adv, size=(IMAGE_SIZE, IMAGE_SIZE),
                                  mode="bilinear", align_corners=False, antialias=True)
        x_n11 = x_resized * 2.0 - 1.0
        logits = model(x_n11)
        loss = logits[0, 1] - logits[0, 0]
        loss.backward()
        with torch.no_grad():
            delta = delta - ALPHA * delta.grad.sign()
            delta = delta.clamp(-EPS_CAP, EPS_CAP)
            delta = ((x_orig + delta).clamp(0.0, 1.0) - x_orig)
            p_nat = float(F.softmax(logits, dim=1)[0, 0].item())
            d_max = float(delta.abs().max().item() * 255)
            d_mean = float(delta.abs().mean().item() * 255)
        traj.append({"step": step, "p_natural": p_nat, "loss": float(loss.item()),
                     "delta_max_units": d_max, "delta_mean_units": d_mean})
        if step % 25 == 0 or step == MAX_STEPS - 1:
            print(f"  step {step:4d}  loss={loss.item():+.4f}  P(nat)={p_nat:.4f}  "
                  f"mean|d|={d_mean:.2f}  max|d|={d_max:.1f}", flush=True)
        if p_nat >= P_THRESHOLD and steps_to_threshold is None:
            steps_to_threshold = step
            delta_at_threshold = (d_max, d_mean)
            print(f"\n  -> reached P(natural) >= {P_THRESHOLD} at step {step}  "
                  f"(max|d|={d_max:.1f}/255, mean|d|={d_mean:.2f}/255)", flush=True)
            break

    # save adversarial image
    with torch.no_grad():
        x_adv_final = (x_orig + delta).clamp(0.0, 1.0)
        arr = (x_adv_final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    adv_pil = Image.fromarray(arr)
    adv_pil.save(OUT_DIR / "adv_native.png", "PNG", optimize=True)
    adv_pil.save(OUT_DIR / "adv_native.jpg", "JPEG", quality=95, optimize=True)

    # delta visualization
    with torch.no_grad():
        d_abs = (delta[0].abs().permute(1, 2, 0).cpu().numpy() * 255)
        d_max_val = d_abs.max()
        d_stretch = (d_abs / max(d_max_val, 1) * 255).astype(np.uint8) if d_max_val > 0 else d_abs.astype(np.uint8)
        Image.fromarray(d_stretch).save(OUT_DIR / "delta_amplified.png", "PNG")

    # final eval through matched-codec
    final_pil = _load_and_match_codec(str(OUT_DIR / "adv_native.jpg"), IMAGE_SIZE, jpeg_q=95)
    final_t = to_tensor_neg1_1(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(final_t), dim=1)[0]
        final = {"p_natural": float(probs[0]), "p_synth": float(probs[1])}
    print(f"\nFINAL (attacked + matched-codec deployed eval):", flush=True)
    print(f"  P(natural) = {final['p_natural']:.4f}  (baseline {baseline['p_natural']:.4f})",
          flush=True)
    print(f"  P(synth)   = {final['p_synth']:.4f}  (baseline {baseline['p_synth']:.4f})",
          flush=True)
    print(f"  detector verdict: "
          f"{'synthetic' if baseline['p_synth'] >= 0.5 else 'natural'} -> "
          f"{'synthetic' if final['p_synth'] >= 0.5 else 'natural'}", flush=True)

    import csv
    with open(OUT_DIR / "trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "p_natural", "loss", "delta_max_units", "delta_mean_units"])
        for r in traj:
            w.writerow([r["step"], f"{r['p_natural']:.6f}", f"{r['loss']:+.6f}",
                        f"{r['delta_max_units']:.4f}", f"{r['delta_mean_units']:.6f}"])

    results = {
        "src": SRC, "ckpt": CKPT, "ckpt_score": ckpt["score"],
        "baseline": baseline, "final": final,
        "steps_to_threshold": steps_to_threshold,
        "delta_max_at_threshold": delta_at_threshold[0] if delta_at_threshold else None,
        "delta_mean_at_threshold": delta_at_threshold[1] if delta_at_threshold else None,
        "eps_cap_units": EPS_CAP * 255,
        "p_threshold": P_THRESHOLD,
        "alpha_units": ALPHA * 255,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
