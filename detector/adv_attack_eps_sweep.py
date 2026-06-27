"""ε-sweep PGD attack against best_ood_balanced_r50 on the v2/edit/00004
traffic-light image. For each ε in a geometric ladder, run PGD until
P(natural) >= 0.99 (or max iters), save the attacked native PNG, and
record perceptual-quality stats (PSNR + LPIPS vs the original) so the
user can correlate "C_eval-fool effort" against "visual degradation"
and (externally) "SynthID survival"."""

from __future__ import annotations

import csv
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

import lpips
from skimage.metrics import peak_signal_noise_ratio

SRC = "/home/trabbani/WAVES/coco_ood_examples/v2/edit/00004.jpg"
CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack/eps_sweep")

EPS_LADDER = [1, 2, 4, 8, 16, 32, 64, 96]   # units of /255
ALPHA = 1.0 / 255.0
MAX_STEPS = 500
IMAGE_SIZE = 384
TARGET_LABEL = 0
# Run a fixed number of full PGD iterations so the perturbation actually
# fills the eps budget. (Early-stopping the moment P(nat) >= 0.99 keeps
# every eps trivially small since the first step already saturates the
# classifier -- we want the worst-case attacked image at each budget,
# not the minimal-attack image.) Each run fills its eps budget after
# ~eps/alpha = eps*255 steps; we add ~50 step cushion.
NUM_ATTACK_STEPS = lambda eps_units: int(eps_units) + 50

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

    net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in net_lpips.parameters():
        p.requires_grad_(False)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    src_pil.save(OUT_DIR / "orig_native.png", "PNG", optimize=True)
    x_orig = pil_to_chw01(src_pil).to(device)
    print(f"native tensor shape: {tuple(x_orig.shape)}", flush=True)

    # baseline via the deployed matched-codec pipeline
    base_pil = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_t = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        p = F.softmax(model(base_t), dim=1)[0]
        baseline = {"p_natural": float(p[0]), "p_synth": float(p[1])}
    print(f"baseline (matched-codec): "
          f"P(natural)={baseline['p_natural']:.4f}  P(synth)={baseline['p_synth']:.4f}",
          flush=True)

    # LPIPS expects [-1, 1] tensors at any res; we'll compare at native res
    x_orig_neg1_1 = (x_orig * 2.0 - 1.0)

    results = []
    target = torch.tensor([TARGET_LABEL], device=device, dtype=torch.long)

    for eps_units in EPS_LADDER:
        eps = eps_units / 255.0
        print(f"\n{'='*60}\nepsilon = {eps_units}/255 ({eps:.4f})\n{'='*60}", flush=True)
        delta = torch.zeros_like(x_orig, requires_grad=True)
        n_steps = NUM_ATTACK_STEPS(eps_units)
        final_step = n_steps - 1
        for step in range(n_steps):
            delta.requires_grad_(True)
            x_adv = (x_orig + delta).clamp(0.0, 1.0)
            x_resized = F.interpolate(x_adv, size=(IMAGE_SIZE, IMAGE_SIZE),
                                      mode="bilinear", align_corners=False,
                                      antialias=True)
            x_n11 = x_resized * 2.0 - 1.0
            logits = model(x_n11)
            # Logit-margin loss: minimize (logit_synth - logit_natural).
            # This is unbounded, so the gradient never vanishes -- we keep
            # filling the eps budget even after the classifier saturates.
            loss = logits[0, 1] - logits[0, 0]
            loss.backward()
            with torch.no_grad():
                delta = delta - ALPHA * delta.grad.sign()
                delta = delta.clamp(-eps, eps)
                delta = ((x_orig + delta).clamp(0.0, 1.0) - x_orig)
                p_nat = float(F.softmax(logits, dim=1)[0, 0].item())
            if step % max(10, n_steps // 5) == 0 or step == n_steps - 1:
                print(f"  step {step:3d}/{n_steps}  loss={loss.item():.4f}  "
                      f"P(nat)={p_nat:.4f}", flush=True)

        with torch.no_grad():
            x_adv_final = (x_orig + delta).clamp(0.0, 1.0)
            # save native PNG
            arr = (x_adv_final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            adv_pil = Image.fromarray(arr)
            png_path = OUT_DIR / f"adv_eps{eps_units:03d}.png"
            adv_pil.save(png_path, "PNG", optimize=True)

            # perceptual quality vs original (native res)
            psnr = float(peak_signal_noise_ratio(
                np.asarray(src_pil, dtype=np.uint8), arr, data_range=255))
            x_adv_n11 = (x_adv_final * 2.0 - 1.0)
            lpips_val = float(net_lpips(x_orig_neg1_1, x_adv_n11).mean().item())

            d_abs = (delta[0].abs().cpu().numpy() * 255)
            d_mean = float(d_abs.mean())
            d_max = float(d_abs.max())

            # final eval through deployed matched-codec
            jpg_path = OUT_DIR / f"adv_eps{eps_units:03d}.jpg"
            adv_pil.save(jpg_path, "JPEG", quality=95, optimize=True)
            final_pil = _load_and_match_codec(str(jpg_path), IMAGE_SIZE, jpeg_q=95)
            final_t = to_tensor_neg1_1(final_pil).unsqueeze(0).to(device)
            probs = F.softmax(model(final_t), dim=1)[0]
            p_nat_final = float(probs[0])
            p_syn_final = float(probs[1])

        row = {
            "eps_units": eps_units,
            "eps": eps,
            "steps_to_threshold": final_step + 1,
            "delta_mean_units": d_mean,
            "delta_max_units": d_max,
            "p_natural_deployed": p_nat_final,
            "p_synth_deployed": p_syn_final,
            "psnr_db": psnr,
            "lpips_vgg": lpips_val,
            "png_path": str(png_path),
        }
        results.append(row)
        print(f"  steps={final_step+1}  mean|d|={d_mean:.2f}  max|d|={d_max:.1f}  "
              f"PSNR={psnr:.1f}dB  LPIPS={lpips_val:.4f}  "
              f"deployed P(nat)={p_nat_final:.4f}", flush=True)

    # Write CSV + JSON
    with open(OUT_DIR / "eps_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(OUT_DIR / "eps_sweep.json", "w") as f:
        json.dump({"src": SRC, "ckpt": CKPT, "baseline": baseline,
                   "results": results}, f, indent=2)

    print("\n========== EPS SWEEP SUMMARY ==========")
    print(f"{'eps':>5s}  {'steps':>5s}  {'mean|d|':>7s}  {'max|d|':>6s}  "
          f"{'PSNR':>6s}  {'LPIPS':>6s}  {'P(nat)deploy':>12s}")
    for r in results:
        print(f"{r['eps_units']:>5d}  {r['steps_to_threshold']:>5d}  "
              f"{r['delta_mean_units']:>7.2f}  {r['delta_max_units']:>6.1f}  "
              f"{r['psnr_db']:>6.2f}  {r['lpips_vgg']:>6.4f}  "
              f"{r['p_natural_deployed']:>12.4f}")


if __name__ == "__main__":
    main()
