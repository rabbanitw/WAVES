"""EOT (Expectation Over Transformations) PGD attack against the
RANDOMIZED-SMOOTHED s48 R18 detector. The smoothed prediction is:

    P_smooth(y|x) = E_noise[softmax(model(x + noise))]

A naive PGD attack just uses model(x) gradients -- this is BPDA's failure
mode and trivially defeats smoothing. EOT instead computes:

    grad_delta L = E_noise[ grad_delta loss(model(x + delta + noise)) ]

by sampling N noise realizations per step and averaging. This is the
"right" attack against randomized smoothing (Athalye et al. 2018,
Obfuscated Gradients).

If the s48 + sigma=0.025 defense holds against this attack at much larger
eps than the 2/255 baseline, the smoothing is providing real robustness
(not gradient masking). If it falls in 2 steps like the baseline, the
"adv robustness" we saw at the pre-generated eval was just transfer-attack
failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import _load_and_match_codec, to_tensor_neg1_1
from models import BinaryClassifier

SRC = "/home/trabbani/WAVES/coco_ood_examples/v2/edit/00001.jpg"
CKPT = "/home/trabbani/WAVES/detector/best_adv_multiattacker_r18.pt"
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack_eot_vs_multiatk_s48")

SIGMA = 0.025          # defense noise level
N_EOT = 20             # noise samples per attack step
N_SMOOTH = 20          # noise samples for smoothed prediction at eval
ALPHA = 1.0 / 255.0
MAX_STEPS = 1000
EPS_CAP = 96.0 / 255.0
IMAGE_SIZE = 384
P_THRESHOLD = 0.999
LOG_EVERY = 10

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def smoothed_p_natural(model, x_neg1_1, sigma, n_samples):
    """Compute E_noise[ P(natural | x + noise) ] via Monte Carlo."""
    probs_sum = torch.zeros(2, device=x_neg1_1.device)
    for _ in range(n_samples):
        noise = torch.randn_like(x_neg1_1) * sigma
        x_noisy = (x_neg1_1 + noise).clamp(-1.0, 1.0)
        probs_sum += F.softmax(model(x_noisy), dim=1)[0]
    return float((probs_sum / n_samples)[0].item())


def main():
    device = "cuda"
    print(f"loading {CKPT} ...", flush=True)
    model = BinaryClassifier(pretrained=False, backbone="resnet18").to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded R18 s48 (expanded-bal score = {ckpt['score']:+.4f})", flush=True)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    x_orig = pil_to_chw01(src_pil).to(device)

    # baseline (matched-codec + smoothed eval)
    base_pil = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_t = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        p_natural_no_smooth = float(F.softmax(model(base_t), dim=1)[0, 0].item())
        p_natural_smooth = smoothed_p_natural(model, base_t, SIGMA, N_SMOOTH)
    print(f"\nbaseline:", flush=True)
    print(f"  P(natural)        (no smoothing)        = {p_natural_no_smooth:.4f}", flush=True)
    print(f"  P(natural)        (smoothed, sigma={SIGMA}) = {p_natural_smooth:.4f}", flush=True)

    print(f"\n--- EOT-PGD attack (N_EOT={N_EOT}, sigma={SIGMA}, alpha=1/255, "
          f"eps_cap={EPS_CAP*255:.0f}/255, threshold={P_THRESHOLD}) ---", flush=True)
    delta = torch.zeros_like(x_orig)
    traj = []
    steps_to_threshold = None
    delta_at_threshold = None

    for step in range(MAX_STEPS):
        # Accumulate gradient over N_EOT noise realizations
        grad_accum = torch.zeros_like(x_orig)
        loss_accum = 0.0
        for _ in range(N_EOT):
            d = delta.clone().detach().requires_grad_(True)
            x_adv = (x_orig + d).clamp(0.0, 1.0)
            x_resized = F.interpolate(x_adv, size=(IMAGE_SIZE, IMAGE_SIZE),
                                      mode="bilinear", align_corners=False, antialias=True)
            x_n11 = x_resized * 2.0 - 1.0
            noise = torch.randn_like(x_n11) * SIGMA
            x_noisy = (x_n11 + noise).clamp(-1.0, 1.0)
            logits = model(x_noisy)
            # loss to MAXIMIZE: P(natural) - P(synth) = -(logit_synth - logit_natural)
            # so for sign-grad ascent we want delta to MAXIMIZE this. Equivalently,
            # MINIMIZE loss = logit_synth - logit_natural and STEP -alpha * sign(grad).
            loss = logits[0, 1] - logits[0, 0]
            loss.backward()
            grad_accum += d.grad.detach()
            loss_accum += float(loss.item())
        grad_avg = grad_accum / N_EOT

        with torch.no_grad():
            delta = delta - ALPHA * grad_avg.sign()
            delta = delta.clamp(-EPS_CAP, EPS_CAP)
            delta = ((x_orig + delta).clamp(0.0, 1.0) - x_orig)

        # Evaluate smoothed P(natural) periodically (expensive)
        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            x_adv_resized = F.interpolate(
                (x_orig + delta).clamp(0.0, 1.0),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False, antialias=True)
            x_eval = x_adv_resized * 2.0 - 1.0
            p_smooth = smoothed_p_natural(model, x_eval, SIGMA, N_SMOOTH)
            d_max = float(delta.abs().max().item() * 255)
            d_mean = float(delta.abs().mean().item() * 255)
            avg_loss = loss_accum / N_EOT
            traj.append({"step": step, "p_natural_smooth": p_smooth,
                         "loss_avg": avg_loss,
                         "delta_max_units": d_max, "delta_mean_units": d_mean})
            print(f"  step {step:4d}  loss_avg={avg_loss:+.4f}  "
                  f"P(nat|smooth)={p_smooth:.4f}  "
                  f"mean|d|={d_mean:.2f}  max|d|={d_max:.1f}", flush=True)

            if p_smooth >= P_THRESHOLD and steps_to_threshold is None:
                steps_to_threshold = step
                delta_at_threshold = (d_max, d_mean)
                print(f"\n  -> SMOOTHED P(natural) >= {P_THRESHOLD} at step {step}  "
                      f"(max|d|={d_max:.1f}/255, mean|d|={d_mean:.2f}/255)", flush=True)
                break

    # save adversarial image
    with torch.no_grad():
        x_adv_final = (x_orig + delta).clamp(0.0, 1.0)
        arr = (x_adv_final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    adv_pil = Image.fromarray(arr)
    adv_pil.save(OUT_DIR / "adv_eot_native.png", "PNG", optimize=True)
    adv_pil.save(OUT_DIR / "adv_eot_native.jpg", "JPEG", quality=95, optimize=True)

    # final eval through matched-codec + smoothed
    final_pil = _load_and_match_codec(str(OUT_DIR / "adv_eot_native.jpg"), IMAGE_SIZE, jpeg_q=95)
    final_t = to_tensor_neg1_1(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        p_nat_nosmooth = float(F.softmax(model(final_t), dim=1)[0, 0].item())
        # Use a higher N for the final eval to reduce variance
        p_nat_smooth = smoothed_p_natural(model, final_t, SIGMA, 200)
    print(f"\nFINAL (attacked + matched-codec deployed eval):", flush=True)
    print(f"  P(natural) NO-SMOOTH = {p_nat_nosmooth:.4f}  "
          f"(baseline {p_natural_no_smooth:.4f})", flush=True)
    print(f"  P(natural) SMOOTHED  = {p_nat_smooth:.4f}     "
          f"(baseline {p_natural_smooth:.4f}, N=200)", flush=True)
    print(f"  detector (smoothed) verdict: "
          f"{'natural' if p_natural_smooth >= 0.5 else 'synthetic'} -> "
          f"{'natural' if p_nat_smooth >= 0.5 else 'synthetic'}", flush=True)

    results = {
        "src": SRC, "ckpt": CKPT, "ckpt_score": ckpt["score"],
        "sigma": SIGMA, "n_eot": N_EOT, "n_smooth": N_SMOOTH,
        "alpha_units": ALPHA * 255, "eps_cap_units": EPS_CAP * 255,
        "p_threshold": P_THRESHOLD,
        "baseline_p_natural_no_smooth": p_natural_no_smooth,
        "baseline_p_natural_smooth": p_natural_smooth,
        "final_p_natural_no_smooth": p_nat_nosmooth,
        "final_p_natural_smooth": p_nat_smooth,
        "steps_to_threshold": steps_to_threshold,
        "delta_max_at_threshold_units":
            delta_at_threshold[0] if delta_at_threshold else None,
        "delta_mean_at_threshold_units":
            delta_at_threshold[1] if delta_at_threshold else None,
        "trajectory": traj,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
