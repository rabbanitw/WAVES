"""EOT-PGD attack against a 4-MODEL ENSEMBLE + randomized smoothing.

Ensemble = probability-average over the 4 noise-aug R18 seeds. At inference:
    P_ens(y|x) = mean_k E_noise[ softmax(model_k(x + noise)) ]

For the attack, EOT now needs gradients through ALL 4 models per noise sample.
This is the hardest single-image defense we can build with what we have.

If the ensemble + smoothing still falls in <=2 PGD steps, no inference-time
defense works for C_eval. Period."""

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
CKPT_DIR = Path("/home/trabbani/WAVES/detector/runs/adv_mixed_noiseaug_r18")
CKPT_NAMES = ["s44", "s48", "s53", "s55"]
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack_eot_vs_ensemble")

SIGMA = 0.025
N_EOT = 8           # noise samples per attack step (8 x 4 models = 32 fwds)
N_SMOOTH = 20
ALPHA = 1.0 / 255.0
MAX_STEPS = 200
EPS_CAP = 96.0 / 255.0
IMAGE_SIZE = 384
P_THRESHOLD = 0.999
LOG_EVERY = 5

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def smoothed_ens_p_natural(models, x_neg1_1, sigma, n_samples):
    probs_sum = torch.zeros(2, device=x_neg1_1.device)
    K = len(models)
    for _ in range(n_samples):
        noise = torch.randn_like(x_neg1_1) * sigma
        x_noisy = (x_neg1_1 + noise).clamp(-1.0, 1.0)
        for m in models:
            probs_sum += F.softmax(m(x_noisy), dim=1)[0]
    return float((probs_sum / (n_samples * K))[0].item())


def main():
    device = "cuda"
    models = []
    scores = []
    for name in CKPT_NAMES:
        ck_path = CKPT_DIR / name / "best.pt"
        print(f"loading {ck_path} ...", flush=True)
        m = BinaryClassifier(pretrained=False, backbone="resnet18").to(device).eval()
        ck = torch.load(ck_path, map_location=device)
        m.load_state_dict(ck["state_dict"])
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
        scores.append(float(ck["score"]))
    print(f"ensemble of {len(models)} R18 ckpts. Per-seed scores: {scores}",
          flush=True)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    x_orig = pil_to_chw01(src_pil).to(device)

    base_pil = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_t = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        per_seed_p = []
        for m in models:
            per_seed_p.append(float(F.softmax(m(base_t), dim=1)[0, 0].item()))
        # ensemble P(natural)
        avg_probs = torch.zeros(2, device=device)
        for m in models:
            avg_probs += F.softmax(m(base_t), dim=1)[0]
        p_natural_no_smooth = float((avg_probs / len(models))[0].item())
        p_natural_smooth = smoothed_ens_p_natural(models, base_t, SIGMA, N_SMOOTH)
    print(f"\nbaseline per-seed P(natural): "
          + ", ".join(f"{n}={p:.3f}" for n, p in zip(CKPT_NAMES, per_seed_p)),
          flush=True)
    print(f"ensemble P(natural)        (no smoothing)        = {p_natural_no_smooth:.4f}",
          flush=True)
    print(f"ensemble P(natural)        (smoothed, sigma={SIGMA}) = {p_natural_smooth:.4f}",
          flush=True)

    print(f"\n--- EOT-PGD vs ensemble (N_EOT={N_EOT}, sigma={SIGMA}, alpha=1/255, "
          f"eps_cap={EPS_CAP*255:.0f}/255, threshold={P_THRESHOLD}) ---", flush=True)
    delta = torch.zeros_like(x_orig)
    traj = []
    steps_to_threshold = None
    delta_at_threshold = None

    for step in range(MAX_STEPS):
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
            # Sum loss across all ensemble members for this noise realization
            loss = 0.0
            for m in models:
                logits = m(x_noisy)
                loss = loss + (logits[0, 1] - logits[0, 0])
            loss = loss / len(models)
            loss.backward()
            grad_accum += d.grad.detach()
            loss_accum += float(loss.item())
        grad_avg = grad_accum / N_EOT

        with torch.no_grad():
            delta = delta - ALPHA * grad_avg.sign()
            delta = delta.clamp(-EPS_CAP, EPS_CAP)
            delta = ((x_orig + delta).clamp(0.0, 1.0) - x_orig)

        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            x_adv_resized = F.interpolate(
                (x_orig + delta).clamp(0.0, 1.0),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False, antialias=True)
            x_eval = x_adv_resized * 2.0 - 1.0
            p_smooth = smoothed_ens_p_natural(models, x_eval, SIGMA, N_SMOOTH)
            d_max = float(delta.abs().max().item() * 255)
            d_mean = float(delta.abs().mean().item() * 255)
            avg_loss = loss_accum / N_EOT
            traj.append({"step": step, "p_natural_smooth": p_smooth,
                         "loss_avg": avg_loss,
                         "delta_max_units": d_max, "delta_mean_units": d_mean})
            print(f"  step {step:4d}  loss_avg={avg_loss:+.4f}  "
                  f"P(nat|ens-smooth)={p_smooth:.4f}  "
                  f"mean|d|={d_mean:.2f}  max|d|={d_max:.1f}", flush=True)

            if p_smooth >= P_THRESHOLD and steps_to_threshold is None:
                steps_to_threshold = step
                delta_at_threshold = (d_max, d_mean)
                print(f"\n  -> ENSEMBLE-SMOOTHED P(natural) >= {P_THRESHOLD} at step {step} "
                      f"(max|d|={d_max:.1f}/255, mean|d|={d_mean:.2f}/255)", flush=True)
                break

    with torch.no_grad():
        x_adv_final = (x_orig + delta).clamp(0.0, 1.0)
        arr = (x_adv_final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT_DIR / "adv_ens.png", "PNG", optimize=True)
    Image.fromarray(arr).save(OUT_DIR / "adv_ens.jpg", "JPEG", quality=95, optimize=True)

    final_pil = _load_and_match_codec(str(OUT_DIR / "adv_ens.jpg"), IMAGE_SIZE, jpeg_q=95)
    final_t = to_tensor_neg1_1(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        per_seed_p_final = []
        for m in models:
            per_seed_p_final.append(float(F.softmax(m(final_t), dim=1)[0, 0].item()))
        p_nat_smooth = smoothed_ens_p_natural(models, final_t, SIGMA, 200)
    print(f"\nFINAL per-seed P(natural): "
          + ", ".join(f"{n}={p:.3f}" for n, p in zip(CKPT_NAMES, per_seed_p_final)),
          flush=True)
    print(f"ensemble smoothed (N=200): {p_nat_smooth:.4f}  "
          f"(baseline {p_natural_smooth:.4f})", flush=True)
    print(f"verdict: {'natural' if p_natural_smooth >= 0.5 else 'synthetic'} -> "
          f"{'natural' if p_nat_smooth >= 0.5 else 'synthetic'}", flush=True)

    results = {
        "src": SRC, "ckpts": [str(CKPT_DIR / n / "best.pt") for n in CKPT_NAMES],
        "scores_per_seed": scores, "sigma": SIGMA, "n_eot": N_EOT,
        "baseline_p_natural_no_smooth": p_natural_no_smooth,
        "baseline_p_natural_smooth": p_natural_smooth,
        "final_p_natural_per_seed": per_seed_p_final,
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
