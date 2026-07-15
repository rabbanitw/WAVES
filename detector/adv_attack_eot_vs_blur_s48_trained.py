"""EOT-PGD attack against the BLUR-TRAINED s48 R18 model. Blur is folded into
the classifier: blur(x + delta) is the input the model sees. Attacker has
gradient through the blur, so the question is whether the model trained on
blurred inputs is robust because its features actually live in low frequencies.

Modes:
  A) Plain PGD through blur (no noise smoothing)
  B) EOT-PGD through blur + Gaussian noise smoothing (sigma=0.025)

Compare steps + max|delta| to flip P(natural) >= 0.999 against the prior
baselines (which all fell in 1-2 steps at max|d|=1-2/255)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import _load_and_match_codec, to_tensor_neg1_1
from models import BinaryClassifier

SRC = "/home/trabbani/WAVES/coco_ood_examples/v2/edit/00001.jpg"
CKPT = "/home/trabbani/WAVES/detector/best_adv_blur_r18.pt"
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack_eot_vs_blur_s48_trained")

BLUR_SIGMA = 1.0   # matches training
ALPHA = 1.0 / 255.0
MAX_STEPS = 500
EPS_CAP = 96.0 / 255.0
IMAGE_SIZE = 384
P_THRESHOLD = 0.999
LOG_EVERY = 10

NOISE_SIGMA = 0.025
N_EOT = 20

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def make_gaussian_kernel(sigma, device, ksize=None):
    if ksize is None:
        ksize = 2 * int(3 * sigma) + 1
    ax = torch.arange(ksize, device=device) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g1 = g1 / g1.sum()
    k2 = g1[:, None] * g1[None, :]
    return k2.view(1, 1, ksize, ksize).expand(3, 1, ksize, ksize).contiguous(), ksize


def blurred_forward(model, x_neg1_1, kernel, ksize):
    pad = ksize // 2
    x_blurred = F.conv2d(x_neg1_1, kernel, padding=pad, groups=3)
    return model(x_blurred)


@torch.no_grad()
def smoothed_p_natural(model, x_neg1_1, kernel, ksize, noise_sigma, n_samples):
    if noise_sigma == 0:
        return float(F.softmax(blurred_forward(model, x_neg1_1, kernel, ksize),
                               dim=1)[0, 0].item())
    probs_sum = torch.zeros(2, device=x_neg1_1.device)
    for _ in range(n_samples):
        noise = torch.randn_like(x_neg1_1) * noise_sigma
        x_noisy = (x_neg1_1 + noise).clamp(-1.0, 1.0)
        probs_sum += F.softmax(blurred_forward(model, x_noisy, kernel, ksize),
                               dim=1)[0]
    return float((probs_sum / n_samples)[0].item())


def attack(model, x_orig, kernel, ksize, mode, label, results_obj):
    device = x_orig.device
    print(f"\n========== Mode {mode} ({label}) ==========", flush=True)

    base_pil = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_t = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
    p_base = smoothed_p_natural(
        model, base_t, kernel, ksize,
        NOISE_SIGMA if mode == "B" else 0, 200 if mode == "B" else 1)
    print(f"baseline (matched-codec + blur)  P(nat) = {p_base:.4f}", flush=True)

    delta = torch.zeros_like(x_orig)
    traj = []
    steps_to_threshold = None
    delta_at_threshold = None

    n_eot_use = N_EOT if mode == "B" else 1
    for step in range(MAX_STEPS):
        grad_accum = torch.zeros_like(x_orig)
        loss_accum = 0.0
        for _ in range(n_eot_use):
            d = delta.clone().detach().requires_grad_(True)
            x_adv = (x_orig + d).clamp(0.0, 1.0)
            x_resized = F.interpolate(x_adv, size=(IMAGE_SIZE, IMAGE_SIZE),
                                      mode="bilinear", align_corners=False, antialias=True)
            x_n11 = x_resized * 2.0 - 1.0
            if mode == "B":
                noise = torch.randn_like(x_n11) * NOISE_SIGMA
                x_n11 = (x_n11 + noise).clamp(-1.0, 1.0)
            logits = blurred_forward(model, x_n11, kernel, ksize)
            loss = logits[0, 1] - logits[0, 0]
            loss.backward()
            grad_accum += d.grad.detach()
            loss_accum += float(loss.item())
        grad_avg = grad_accum / n_eot_use

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
            p_smooth = smoothed_p_natural(
                model, x_eval, kernel, ksize,
                NOISE_SIGMA if mode == "B" else 0, 50 if mode == "B" else 1)
            d_max = float(delta.abs().max().item() * 255)
            d_mean = float(delta.abs().mean().item() * 255)
            avg_loss = loss_accum / n_eot_use
            traj.append({"step": step, "p_natural": p_smooth,
                         "loss_avg": avg_loss,
                         "delta_max_units": d_max, "delta_mean_units": d_mean})
            print(f"  step {step:4d}  loss_avg={avg_loss:+.4f}  "
                  f"P(nat)={p_smooth:.4f}  "
                  f"mean|d|={d_mean:.2f}  max|d|={d_max:.1f}", flush=True)
            if p_smooth >= P_THRESHOLD and steps_to_threshold is None:
                steps_to_threshold = step
                delta_at_threshold = (d_max, d_mean)
                print(f"\n  -> P(natural) >= {P_THRESHOLD} at step {step} "
                      f"(max|d|={d_max:.1f}/255, mean|d|={d_mean:.2f}/255)", flush=True)
                break

    results_obj[label] = {
        "mode": mode, "n_eot": n_eot_use,
        "baseline_p_natural_smooth_matched_codec": p_base,
        "steps_to_threshold": steps_to_threshold,
        "delta_max_at_threshold_units":
            delta_at_threshold[0] if delta_at_threshold else None,
        "delta_mean_at_threshold_units":
            delta_at_threshold[1] if delta_at_threshold else None,
        "trajectory": traj,
    }


def main():
    device = "cuda"
    print(f"loading {CKPT}", flush=True)
    model = BinaryClassifier(pretrained=False, backbone="resnet18").to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded blur-trained s48 (score = {ckpt['score']:+.4f})", flush=True)

    kernel, ksize = make_gaussian_kernel(BLUR_SIGMA, device)
    print(f"blur kernel: sigma={BLUR_SIGMA}, ksize={ksize}", flush=True)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    x_orig = pil_to_chw01(src_pil).to(device)

    results_obj = {}
    attack(model, x_orig, kernel, ksize, mode="A",
           label=f"plain_PGD_through_blur", results_obj=results_obj)
    attack(model, x_orig, kernel, ksize, mode="B",
           label=f"EOT_PGD_blur+noise_sigma{NOISE_SIGMA}",
           results_obj=results_obj)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"ckpt": CKPT, "src": SRC, "blur_sigma": BLUR_SIGMA,
                   "results": results_obj}, f, indent=2)
    print(f"\nwrote {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
