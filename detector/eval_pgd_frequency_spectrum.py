"""Measure WHERE in the frequency spectrum PGD perturbations live.

Hypothesis (consistent with the in-processing watermark framing): pixel-space
PGD's sign-of-gradient updates concentrate energy in HIGH frequencies (because
sign() of any reasonable gradient is largely orthogonal to natural images,
which are low-pass).

We attack each detector on N COCO edits with a short PGD run, compute the
2D DFT magnitude of each |delta|, and report % of total spectral energy in
each of 4 radial bands. We also compare to the spectrum of the underlying
clean image so the contrast is honest."""

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

IMG_DIR = Path("/home/trabbani/WAVES/coco_ood_examples/v2/edit")
TARGETS = [
    ("baseline_r50",        "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt",      "resnet50"),
    ("mixed_s48_r18",       "/home/trabbani/WAVES/detector/best_adv_mixed_r18.pt",         "resnet18"),
    ("noiseaug_s53_r18",    "/home/trabbani/WAVES/detector/best_adv_mixed_noiseaug_r18.pt","resnet18"),
    ("multiatk_s48_r18",    "/home/trabbani/WAVES/detector/best_adv_multiattacker_r18.pt", "resnet18"),
]
IMAGE_SIZE = 384
N_IMAGES = 10
PGD_STEPS = 8           # short attack, enough to converge on these models
ALPHA = 1.0 / 255.0
EPS_CAP = 8.0 / 255.0   # imperceptible-zone
OUT = Path("/home/trabbani/WAVES/detector/runs/pgd_frequency_spectrum.json")


def pgd_attack(model, x_neg1_1, y_target, alpha, eps_cap, k):
    """Untargeted PGD: maximize CE on y_correct (or equivalently logit_synth -
    logit_natural for a single image labeled synth). We just want delta, not
    p_natural saturation, so a short fixed-step run is fine."""
    x_orig = x_neg1_1.detach()
    delta = torch.zeros_like(x_orig)
    for _ in range(k):
        delta.requires_grad_(True)
        x_pert = (x_orig + delta).clamp(-1.0, 1.0)
        logits = model(x_pert)
        # Increase P(natural) <=> minimize (logit_synth - logit_natural)
        loss = logits[0, 1] - logits[0, 0]
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * 2.0 * grad.sign()   # *2 because eps is in [-1,1] space
            delta = delta.clamp(-eps_cap * 2.0, eps_cap * 2.0)
            delta = ((x_orig + delta).clamp(-1.0, 1.0) - x_orig)
    return delta.detach()


def radial_band_energy(spec_mag):
    """Given a 2D DFT magnitude (already fftshifted), return % of total energy
    in 4 radial bands: low (0-25%), mid-low (25-50%), mid-high (50-75%), high
    (75-100%) of the Nyquist range."""
    H, W = spec_mag.shape
    cy, cx = H // 2, W // 2
    y = np.arange(H)[:, None] - cy
    x = np.arange(W)[None, :] - cx
    r = np.sqrt(y * y + x * x)
    r_max = np.sqrt(cy * cy + cx * cx)
    r_norm = r / r_max
    e_total = (spec_mag ** 2).sum() + 1e-12
    bands = {
        "low_0_25":     ((spec_mag ** 2) * (r_norm <= 0.25)).sum() / e_total,
        "mid_25_50":    ((spec_mag ** 2) * (r_norm > 0.25) * (r_norm <= 0.50)).sum() / e_total,
        "mid_50_75":    ((spec_mag ** 2) * (r_norm > 0.50) * (r_norm <= 0.75)).sum() / e_total,
        "high_75_100":  ((spec_mag ** 2) * (r_norm > 0.75)).sum() / e_total,
    }
    return {k: float(v) for k, v in bands.items()}


def spectrum_of_tensor(x):
    """x: torch tensor (3, H, W), returns per-channel-averaged DFT magnitude."""
    arr = x.detach().cpu().numpy()
    mag_acc = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float64)
    for c in range(arr.shape[0]):
        F = np.fft.fftshift(np.fft.fft2(arr[c]))
        mag_acc += np.abs(F)
    return mag_acc / arr.shape[0]


def main():
    device = "cuda"
    paths = sorted(IMG_DIR.glob("*.jpg"))[:N_IMAGES]
    print(f"using {len(paths)} COCO edits", flush=True)

    results = {}
    for name, ckpt_path, backbone in TARGETS:
        print(f"\n========================================================")
        print(f"  {name}  (backbone={backbone})")
        print(f"========================================================", flush=True)
        model = BinaryClassifier(pretrained=False, backbone=backbone).to(device).eval()
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        for p in model.parameters():
            p.requires_grad_(False)

        clean_spec = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        delta_spec = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        successes = 0
        for p in paths:
            base_pil = _load_and_match_codec(str(p), IMAGE_SIZE, jpeg_q=95)
            x = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
            # PGD on x; goal: flip P(natural) up
            delta = pgd_attack(model, x, None, ALPHA, EPS_CAP, PGD_STEPS)
            with torch.no_grad():
                p_orig = F.softmax(model(x), dim=1)[0, 0].item()
                p_adv  = F.softmax(model((x + delta).clamp(-1, 1)), dim=1)[0, 0].item()
            successes += int(p_adv > 0.5)
            clean_spec += spectrum_of_tensor(x[0])
            delta_spec += spectrum_of_tensor(delta[0])
            print(f"  {p.name}: P(nat) {p_orig:.3f} -> {p_adv:.3f}  "
                  f"max|d|={delta.abs().max().item() * 127.5:.1f}/255 (in [-1,1] eps_cap=8/255)",
                  flush=True)
        clean_spec /= len(paths)
        delta_spec /= len(paths)
        clean_bands = radial_band_energy(clean_spec)
        delta_bands = radial_band_energy(delta_spec)
        results[name] = {
            "successes": successes, "n": len(paths),
            "clean_bands": clean_bands, "delta_bands": delta_bands,
        }

        print(f"\n  attack success: {successes}/{len(paths)}", flush=True)
        print(f"  CLEAN image spectrum:", flush=True)
        for k, v in clean_bands.items():
            print(f"     {k:>13s}  {v*100:6.2f}%", flush=True)
        print(f"  PGD DELTA spectrum:", flush=True)
        for k, v in delta_bands.items():
            print(f"     {k:>13s}  {v*100:6.2f}%", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)

    print(f"\n========== SUMMARY (% energy in each band) ==========")
    print(f"{'model':<22s} {'side':<7s} | "
          f"{'low':>7s} {'midlo':>7s} {'midhi':>7s} {'high':>7s}")
    for name, r in results.items():
        cb = r["clean_bands"]; db = r["delta_bands"]
        print(f"{name:<22s} {'clean':<7s} | "
              f"{cb['low_0_25']*100:>6.2f}% {cb['mid_25_50']*100:>6.2f}% "
              f"{cb['mid_50_75']*100:>6.2f}% {cb['high_75_100']*100:>6.2f}%")
        print(f"{'':<22s} {'delta':<7s} | "
              f"{db['low_0_25']*100:>6.2f}% {db['mid_25_50']*100:>6.2f}% "
              f"{db['mid_50_75']*100:>6.2f}% {db['high_75_100']*100:>6.2f}%")


if __name__ == "__main__":
    main()
