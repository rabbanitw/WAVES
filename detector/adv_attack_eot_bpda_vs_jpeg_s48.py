"""EOT-PGD-BPDA attack against the JPEG-AUG-trained s48 R18 model with
INFERENCE-TIME JPEG TTA defense.

Inference pipeline: for each prediction, sample q from {30,50,75,95},
JPEG-encode the image, classify, average over N=20 samples.

JPEG quantization is non-differentiable (involves rounding). The attacker
uses BPDA: forward pass goes through actual PIL JPEG encode/decode,
backward pass treats it as identity. This is the standard adaptive attack
against non-differentiable defenses (Athalye et al. 2018).

If this also breaks at max|delta|<=2/255, then NO inference-time pixel-domain
defense works for C_eval -- regardless of training recipe or inference
preprocessing. Time to think outside pixel space (frequency-domain features
that don't depend on smooth gradients)."""

from __future__ import annotations

import io
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
CKPT = "/home/trabbani/WAVES/detector/best_adv_jpegaug_r18.pt"
OUT_DIR = Path("/home/trabbani/WAVES/adv_attack_eot_bpda_jpeg_s48")

JPEG_Q_TTA = [30, 50, 75, 95]   # matches training distribution
N_SMOOTH = 20                    # noise/q samples for smoothed eval
N_EOT = 8                        # BPDA gradient samples per PGD step
ALPHA = 1.0 / 255.0
MAX_STEPS = 500
EPS_CAP = 96.0 / 255.0
IMAGE_SIZE = 384
P_THRESHOLD = 0.999
LOG_EVERY = 10

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


class JPEGQuantBPDA(torch.autograd.Function):
    """Forward: PIL JPEG encode/decode at the specified quality.
    Backward: identity (BPDA -- treat the non-diff op as the identity).
    Input is in [-1, 1]."""

    @staticmethod
    def forward(ctx, x_neg1_1, quality):
        B, C, H, W = x_neg1_1.shape
        out = torch.empty_like(x_neg1_1)
        for b in range(B):
            arr = ((x_neg1_1[b] + 1.0) / 2.0).clamp(0, 1).mul(255).byte() \
                .permute(1, 2, 0).cpu().numpy()
            pil = Image.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            pil2 = Image.open(buf).convert("RGB")
            arr2 = np.asarray(pil2, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr2).permute(2, 0, 1).to(x_neg1_1.device)
            out[b] = (t * 2.0 - 1.0)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def smoothed_p_natural_jpeg(model, x_neg1_1, q_choices, n_samples):
    """Average P(natural) over JPEG-q-sampled forward passes."""
    probs_sum = torch.zeros(2, device=x_neg1_1.device)
    with torch.no_grad():
        for _ in range(n_samples):
            q = np.random.choice(q_choices)
            x_q = JPEGQuantBPDA.apply(x_neg1_1, q)
            probs_sum += F.softmax(model(x_q), dim=1)[0]
    return float((probs_sum / n_samples)[0].item())


def attack(model, x_orig, q_choices, n_eot, label, results_obj):
    device = x_orig.device
    print(f"\n========== {label} ==========", flush=True)
    print(f"  q_choices={q_choices}  N_EOT={n_eot}  N_smooth={N_SMOOTH}", flush=True)

    base_pil = _load_and_match_codec(SRC, IMAGE_SIZE, jpeg_q=95)
    base_t = to_tensor_neg1_1(base_pil).unsqueeze(0).to(device)
    p_base = smoothed_p_natural_jpeg(model, base_t, q_choices, 200)
    print(f"baseline (matched-codec + JPEG TTA) P(nat) = {p_base:.4f}", flush=True)

    delta = torch.zeros_like(x_orig)
    traj = []
    steps_to_threshold = None
    delta_at_threshold = None

    for step in range(MAX_STEPS):
        grad_accum = torch.zeros_like(x_orig)
        loss_accum = 0.0
        for _ in range(n_eot):
            d = delta.clone().detach().requires_grad_(True)
            x_adv = (x_orig + d).clamp(0.0, 1.0)
            x_resized = F.interpolate(x_adv, size=(IMAGE_SIZE, IMAGE_SIZE),
                                      mode="bilinear", align_corners=False, antialias=True)
            x_n11 = x_resized * 2.0 - 1.0
            q = np.random.choice(q_choices)
            x_q = JPEGQuantBPDA.apply(x_n11, q)
            logits = model(x_q)
            loss = logits[0, 1] - logits[0, 0]
            loss.backward()
            grad_accum += d.grad.detach()
            loss_accum += float(loss.item())
        grad_avg = grad_accum / n_eot

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
            p_smooth = smoothed_p_natural_jpeg(model, x_eval, q_choices, N_SMOOTH)
            d_max = float(delta.abs().max().item() * 255)
            d_mean = float(delta.abs().mean().item() * 255)
            avg_loss = loss_accum / n_eot
            traj.append({"step": step, "p_natural": p_smooth,
                         "loss_avg": avg_loss,
                         "delta_max_units": d_max, "delta_mean_units": d_mean})
            print(f"  step {step:4d}  loss_avg={avg_loss:+.4f}  "
                  f"P(nat|jpeg-TTA)={p_smooth:.4f}  "
                  f"mean|d|={d_mean:.2f}  max|d|={d_max:.1f}", flush=True)
            if p_smooth >= P_THRESHOLD and steps_to_threshold is None:
                steps_to_threshold = step
                delta_at_threshold = (d_max, d_mean)
                print(f"\n  -> P(natural) >= {P_THRESHOLD} at step {step} "
                      f"(max|d|={d_max:.1f}/255, mean|d|={d_mean:.2f}/255)", flush=True)
                break

    results_obj[label] = {
        "n_eot": n_eot, "q_choices": list(q_choices),
        "baseline_p_natural_smooth": p_base,
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
    print(f"loaded jpeg-aug s48 (score = {ckpt['score']:+.4f})", flush=True)

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC}  size: {src_pil.size}", flush=True)
    x_orig = pil_to_chw01(src_pil).to(device)

    results_obj = {}
    attack(model, x_orig, q_choices=JPEG_Q_TTA, n_eot=N_EOT,
           label="EOT_BPDA_JPEG-TTA_q30-95", results_obj=results_obj)
    # Also test with only heavy JPEG (q=30) for comparison
    attack(model, x_orig, q_choices=[30], n_eot=N_EOT,
           label="EOT_BPDA_JPEG_q30_only", results_obj=results_obj)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"ckpt": CKPT, "src": SRC, "results": results_obj}, f, indent=2)
    print(f"\nwrote {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
