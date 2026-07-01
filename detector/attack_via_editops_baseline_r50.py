"""Non-adversarial 'darkroom' attacks against baseline_r50: apply everyday
PIL image operations (blur, brightness, saturation, sharpen, JPEG requantize,
downsample+upsample, gamma) until the detector flips its call.

This is a completely different threat model from PGD -- no gradients, just
transforms an editor or a bad phone camera might apply. If baseline_r50 flips
under mild versions of these, the detector is brittle even in a 'clean-user'
threat model."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from data import _load_and_match_codec, to_tensor_neg1_1
from models import BinaryClassifier

CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
IMG_DIR = Path("/home/trabbani/WAVES/coco_ood_examples/v2/edit")
OUT_DIR = Path("/home/trabbani/WAVES/darkroom_attack_baseline_r50")
IMAGE_SIZE = 384
JPEG_Q_DEPLOY = 95  # matched-codec at deploy


def load_model(device):
    m = BinaryClassifier(pretrained=False, backbone="resnet50").to(device).eval()
    ck = torch.load(CKPT, map_location=device)
    m.load_state_dict(ck["state_dict"])
    return m


@torch.no_grad()
def score(model, pil, device):
    """Round-trip pil through matched-codec pipeline and return P(natural)."""
    # First: emulate the deploy pipeline: resize->JPEG q95->decode->to_tensor
    pil2 = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    pil2.save(buf, format="JPEG", quality=JPEG_Q_DEPLOY)
    buf.seek(0)
    pil3 = Image.open(buf).convert("RGB")
    x = to_tensor_neg1_1(pil3).unsqueeze(0).to(device)
    p = F.softmax(model(x), dim=1)[0]
    return float(p[0].item()), float(p[1].item())


# ---- individual operations ----
def op_blur(pil, r):        return pil.filter(ImageFilter.GaussianBlur(radius=r))
def op_bright(pil, f):      return ImageEnhance.Brightness(pil).enhance(f)
def op_sat(pil, f):         return ImageEnhance.Color(pil).enhance(f)
def op_contrast(pil, f):    return ImageEnhance.Contrast(pil).enhance(f)
def op_sharp(pil, f):       return ImageEnhance.Sharpness(pil).enhance(f)
def op_jpeg(pil, q):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")
def op_down_up(pil, factor):
    w, h = pil.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    small = pil.resize((nw, nh), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)
def op_gamma(pil, g):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return Image.fromarray((np.clip(arr ** g, 0, 1) * 255).astype(np.uint8))
def op_noise(pil, std_units):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32)
    n = np.random.randn(*arr.shape) * std_units
    return Image.fromarray(np.clip(arr + n, 0, 255).astype(np.uint8))


# ---- single-op sweeps ----
SINGLE_OP_SWEEPS = [
    ("blur",    op_blur,     [0.5, 1.0, 1.5, 2.0, 3.0]),
    ("bright",  op_bright,   [0.6, 0.8, 1.2, 1.5, 1.8]),
    ("sat",     op_sat,      [0.0, 0.4, 0.7, 1.5, 2.0, 3.0]),
    ("contrast",op_contrast, [0.5, 0.7, 1.3, 1.6, 2.0]),
    ("sharp",   op_sharp,    [0.0, 0.5, 2.0, 4.0, 8.0]),
    ("jpeg",    op_jpeg,     [90, 75, 50, 30, 20, 10]),
    ("down_up", op_down_up,  [0.75, 0.5, 0.33, 0.25, 0.15]),
    ("gamma",   op_gamma,    [0.5, 0.7, 1.4, 2.0, 3.0]),
    ("noise",   op_noise,    [2, 5, 10, 20, 40]),
]


def pick_source(model, device):
    """Find a COCO edit baseline_r50 confidently calls synth."""
    for k in range(50):
        p = IMG_DIR / f"{k:05d}.jpg"
        if not p.exists():
            continue
        pil = Image.open(p).convert("RGB")
        p_nat, p_synth = score(model, pil, device)
        if p_synth >= 0.98:
            return p, pil, p_nat
    raise RuntimeError("no strongly-synth image found in first 50")


def combo_sweep(model, src_pil, device):
    """Try a few sensible multi-op combos likely to defeat texture cues."""
    combos = [
        ("blur1.5+jpeg50",   [(op_blur, 1.5), (op_jpeg, 50)]),
        ("blur1.5+jpeg30",   [(op_blur, 1.5), (op_jpeg, 30)]),
        ("blur2+jpeg50",     [(op_blur, 2.0), (op_jpeg, 50)]),
        ("blur1+contrast0.7",[(op_blur, 1.0), (op_contrast, 0.7)]),
        ("down0.5+up+jpeg50",[(op_down_up, 0.5), (op_jpeg, 50)]),
        ("blur1+jpeg50+sat0.7",  [(op_blur, 1.0), (op_jpeg, 50), (op_sat, 0.7)]),
        ("blur2+jpeg30+bright1.2",[(op_blur, 2.0), (op_jpeg, 30), (op_bright, 1.2)]),
        ("gamma1.4+blur1+jpeg50",[(op_gamma, 1.4), (op_blur, 1.0), (op_jpeg, 50)]),
        ("noise10+blur1.5",  [(op_noise, 10), (op_blur, 1.5)]),
        ("down0.33+up+blur1+jpeg50", [(op_down_up, 0.33), (op_blur, 1.0), (op_jpeg, 50)]),
    ]
    rows = []
    for label, ops in combos:
        pil = src_pil.convert("RGB")
        for op, arg in ops:
            pil = op(pil, arg)
        p_nat, p_synth = score(model, pil, device)
        rows.append({"combo": label, "p_natural": p_nat, "p_synth": p_synth})
    return rows


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {CKPT}", flush=True)
    model = load_model(device)

    src_path, src_pil, p_nat_orig = pick_source(model, device)
    print(f"picked: {src_path.name}  size={src_pil.size}  "
          f"baseline P(natural)={p_nat_orig:.4f} "
          f"(P(synth)={1-p_nat_orig:.4f})", flush=True)
    src_pil.save(OUT_DIR / "orig.png", "PNG")

    results = {"src": str(src_path),
               "baseline_p_natural": p_nat_orig, "single_op": {}, "combos": []}

    print(f"\n{'='*70}\nSINGLE-OP SWEEPS\n{'='*70}", flush=True)
    for name, op, params in SINGLE_OP_SWEEPS:
        results["single_op"][name] = []
        print(f"\n--- {name} ---", flush=True)
        for arg in params:
            pil = op(src_pil.convert("RGB"), arg)
            p_nat, p_synth = score(model, pil, device)
            verdict = "*** FLIPPED to natural" if p_nat > 0.5 else ""
            print(f"  {name}={arg}   P(nat)={p_nat:.4f}  P(synth)={p_synth:.4f}  {verdict}",
                  flush=True)
            results["single_op"][name].append(
                {"arg": arg, "p_natural": p_nat, "p_synth": p_synth})

    print(f"\n{'='*70}\nCOMBO SWEEPS\n{'='*70}", flush=True)
    combos = combo_sweep(model, src_pil, device)
    for r in combos:
        verdict = "*** FLIPPED to natural" if r["p_natural"] > 0.5 else ""
        print(f"  {r['combo']:<40s} P(nat)={r['p_natural']:.4f}  "
              f"P(synth)={r['p_synth']:.4f}  {verdict}", flush=True)
    results["combos"] = combos

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save one representative flipping example if any
    print(f"\n{'='*70}\nBEST SINGLE-OP FLIPS\n{'='*70}", flush=True)
    for name, rows in results["single_op"].items():
        best = max(rows, key=lambda r: r["p_natural"])
        marker = "  ***" if best["p_natural"] > 0.5 else ""
        print(f"  {name}={best['arg']}  P(nat)={best['p_natural']:.4f}{marker}",
              flush=True)


if __name__ == "__main__":
    main()
