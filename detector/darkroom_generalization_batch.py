"""Aggregate darkroom-attack generalization: run the 65-op sweep on many
COCO edits and compute per-op flip rates for baseline_r50 vs regen_aug_s55.

Filter: only include images where BOTH detectors initially call P(synth) >=
0.7 on the un-attacked original. That way any flip after an op is
unambiguous ('correctly-classified synth becomes wrongly-classified natural')."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from data import to_tensor_neg1_1
from models import BinaryClassifier

IMG_DIR = Path("/home/trabbani/WAVES/coco_ood_examples/v2/edit")
IMAGE_SIZE = 384
MAX_IMAGES = 40   # scan up to this many, keep the ones both detectors call synth
MIN_P_SYNTH = 0.70
OUT = Path("/home/trabbani/WAVES/detector/runs/darkroom_batch.json")

TARGETS = [
    ("baseline_r50",       "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt",         "resnet50"),
    ("regen_aug_s55",      "/home/trabbani/WAVES/detector/best_regen_aug_r18.pt",            "resnet18"),
    ("regen_geom_aug_s48", "/home/trabbani/WAVES/detector/best_regen_geom_aug_r18.pt",       "resnet18"),
]


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
    return pil.resize((nw, nh), Image.LANCZOS).resize((w, h), Image.LANCZOS)
def op_gamma(pil, g):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return Image.fromarray((np.clip(arr ** g, 0, 1) * 255).astype(np.uint8))
def op_noise(pil, s):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32)
    return Image.fromarray(
        np.clip(arr + np.random.randn(*arr.shape) * s, 0, 255).astype(np.uint8))
def op_rotate(pil, deg):
    return pil.rotate(deg, resample=Image.BILINEAR, expand=False)
def op_crop_center(pil, keep):
    w, h = pil.size
    kw, kh = int(w * keep), int(h * keep)
    left, top = (w - kw) // 2, (h - kh) // 2
    return pil.crop((left, top, left + kw, top + kh)).resize(
        (w, h), Image.LANCZOS)


SWEEPS = [
    ("blur",     op_blur,     [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]),
    ("bright",   op_bright,   [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2]),
    ("sat",      op_sat,      [0.0, 0.5, 1.5, 2.0, 3.0]),
    ("contrast", op_contrast, [0.5, 0.7, 0.9, 1.3, 1.6, 2.0]),
    ("sharp",    op_sharp,    [0.0, 0.5, 2.0, 4.0, 8.0, 12.0]),
    ("jpeg",     op_jpeg,     [95, 90, 75, 50, 30, 20, 10]),
    ("down_up",  op_down_up,  [0.75, 0.5, 0.33, 0.25, 0.15]),
    ("gamma",    op_gamma,    [0.5, 0.7, 0.9, 1.4, 2.0, 3.0]),
    ("noise",    op_noise,    [2, 5, 10, 15, 20, 25, 30, 40]),
    ("rotate",   op_rotate,   [1, 3, 5, 10]),
    ("crop",     op_crop_center, [0.95, 0.90, 0.80, 0.60]),
]


@torch.no_grad()
def score(model, pil, device):
    pil2 = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    pil2.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pil3 = Image.open(buf).convert("RGB")
    x = to_tensor_neg1_1(pil3).unsqueeze(0).to(device)
    return float(F.softmax(model(x), dim=1)[0, 0].item())


def main():
    np.random.seed(0)
    device = "cuda"
    models = {}
    for name, ck, bb in TARGETS:
        m = BinaryClassifier(pretrained=False, backbone=bb).to(device).eval()
        m.load_state_dict(torch.load(ck, map_location=device)["state_dict"])
        models[name] = m

    # Pick images both detectors confidently call synth
    all_paths = sorted(IMG_DIR.glob("*.jpg"))[:MAX_IMAGES]
    kept = []
    for p in all_paths:
        pil = Image.open(p).convert("RGB")
        p_nat = {n: score(m, pil, device) for n, m in models.items()}
        if all(1.0 - v >= MIN_P_SYNTH for v in p_nat.values()):
            kept.append(p)
    print(f"kept {len(kept)}/{len(all_paths)} images where both detectors "
          f"say P(synth)>={MIN_P_SYNTH} on original", flush=True)
    if not kept:
        print("no images meet the confidence bar")
        return
    for p in kept:
        print(f"  {p.name}")

    # Aggregate flip counts per (op, arg, model)
    # Structure: results[op_name][arg_idx][model_name] = list of P(nat) values,
    # one per image.
    agg = {op: {} for op, _, _ in SWEEPS}
    for op_name, op_fn, params in SWEEPS:
        for arg in params:
            agg[op_name][str(arg)] = {n: [] for n in models}
        print(f"\n--- {op_name} ---", flush=True)
        for arg in params:
            for path in kept:
                pil = Image.open(path).convert("RGB")
                pil_out = op_fn(pil, arg)
                for n, m in models.items():
                    p_nat = score(m, pil_out, device)
                    agg[op_name][str(arg)][n].append(p_nat)
            # print row
            names = list(models.keys())
            parts = []
            for n in names:
                vals = agg[op_name][str(arg)][n]
                flip = sum(1 for v in vals if v > 0.5)
                parts.append(f"{n[:20]}: mean={float(np.mean(vals)):.3f} flip={flip}/{len(kept)}")
            print(f"  {op_name}={str(arg):<6}  " + "  |  ".join(parts), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"kept": [p.name for p in kept], "agg": agg}, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)

    # Category-level rollup
    print(f"\n{'='*70}")
    print(f"CATEGORY ROLLUP (mean flip rate across all values of each op)")
    print(f"{'='*70}")
    n = len(kept)
    names = list(agg[SWEEPS[0][0]][str(SWEEPS[0][2][0])].keys())
    print("  op         | " + " | ".join(f"{n[:18]:>18s}" for n in names))
    grand = {n: 0 for n in names}
    grand_total = 0
    for op_name, _, params in SWEEPS:
        counts = {n: sum(sum(1 for v in agg[op_name][str(a)][n] if v > 0.5) for a in params) for n in names}
        total  = len(params) * n
        for k, c in counts.items():
            grand[k] += c
        grand_total += total
        parts = [f"{c:>4d}/{total:>4d} {c/total:>6.1%}" for c in counts.values()]
        print(f"  {op_name:<10s} | " + " | ".join(parts))
    print("  " + "-" * 60)
    parts = [f"{grand[n]:>4d}/{grand_total:>4d} {grand[n]/grand_total:>6.1%}" for n in names]
    print(f"  {'OVERALL':<10s} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
