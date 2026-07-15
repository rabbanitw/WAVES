"""Full darkroom-attack generalization profile of regen_aug_s55.

For a strongly-synth-classified NB edit (00001.jpg baseline P(synth)=0.996 on
baseline_r50, P(synth)=0.995 on regen_aug_s55), sweep a broad set of PIL image
operations and report side-by-side P(natural) for baseline_r50 vs s55.

If s55 was a real defense generalization (not just a regen-specific trick),
it should hold on more single-op transforms than baseline_r50, especially the
ones that flipped baseline_r50 (brightness, sharpen, noise)."""

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

SRC = Path("/home/trabbani/WAVES/coco_ood_examples/v2/edit/00001.jpg")
IMAGE_SIZE = 384
OUT = Path("/home/trabbani/WAVES/detector/runs/darkroom_generalization_s55.json")

TARGETS = [
    ("baseline_r50",  "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt",  "resnet50"),
    ("regen_aug_s55", "/home/trabbani/WAVES/detector/best_regen_aug_r18.pt",     "resnet18"),
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
    return Image.fromarray(np.clip(arr + np.random.randn(*arr.shape) * s, 0, 255).astype(np.uint8))
def op_rotate(pil, deg):
    return pil.rotate(deg, resample=Image.BILINEAR, expand=False)
def op_crop_center(pil, keep):
    w, h = pil.size
    kw, kh = int(w * keep), int(h * keep)
    left, top = (w - kw) // 2, (h - kh) // 2
    return pil.crop((left, top, left + kw, top + kh)).resize((w, h), Image.LANCZOS)


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

    src_pil = Image.open(SRC).convert("RGB")
    print(f"src={SRC.name}  native={src_pil.size}")
    p_base = {n: score(m, src_pil, device) for n, m in models.items()}
    print(f"baseline P(nat): "
          + "  ".join(f"{n}={v:.4f}" for n, v in p_base.items()))

    all_rows = {"baseline_p_natural": p_base, "sweeps": {}}
    n_flip_baseline = 0
    n_flip_s55      = 0
    n_ops           = 0

    for name, op, params in SWEEPS:
        print(f"\n--- {name} ---")
        rows = []
        print(f"  {'value':>8s} | "
              + "  ".join(f"{n:>16s}" for n in TARGETS[0][:1] + TARGETS[1][:1]))
        # header
        print(f"  {'value':>8s} | "
              f"{'baseline_r50':>14s}   {'regen_aug_s55':>15s}")
        for arg in params:
            pil = op(src_pil.convert("RGB"), arg)
            p = {n: score(m, pil, device) for n, m in models.items()}
            n_ops += 1
            fb = p["baseline_r50"] > 0.5
            fs = p["regen_aug_s55"] > 0.5
            n_flip_baseline += int(fb)
            n_flip_s55      += int(fs)
            mb = "*" if fb else " "
            ms = "*" if fs else " "
            print(f"  {str(arg):>8s} | "
                  f"{p['baseline_r50']:>12.4f}{mb}   "
                  f"{p['regen_aug_s55']:>13.4f}{ms}")
            rows.append({"arg": arg, **p})
        all_rows["sweeps"][name] = rows

    all_rows["summary"] = {
        "n_ops":               n_ops,
        "n_flipped_baseline":  n_flip_baseline,
        "n_flipped_s55":       n_flip_s55,
    }
    print(f"\n{'='*60}")
    print(f"SUMMARY across {n_ops} single-op variants:")
    print(f"  baseline_r50 fooled by {n_flip_baseline} / {n_ops}  "
          f"({n_flip_baseline/n_ops:.1%})")
    print(f"  regen_aug_s55 fooled by {n_flip_s55} / {n_ops}  "
          f"({n_flip_s55/n_ops:.1%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
