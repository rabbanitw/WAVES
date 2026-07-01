"""Regenerate + save the darkroom-attack images that flipped baseline_r50
on 00001.jpg, plus the original for direct A/B comparison."""

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from data import to_tensor_neg1_1
from models import BinaryClassifier

CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
SRC = Path("/home/trabbani/WAVES/coco_ood_examples/v2/edit/00001.jpg")
OUT_DIR = Path("/home/trabbani/WAVES/darkroom_attack_flips")
IMAGE_SIZE = 384


def op_bright(pil, f):
    return ImageEnhance.Brightness(pil).enhance(f)


def op_sharp(pil, f):
    return ImageEnhance.Sharpness(pil).enhance(f)


def op_noise(pil, std_units):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32)
    n = np.random.randn(*arr.shape) * std_units
    return Image.fromarray(np.clip(arr + n, 0, 255).astype(np.uint8))


@torch.no_grad()
def score(model, pil, device):
    """Deploy pipeline: resize->JPEG q95->decode->tensor."""
    pil2 = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    pil2.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pil3 = Image.open(buf).convert("RGB")
    x = to_tensor_neg1_1(pil3).unsqueeze(0).to(device)
    p = F.softmax(model(x), dim=1)[0]
    return float(p[0].item()), float(p[1].item())


def main():
    np.random.seed(0)
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = BinaryClassifier(pretrained=False, backbone="resnet50").to(device).eval()
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["state_dict"])

    src_pil = Image.open(SRC).convert("RGB")

    variants = [
        ("00_original",        src_pil),
        ("01_bright1p5",       op_bright(src_pil, 1.5)),
        ("02_bright1p8",       op_bright(src_pil, 1.8)),
        ("03_sharp8",          op_sharp(src_pil, 8.0)),
        ("04_noise_std20",     op_noise(src_pil, 20)),
        ("05_noise_std40",     op_noise(src_pil, 40)),
    ]

    rows = []
    for name, pil in variants:
        # save at native resolution (what the user would see & feed to Gemini)
        native_path = OUT_DIR / f"{name}.jpg"
        pil.save(native_path, "JPEG", quality=95)
        p_nat, p_synth = score(model, pil, device)
        print(f"  {name:<24s} native={pil.size}  "
              f"P(nat)={p_nat:.4f}  P(synth)={p_synth:.4f}")
        rows.append({
            "name": name, "path": str(native_path),
            "size": list(pil.size),
            "p_natural": p_nat, "p_synth": p_synth,
        })

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump({
            "source": str(SRC),
            "ckpt": CKPT,
            "detector": "baseline_r50 (best_ood_balanced_r50.pt)",
            "note": ("All variants scored through the deploy pipeline: "
                     "resize 384x384 + JPEG q=95. Native-resolution files "
                     "saved so the user (or Gemini) sees the same variant."),
            "variants": rows,
        }, f, indent=2)
    print(f"\nwrote {OUT_DIR}/manifest.json  ({len(rows)} images)")


if __name__ == "__main__":
    main()
