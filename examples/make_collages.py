"""Build 5-panel collages showing a valid_512 source image across
N = {0 (original), 10, 20, 40, 80} symmetric diffusive-regen depths.

Run from the repo root after the four regen output directories exist:
  dev_test/regen_N010/, regen_N020/, regen_N040_valid512/, regen_N080_valid512/
"""

import argparse
import os
from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = "images"
REGEN_DIRS = {
    10: "regen_outputs/N010",
    20: "regen_outputs/N020",
    40: "regen_outputs/N040",
    80: "regen_outputs/N080",
}
PANEL = 384  # downscale 512 -> 384 to keep collage size reasonable
LABEL_H = 40


def paths_for(prompt_idx: int):
    base = f"prompt_{prompt_idx}_attempt_1_img_0_512x512"
    src = os.path.join(SOURCE_DIR, f"{base}.jpg")
    regens = {n: os.path.join(d, f"{base}.png") for n, d in REGEN_DIRS.items()}
    return src, regens


def make_collage(prompt_idx: int, out_path: str):
    src, regens = paths_for(prompt_idx)
    panels = [(0, src)] + sorted(regens.items())
    images = []
    labels = []
    for n, p in panels:
        img = Image.open(p).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)
        images.append(img)
        labels.append("original" if n == 0 else f"N = {n} regen steps")

    w = PANEL * len(images)
    h = PANEL + LABEL_H
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * PANEL, LABEL_H))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        draw.text((i * PANEL + (PANEL - tw) // 2, (LABEL_H - th) // 2 - 2),
                  label, fill=(20, 20, 20), font=font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"wrote {out_path}  ({canvas.size[0]} x {canvas.size[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, nargs="+", default=[0, 25, 100])
    ap.add_argument("--out-dir", default="examples/regen_progression")
    args = ap.parse_args()
    for idx in args.prompts:
        make_collage(idx, os.path.join(args.out_dir, f"prompt_{idx:03d}_progression.png"))


if __name__ == "__main__":
    main()
