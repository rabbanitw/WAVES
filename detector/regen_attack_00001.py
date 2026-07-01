"""Regeneration attack against the Nano-Banana edit 00001.jpg.

Pipeline: SDXL img2img with variable strength. Low strength = light rinse
(preserves content, mild watermark removal). High strength = heavy resample
(removes watermark but may drift from original content).

Runs a strength/round ladder; scores each variant through baseline_r50 and
saves at native resolution so SynthID / Gemini can independently check
whether the diffusion watermark survives."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf_cache")
os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")
os.environ.setdefault("TORCH_HOME", "/mnt/data/torch_cache")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline

sys.path.insert(0, str(Path(__file__).parent))
from data import to_tensor_neg1_1
from models import BinaryClassifier

CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
SRC = Path("/home/trabbani/WAVES/coco_ood_examples/v2/edit/00001.jpg")
OUT_DIR = Path("/home/trabbani/WAVES/regen_attack_flips")
IMAGE_SIZE = 384
MODEL_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_INFERENCE_STEPS = 1000
GUIDANCE = 1.0
SEED = 1024

# (n_steps_per_round, n_rounds) - matches WAVES rinse notation
# 10 = light rinse, 20 = medium, 40 = heavy. 2 rounds = "rinse" (fresh noise).
LADDER = [
    (10, 1),   # 1x10 - single light regen
    (20, 1),
    (40, 1),
    (10, 2),   # 2x10 - light rinse
    (20, 2),   # 2x20 - medium rinse  (this is what WAVES uses)
    (40, 2),   # 2x40 - heavy rinse
]


def round8(x):
    return int(x) - (int(x) % 8)


def prep_for_sdxl(pil):
    """SDXL wants 8-divisible dims. Center-crop to next 8-multiple down."""
    W, H = pil.size
    W8, H8 = round8(W), round8(H)
    if (W8, H8) == (W, H):
        return pil
    left, top = (W - W8) // 2, (H - H8) // 2
    return pil.crop((left, top, left + W8, top + H8))


def regen_one(pipe, img, n_steps, seed):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt="", prompt_2="", image=img,
        strength=n_steps / NUM_INFERENCE_STEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    ).images[0]


def regen_rinse(pipe, img, n_steps, n_rounds, seed_base):
    out = img
    for r in range(n_rounds):
        out = regen_one(pipe, out, n_steps, seed_base + r)
    return out


@torch.no_grad()
def score(model, pil, device):
    pil2 = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    pil2.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pil3 = Image.open(buf).convert("RGB")
    x = to_tensor_neg1_1(pil3).unsqueeze(0).to(device)
    p = F.softmax(model(x), dim=1)[0]
    return float(p[0].item()), float(p[1].item())


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load detector
    print(f"loading {CKPT}", flush=True)
    detector = BinaryClassifier(pretrained=False, backbone="resnet50").to(device).eval()
    ck = torch.load(CKPT, map_location=device)
    detector.load_state_dict(ck["state_dict"])

    # Load SDXL img2img
    print(f"loading {MODEL_SDXL} ...", flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_SDXL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # Load source
    src_pil = Image.open(SRC).convert("RGB")
    print(f"src: {SRC.name}  native={src_pil.size}", flush=True)
    src_prepped = prep_for_sdxl(src_pil)
    if src_prepped.size != src_pil.size:
        print(f"  cropped to 8-multiple: {src_prepped.size}", flush=True)

    p_nat_orig, p_synth_orig = score(detector, src_pil, device)
    print(f"  baseline P(nat)={p_nat_orig:.4f}  P(synth)={p_synth_orig:.4f}",
          flush=True)

    # Save original at native resolution to compare against
    src_pil.save(OUT_DIR / "00_original.jpg", "JPEG", quality=95)

    rows = [{
        "name": "00_original",
        "path": str(OUT_DIR / "00_original.jpg"),
        "size": list(src_pil.size),
        "p_natural": p_nat_orig, "p_synth": p_synth_orig,
    }]

    for i, (n_steps, n_rounds) in enumerate(LADDER, start=1):
        label = f"{n_rounds}x{n_steps}" if n_rounds > 1 else f"1x{n_steps}"
        print(f"\n[{i}/{len(LADDER)}] regen {label} "
              f"(strength={n_steps/NUM_INFERENCE_STEPS:.3f})...", flush=True)
        out_pil = regen_rinse(pipe, src_prepped, n_steps, n_rounds, SEED)
        name = f"{i:02d}_regen_{label}"
        out_path = OUT_DIR / f"{name}.jpg"
        out_pil.save(out_path, "JPEG", quality=95)
        p_nat, p_synth = score(detector, out_pil, device)
        marker = "*** FLIPPED to natural" if p_nat > 0.5 else ""
        print(f"  size={out_pil.size}  P(nat)={p_nat:.4f}  "
              f"P(synth)={p_synth:.4f}  {marker}", flush=True)
        rows.append({
            "name": name, "path": str(out_path),
            "n_steps_per_round": n_steps, "n_rounds": n_rounds,
            "strength": n_steps / NUM_INFERENCE_STEPS,
            "size": list(out_pil.size),
            "p_natural": p_nat, "p_synth": p_synth,
        })

    manifest = {
        "source": str(SRC),
        "ckpt": CKPT,
        "detector": "baseline_r50 (best_ood_balanced_r50.pt)",
        "sdxl_model": MODEL_SDXL, "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance": GUIDANCE, "seed": SEED,
        "note": ("SDXL img2img regeneration attack. 'strength' scales the "
                 "amount of noise added before denoising: low strength = "
                 "light rinse (preserves content), high = heavy rewrite. "
                 "'NxM' means N rounds of regen at M inference steps each; "
                 "each round uses a fresh noise seed (WAVES rinse convention)."),
        "variants": rows,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {OUT_DIR}/manifest.json  ({len(rows)} images)", flush=True)


if __name__ == "__main__":
    main()
