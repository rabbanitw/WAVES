"""Pre-generate SDXL 1x10 (strength=0.01) regens of training pairs +
OOD-COCO eval sets. Regen output stored at 384x384 JPEG q95 (matches the
detector input pipeline) so training can load them as drop-in samples."""

from __future__ import annotations

import io
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf_cache")
os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")
os.environ.setdefault("TORCH_HOME", "/mnt/data/torch_cache")

import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline

sys.path.insert(0, str(Path(__file__).parent))
from data import load_metadata

# ---- config ----
MODEL_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_INFERENCE_STEPS = 1000
N_STEPS_REGEN = 10   # -> strength = 0.010
GUIDANCE = 1.0
SEED_BASE = 1024
IMAGE_SIZE = 384

PHOTOREAL_ROOT = "/mnt/data/pico-banana-400k/sample_6k"
PHOTOREAL_META = "/mnt/data/pico-banana-400k/sample_6k/metadata.jsonl"
COCO_OOD_ROOT = Path("/mnt/data/coco_ood_v2_500")

OUT_ROOT = Path("/mnt/data/pico_regen_10step_v1")
N_TRAIN_PAIRS = 2500   # 5000 images
JPEG_Q = 95


def prepare_input(pil):
    pil = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    # matched-codec pass at q95 first, to match what the detector eats
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_Q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def regen_one(pipe, pil, seed):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt="", prompt_2="", image=pil,
        strength=N_STEPS_REGEN / NUM_INFERENCE_STEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE, generator=gen,
    ).images[0]


def do_dir(pipe, in_paths, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for k, path in enumerate(in_paths):
        outp = out_dir / f"{Path(path).stem}.jpg"
        if outp.exists():
            done += 1
            continue
        try:
            pil = Image.open(path)
        except Exception as e:
            print(f"  skip {path}: {e}", flush=True)
            continue
        pil = prepare_input(pil)
        out = regen_one(pipe, pil, SEED_BASE + k)
        # save at 384 q95 (matches detector input distribution)
        out.save(outp, "JPEG", quality=JPEG_Q)
        done += 1
        if done % 100 == 0:
            print(f"  [{tag}] {done}/{len(in_paths)}", flush=True)
    print(f"  [{tag}] DONE {done}/{len(in_paths)}", flush=True)


def main():
    device = "cuda"
    print(f"loading SDXL...", flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_SDXL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # ---- training subset ----
    items = load_metadata(PHOTOREAL_META, photoreal_only=True)
    rng = random.Random(42)
    rng.shuffle(items)
    train_items = items[:N_TRAIN_PAIRS]
    print(f"training pairs: {len(train_items)}", flush=True)

    natural_paths = [os.path.join(PHOTOREAL_ROOT, d["src_path"]) for d in train_items]
    synth_paths   = [os.path.join(PHOTOREAL_ROOT, d["edit_path"]) for d in train_items]

    do_dir(pipe, natural_paths, OUT_ROOT / "train_natural", "train_natural")
    do_dir(pipe, synth_paths,   OUT_ROOT / "train_synth",   "train_synth")

    # ---- OOD-COCO eval sets ----
    coco_edit_paths = sorted((COCO_OOD_ROOT / "edit").glob("*.jpg"))
    coco_nat_paths  = sorted((COCO_OOD_ROOT / "orig").glob("*.jpg"))
    do_dir(pipe, [str(p) for p in coco_edit_paths],
           OUT_ROOT / "ood_coco_edit", "ood_coco_edit")
    do_dir(pipe, [str(p) for p in coco_nat_paths],
           OUT_ROOT / "ood_coco_nat", "ood_coco_nat")

    # index
    idx = {
        "sdxl_model": MODEL_SDXL,
        "n_steps_regen": N_STEPS_REGEN,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance": GUIDANCE, "seed_base": SEED_BASE,
        "image_size": IMAGE_SIZE, "jpeg_q": JPEG_Q,
        "train_pairs_ids": [d["slot"] for d in train_items],
        "n_train_pairs": N_TRAIN_PAIRS,
    }
    with open(OUT_ROOT / "index.json", "w") as f:
        json.dump(idx, f, indent=2)
    print(f"wrote {OUT_ROOT}/index.json", flush=True)


if __name__ == "__main__":
    main()
