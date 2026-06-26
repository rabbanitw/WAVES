"""Run symmetric DDIM regen at N=80 (SD 1.4, dense 1000-step) on the
full 1904-image OOD set, save results at 384x384 JPEG q=95 to match the
detector's input format."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf_cache")
os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")
os.environ.setdefault("TORCH_HOME", "/mnt/data/torch_cache")

sys.path.insert(0, "/home/trabbani/WAVES")
from regen import build_pipeline, regen_symmetric

SRC_DIR = Path("/mnt/data/synthid_ood/jpg384")
OUT_DIR = Path("/mnt/data/synthid_ood_regen80/jpg384")
N_STEPS = 80
SEED = 1024
INTERNAL_SIZE = 512   # SD 1.4 native
SAVE_SIZE = 384       # match photoreal_v2 / detector input
JPEG_Q = 95

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = "cuda"
    print("loading SD 1.4 pipeline...", flush=True)
    pipe = build_pipeline("CompVis/stable-diffusion-v1-4", device=device)
    print("loaded", flush=True)

    paths = sorted(p for p in os.listdir(SRC_DIR) if p.endswith(".jpg"))
    print(f"{len(paths)} OOD images to attack", flush=True)

    # Resume: skip already done
    done = {p for p in os.listdir(OUT_DIR) if p.endswith(".jpg")}
    pending = [p for p in paths if p not in done]
    print(f"{len(done)} already done; {len(pending)} pending", flush=True)

    t0 = time.time()
    for i, name in enumerate(pending, 1):
        src = Image.open(SRC_DIR / name).convert("RGB")
        # regen_symmetric internally resizes to 512; we feed it the 384
        # image and let it round-trip there, then resize to 384 for save.
        out = regen_symmetric(src, pipe, n_steps=N_STEPS, seed=SEED)
        out = out.resize((SAVE_SIZE, SAVE_SIZE), Image.LANCZOS)
        out.save(OUT_DIR / name, "JPEG", quality=JPEG_Q, optimize=True)

        if i % 20 == 0 or i == len(pending):
            el = time.time() - t0
            rate = i / el
            eta_min = (len(pending) - i) / rate / 60
            print(f"  [{i:4d}/{len(pending)}]  {el:6.0f}s  ETA {eta_min:6.1f} min  "
                  f"({rate*60:.1f}/min)", flush=True)

    print(f"\ndone. wrote {len(pending)} attacked JPEGs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
