"""Compute PSNR between (orig, edit) for every Pico-Banana photoreal pair,
at 384x384 with matched-codec JPEG q=95 (same pipeline the detector eats).

Saves a JSON keyed by pair slot -> psnr. Downstream: filter training to
the high-PSNR (near-identical) subset so the detector must learn a
diffusion fingerprint substrate, not semantic content differences."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import load_metadata

PHOTOREAL_ROOT = "/mnt/data/pico-banana-400k/photoreal_v2"
PHOTOREAL_META = "/mnt/data/pico-banana-400k/photoreal_v2/metadata.jsonl"
IMAGE_SIZE = 384
JPEG_Q = 95
OUT = Path("/mnt/data/pico_pair_psnr_photoreal_v2.json")


def load_matched(path):
    pil = Image.open(path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_Q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)


def psnr(a, b):
    mse = np.mean((a - b) ** 2) + 1e-12
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def main():
    items = load_metadata(PHOTOREAL_META, photoreal_only=True)
    print(f"pairs: {len(items)}", flush=True)

    scores = {}
    if OUT.exists():
        scores = json.load(open(OUT))
        print(f"resuming from {len(scores)} already-scored pairs", flush=True)

    for i, d in enumerate(items):
        slot = d["slot"]
        if slot in scores:
            continue
        try:
            a = load_matched(os.path.join(PHOTOREAL_ROOT, d["src_path"]))
            b = load_matched(os.path.join(PHOTOREAL_ROOT, d["edit_path"]))
            scores[slot] = float(psnr(a, b))
        except Exception as e:
            print(f"skip slot {slot}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(items)}  last psnr={scores.get(slot, float('nan')):.2f}",
                  flush=True)
            # atomic save
            tmp = str(OUT) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(scores, f)
            os.replace(tmp, str(OUT))
    # final save
    with open(OUT, "w") as f:
        json.dump(scores, f)
    print(f"wrote {OUT}  ({len(scores)} pairs)", flush=True)

    # Summary stats + percentiles
    vals = np.array(list(scores.values()))
    print(f"\nPSNR distribution across {len(vals)} pairs:", flush=True)
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:>2d} = {np.percentile(vals, q):.2f} dB", flush=True)
    print(f"  mean = {vals.mean():.2f} dB, std = {vals.std():.2f} dB", flush=True)
    for thr in [15, 20, 25, 30, 35, 40]:
        n = int((vals >= thr).sum())
        print(f"  #pairs with PSNR >= {thr}: {n}  ({100*n/len(vals):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
