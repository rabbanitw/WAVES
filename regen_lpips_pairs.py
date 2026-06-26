"""For 100 (orig=Open Images, edit=Nano-Banana) pairs, apply symmetric DDIM
regen at N=10,20,40,60 to BOTH sides of the pair and measure
LPIPS(input, regen(input)). Test whether NB edits suffer worse perceptual
degradation than the natural-photo originals under the same attack.

Outputs:
  regen_lpips_pairs.json     per-(pair, N) LPIPS for orig and edit
  prints per-N: fraction of pairs where LPIPS(edit) > LPIPS(orig), and
                 the mean LPIPS gap
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import lpips
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/trabbani/WAVES")
from regen import build_pipeline, regen_symmetric  # noqa: E402

META = "/mnt/data/pico-banana-400k/photoreal_v2/metadata.jsonl"
ROOT = "/mnt/data/pico-banana-400k/photoreal_v2"
OUT = "/home/trabbani/WAVES/regen_lpips_pairs.json"

N_PAIRS = 100
N_STEPS = [10, 20, 40, 60]
SEED = 1024


def lpips_pil(a: Image.Image, b: Image.Image, net, device) -> float:
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    at = torch.from_numpy(aa).permute(2, 0, 1).unsqueeze(0).to(device)
    bt = torch.from_numpy(bb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(net(at, bt).mean().item())


def main():
    device = "cuda"
    print(f"loading SD pipeline...")
    pipe = build_pipeline("CompVis/stable-diffusion-v1-4", device=device)
    print("loading LPIPS...")
    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # First N_PAIRS pairs from the original seed=100 batch (slot 0..34999)
    items = []
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            if d["slot"] < 35000:
                items.append(d)
            if len(items) == N_PAIRS:
                break
    items.sort(key=lambda d: d["slot"])
    print(f"selected {len(items)} pairs from photoreal_v2")

    rows = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        orig = Image.open(os.path.join(ROOT, item["src_path"])).convert("RGB").resize((512, 512), Image.LANCZOS)
        edit = Image.open(os.path.join(ROOT, item["edit_path"])).convert("RGB").resize((512, 512), Image.LANCZOS)

        rec = {"slot": item["slot"], "edit_type": item.get("edit_type", "")}
        for N in N_STEPS:
            r_orig = regen_symmetric(orig, pipe, n_steps=N, seed=SEED)
            r_edit = regen_symmetric(edit, pipe, n_steps=N, seed=SEED)
            l_orig = lpips_pil(orig, r_orig, net, device)
            l_edit = lpips_pil(edit, r_edit, net, device)
            rec[f"lpips_orig_N{N}"] = l_orig
            rec[f"lpips_edit_N{N}"] = l_edit
        rows.append(rec)

        if i % 5 == 0 or i == len(items):
            el = time.time() - t0
            rate = i / el
            eta = (len(items) - i) / rate
            print(f"  [{i:3d}/{len(items)}]  {el:5.0f}s elapsed  ETA {eta/60:5.1f} min", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT}")

    # Summary
    print(f"\n{'N':>4s}  {'mean LPIPS(orig)':>16s}  {'mean LPIPS(edit)':>16s}  "
          f"{'edit>orig %':>12s}  {'mean gap':>10s}")
    for N in N_STEPS:
        lo = np.array([r[f"lpips_orig_N{N}"] for r in rows])
        le = np.array([r[f"lpips_edit_N{N}"] for r in rows])
        frac = (le > lo).mean()
        gap = (le - lo).mean()
        print(f"  {N:>3d}  {lo.mean():>16.4f}  {le.mean():>16.4f}  "
              f"{100*frac:>11.1f}%  {gap:>+10.4f}")


if __name__ == "__main__":
    main()
