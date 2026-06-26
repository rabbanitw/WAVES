"""Same as regen_lpips_pairs.py but on sample_6k/ (raw 1024-class edits and
1-22 MP originals, no preprocessing), so the hypothesis test runs on
unmolested image content."""

from __future__ import annotations

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

META = "/mnt/data/pico-banana-400k/sample_6k/metadata.jsonl"
ROOT = "/mnt/data/pico-banana-400k/sample_6k"
OUT = "/home/trabbani/WAVES/regen_lpips_pairs_sample6k.json"

N_PAIRS = 100
N_STEPS = [10, 20, 40, 60]
SEED = 1024


def lpips_pil(a, b, net, device):
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    at = torch.from_numpy(aa).permute(2, 0, 1).unsqueeze(0).to(device)
    bt = torch.from_numpy(bb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(net(at, bt).mean().item())


def main():
    device = "cuda"
    pipe = build_pipeline("CompVis/stable-diffusion-v1-4", device=device)
    print("SD loaded")
    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print("LPIPS loaded")

    # First N_PAIRS pairs from sample_6k metadata
    items = []
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            items.append(d)
            if len(items) == N_PAIRS:
                break
    print(f"selected {len(items)} pairs from sample_6k")

    rows = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        orig = Image.open(os.path.join(ROOT, item["src_path"])).convert("RGB").resize((512, 512), Image.LANCZOS)
        edit = Image.open(os.path.join(ROOT, item["edit_path"])).convert("RGB").resize((512, 512), Image.LANCZOS)

        rec = {"slot": item["slot"], "edit_type": item.get("edit_type", "")}
        for N in N_STEPS:
            r_orig = regen_symmetric(orig, pipe, n_steps=N, seed=SEED)
            r_edit = regen_symmetric(edit, pipe, n_steps=N, seed=SEED)
            rec[f"lpips_orig_N{N}"] = lpips_pil(orig, r_orig, net, device)
            rec[f"lpips_edit_N{N}"] = lpips_pil(edit, r_edit, net, device)
        rows.append(rec)

        if i % 5 == 0 or i == len(items):
            el = time.time() - t0
            rate = i / el
            print(f"  [{i:3d}/{len(items)}]  {el:5.0f}s  ETA {(len(items)-i)/rate/60:5.1f} min", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT}")

    print(f"\n{'N':>4s}  {'mean LPIPS(orig)':>16s}  {'mean LPIPS(edit)':>16s}  "
          f"{'edit>orig %':>12s}  {'mean gap':>10s}")
    for N in N_STEPS:
        lo = np.array([r[f"lpips_orig_N{N}"] for r in rows])
        le = np.array([r[f"lpips_edit_N{N}"] for r in rows])
        print(f"  {N:>3d}  {lo.mean():>16.4f}  {le.mean():>16.4f}  "
              f"{100*(le > lo).mean():>11.1f}%  {(le - lo).mean():>+10.4f}")


if __name__ == "__main__":
    main()
