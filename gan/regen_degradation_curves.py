"""Test the hypothesis: PSNR/LPIPS between input and symmetric N-step regen
output degrades FASTER for Nano-Banana edits than for natural Open Images
photos, with the gap growing in N.

For 8 photoreal pairs (each spanning a different edit_type), sweep
N in {1, 5, 10, 20, 40, 80, 160} and run symmetric regen on BOTH the
Open Images source and the NB edit. Compute PSNR and LPIPS(input,
regen(input)) at each N for each side. Plot per-pair lines + the mean.

If the hypothesis holds, the "edit" (NB) curve should sit below the
"orig" (Open Images) curve in PSNR (and above it in LPIPS) at every N.
"""

import io
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/trabbani/WAVES/dev_test")
sys.path.insert(0, "/home/trabbani/WAVES/gan")
from regen_sweep import regen_symmetric  # noqa: E402

from data import is_photoreal  # noqa: E402
from diffusers import ReSDPipeline, DDIMScheduler  # noqa: E402
import lpips  # noqa: E402


SAMPLE_ROOT = "/home/trabbani/pico-banana-400k/sample_6k"
SAMPLE_META = "/home/trabbani/pico-banana-400k/sample_6k/metadata.jsonl"
OUT_DIR = "/home/trabbani/WAVES/gan/runs/regen_curves"
N_VALUES = [1, 5, 10, 20, 40, 80, 160]
K_PAIRS = 8


def psnr_pil(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float32)
    y = np.asarray(b.convert("RGB"), dtype=np.float32)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def lpips_pil(a: Image.Image, b: Image.Image, lpips_net, device) -> float:
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    at = torch.from_numpy(aa).permute(2, 0, 1).unsqueeze(0).to(device)
    bt = torch.from_numpy(bb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(lpips_net(at, bt).mean().item())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"

    # Pick K diverse photoreal pairs (one per edit_type)
    with open(SAMPLE_META) as f:
        meta = [json.loads(l) for l in f]
    photoreal = [d for d in meta if is_photoreal(d.get("edit_type", ""))]
    seen = set()
    picks = []
    for d in photoreal:
        if d["edit_type"] not in seen:
            seen.add(d["edit_type"])
            picks.append(d)
        if len(picks) == K_PAIRS:
            break
    print(f"selected {len(picks)} pairs (one per edit_type):")
    for i, d in enumerate(picks):
        print(f"  {i}  slot={d['slot']:>5}  {d['edit_type']}")

    # Pipe + LPIPS
    pipe = ReSDPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16"
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    # Run regen at each N on both orig and edit, record PSNR + LPIPS
    rows = []
    t0 = time.time()
    for pi, d in enumerate(picks):
        orig = Image.open(os.path.join(SAMPLE_ROOT, d["src_path"])).convert("RGB").resize((512, 512), Image.LANCZOS)
        edit = Image.open(os.path.join(SAMPLE_ROOT, d["edit_path"])).convert("RGB").resize((512, 512), Image.LANCZOS)
        for N in N_VALUES:
            r_orig = regen_symmetric(orig, pipe, n_steps=N)
            r_edit = regen_symmetric(edit, pipe, n_steps=N)
            p_o = psnr_pil(orig, r_orig); p_e = psnr_pil(edit, r_edit)
            l_o = lpips_pil(orig, r_orig, lpips_net, device)
            l_e = lpips_pil(edit, r_edit, lpips_net, device)
            rows.append(dict(pair=pi, slot=d["slot"], edit_type=d["edit_type"], N=N,
                             psnr_orig=p_o, psnr_edit=p_e,
                             lpips_orig=l_o, lpips_edit=l_e))
            elapsed = time.time() - t0
            print(f"  pair {pi} N={N:3d}  "
                  f"PSNR orig={p_o:5.2f} edit={p_e:5.2f}  |  "
                  f"LPIPS orig={l_o:.3f} edit={l_e:.3f}  ({elapsed:5.0f}s)", flush=True)

    # Save raw rows
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Plot
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    for pair_rows in by_pair.values():
        pair_rows.sort(key=lambda r: r["N"])

    fig, (axp, axl) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10.colors
    for pi, prs in by_pair.items():
        ns = [r["N"] for r in prs]
        axp.plot(ns, [r["psnr_orig"] for r in prs], "-",  color=colors[pi % 10], alpha=0.45)
        axp.plot(ns, [r["psnr_edit"] for r in prs], "--", color=colors[pi % 10], alpha=0.45)
        axl.plot(ns, [r["lpips_orig"] for r in prs], "-",  color=colors[pi % 10], alpha=0.45)
        axl.plot(ns, [r["lpips_edit"] for r in prs], "--", color=colors[pi % 10], alpha=0.45)

    # Mean curves
    ns_all = sorted({r["N"] for r in rows})
    def m(key, n):
        return float(np.mean([r[key] for r in rows if r["N"] == n]))
    mp_o = [m("psnr_orig", n) for n in ns_all]
    mp_e = [m("psnr_edit", n) for n in ns_all]
    ml_o = [m("lpips_orig", n) for n in ns_all]
    ml_e = [m("lpips_edit", n) for n in ns_all]
    axp.plot(ns_all, mp_o, "-",  color="black", lw=3, label="mean orig (Open Images)")
    axp.plot(ns_all, mp_e, "--", color="black", lw=3, label="mean edit (Nano-Banana)")
    axl.plot(ns_all, ml_o, "-",  color="black", lw=3, label="mean orig")
    axl.plot(ns_all, ml_e, "--", color="black", lw=3, label="mean edit")

    for ax, ylabel, title in [(axp, "PSNR (dB)", "PSNR vs regen depth"),
                              (axl, "LPIPS", "LPIPS vs regen depth")]:
        ax.set_xlabel("N (regen steps)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=150)
    print(f"\nsaved curves to {OUT_DIR}/curves.png")
    print(f"data: {OUT_DIR}/results.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
