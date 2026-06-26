"""SDXL-native version of the regen-LPIPS hypothesis test.

For 100 sample_6k pairs: open both images at their native resolution
(no resampling), center-crop the original to the edit's exact size,
then run symmetric DDIM regen on each side at the SDXL training
resolution using SDXL-base-1.0 (img2img with 1000-step dense schedule
and strength = N/1000).

Outputs the same per-N orig vs edit LPIPS comparison as the SD 1.4
sample_6k run, but with no Lanczos resize anywhere in the pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time

import lpips
import numpy as np
import torch
from PIL import Image

# Cache SDXL weights on /mnt/data (root partition is tight)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf_cache")
os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")
os.makedirs("/mnt/data/hf_cache", exist_ok=True)

from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline  # noqa: E402

META = "/mnt/data/pico-banana-400k/sample_6k/metadata.jsonl"
ROOT = "/mnt/data/pico-banana-400k/sample_6k"
OUT = "/home/trabbani/WAVES/regen_lpips_pairs_sdxl.json"

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
N_PAIRS = 100
N_STEPS = [10, 20, 40, 60]
NUM_INFERENCE_STEPS = 1000  # dense schedule, same as SD 1.4 runs
SEED = 1024
GUIDANCE = 1.0  # unconditional regen


def round8(x):
    return int(x) - (int(x) % 8)


def center_crop(img, target_w, target_h):
    w, h = img.size
    assert w >= target_w and h >= target_h, f"need {target_w}x{target_h} but image is {w}x{h}"
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def lpips_pil(a, b, net, device):
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    at = torch.from_numpy(aa).permute(2, 0, 1).unsqueeze(0).to(device)
    bt = torch.from_numpy(bb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(net(at, bt).mean().item())


def regen_xl(pipe, img, n_steps, seed):
    """Symmetric DDIM regen at native size using SDXL img2img.
    strength = n_steps / NUM_INFERENCE_STEPS so exactly n_steps UNet
    forwards are run (last n_steps of the 1000-step schedule)."""
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt="",
        prompt_2="",
        image=img,
        strength=n_steps / NUM_INFERENCE_STEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    ).images[0]
    return out


def main():
    device = "cuda"
    print(f"loading SDXL ({MODEL}) ...", flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print("SDXL loaded", flush=True)

    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print("LPIPS loaded", flush=True)

    # Take the first N_PAIRS where: edit dims are multiples of 8 and
    # original is at least as large as edit on each axis.
    items = []
    skipped = 0
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            try:
                eW, eH = Image.open(os.path.join(ROOT, d["edit_path"])).size
                oW, oH = Image.open(os.path.join(ROOT, d["src_path"])).size
            except Exception:
                skipped += 1
                continue
            # Round edit to multiple of 8 (drop a few pixels at most)
            tW, tH = round8(eW), round8(eH)
            if tW < 512 or tH < 512:
                skipped += 1
                continue
            if oW < tW or oH < tH:
                skipped += 1
                continue
            d["_tW"], d["_tH"] = tW, tH
            items.append(d)
            if len(items) == N_PAIRS:
                break
    print(f"selected {len(items)} pairs (skipped {skipped})", flush=True)

    rows = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        tW, tH = item["_tW"], item["_tH"]
        orig_full = Image.open(os.path.join(ROOT, item["src_path"])).convert("RGB")
        edit_full = Image.open(os.path.join(ROOT, item["edit_path"])).convert("RGB")
        orig = center_crop(orig_full, tW, tH)
        edit = center_crop(edit_full, tW, tH)

        rec = {"slot": item["slot"], "edit_type": item.get("edit_type", ""),
               "size": [tW, tH]}
        for N in N_STEPS:
            r_orig = regen_xl(pipe, orig, n_steps=N, seed=SEED)
            r_edit = regen_xl(pipe, edit, n_steps=N, seed=SEED)
            rec[f"lpips_orig_N{N}"] = lpips_pil(orig, r_orig, net, device)
            rec[f"lpips_edit_N{N}"] = lpips_pil(edit, r_edit, net, device)
        rows.append(rec)

        if i % 2 == 0 or i == len(items):
            el = time.time() - t0
            rate = i / el
            print(f"  [{i:3d}/{len(items)}]  {el:6.0f}s  ETA {(len(items)-i)/rate/60:6.1f} min  "
                  f"last={tW}x{tH}", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)

    print(f"\n{'N':>4s}  {'mean LPIPS(orig)':>16s}  {'mean LPIPS(edit)':>16s}  "
          f"{'edit>orig %':>12s}  {'mean gap':>10s}")
    for N in N_STEPS:
        lo = np.array([r[f"lpips_orig_N{N}"] for r in rows])
        le = np.array([r[f"lpips_edit_N{N}"] for r in rows])
        print(f"  {N:>3d}  {lo.mean():>16.4f}  {le.mean():>16.4f}  "
              f"{100*(le > lo).mean():>11.1f}%  {(le - lo).mean():>+10.4f}")


if __name__ == "__main__":
    main()
