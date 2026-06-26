"""Iteratively rinse-regen a pair (orig, edit) at N=10 SDXL each round,
fresh seed per round. After each rinse, measure LPIPS and PSNR between
the *starting* image and the current rinse. Stop after MAX_RINSES or
when both metrics flatten."""

from __future__ import annotations

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

import lpips                                                          # noqa: E402
from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio                   # noqa: E402

META = "/mnt/data/pico-banana-400k/sample_6k/metadata.jsonl"
ROOT = "/mnt/data/pico-banana-400k/sample_6k"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
OUT = "/home/trabbani/WAVES/rinse_until_plateau.json"

N_STEPS_PER_RINSE = 10
NUM_INFERENCE_STEPS = 1000
SEED_BASE = 1024
GUIDANCE = 1.0
MAX_RINSES = 40


def round8(x):
    return int(x) - (int(x) % 8)


def center_crop(img, w, h):
    W, H = img.size
    return img.crop(((W - w) // 2, (H - h) // 2,
                     (W - w) // 2 + w, (H - h) // 2 + h))


def pil_chw01(img, device):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def regen_once(pipe, img, n_steps, seed):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt="", prompt_2="", image=img,
                strength=n_steps / NUM_INFERENCE_STEPS,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE,
                generator=gen).images[0]


def main():
    device = "cuda"
    print(f"loading SDXL ({MODEL}) ...", flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print("SDXL loaded", flush=True)

    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print("LPIPS loaded", flush=True)

    # First viable pair (same selection criterion as the metrics runs)
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            try:
                eW, eH = Image.open(os.path.join(ROOT, d["edit_path"])).size
                oW, oH = Image.open(os.path.join(ROOT, d["src_path"])).size
            except Exception:
                continue
            tW, tH = round8(eW), round8(eH)
            if tW >= 512 and tH >= 512 and oW >= tW and oH >= tH:
                d["_tW"], d["_tH"] = tW, tH
                break
        else:
            raise RuntimeError("no viable pair found in metadata")

    print(f"\nusing slot {d['slot']}, crop size {d['_tW']}x{d['_tH']}, "
          f"edit_type='{d.get('edit_type', '')}'", flush=True)
    orig = center_crop(
        Image.open(os.path.join(ROOT, d["src_path"])).convert("RGB"),
        d["_tW"], d["_tH"])
    edit = center_crop(
        Image.open(os.path.join(ROOT, d["edit_path"])).convert("RGB"),
        d["_tW"], d["_tH"])

    record = {"slot": d["slot"], "size": [d["_tW"], d["_tH"]],
              "edit_type": d.get("edit_type", ""),
              "n_steps_per_rinse": N_STEPS_PER_RINSE,
              "seed_base": SEED_BASE,
              "trajectories": {}}

    for side, start in [("orig", orig), ("edit", edit)]:
        print(f"\n=== {side} ===", flush=True)
        start01 = pil_chw01(start, device)
        current = start
        traj = []
        t0 = time.time()
        for k in range(1, MAX_RINSES + 1):
            current = regen_once(pipe, current, N_STEPS_PER_RINSE, SEED_BASE + k)
            cur01 = pil_chw01(current, device)
            with torch.no_grad():
                lp = float(net(start01 * 2 - 1, cur01 * 2 - 1).mean().item())
            A = np.asarray(start.convert("RGB"), dtype=np.uint8)
            B = np.asarray(current.convert("RGB"), dtype=np.uint8)
            psnr = float(peak_signal_noise_ratio(A, B, data_range=255))
            el = time.time() - t0
            print(f"  rinse {k:3d}  LPIPS={lp:.4f}  PSNR={psnr:.2f} dB  ({el:.0f}s)",
                  flush=True)
            traj.append({"rinse": k, "lpips": lp, "psnr": psnr})
        record["trajectories"][side] = traj

        # Detect plateau on this side's trajectory (post-hoc)
        lps = [t["lpips"] for t in traj]
        psnrs = [t["psnr"] for t in traj]
        # Plateau heuristic: first k where the last 5 LPIPS values span < 0.005
        def first_plateau(seq, win=5, eps=0.005):
            for i in range(win, len(seq) + 1):
                window = seq[i - win:i]
                if max(window) - min(window) < eps:
                    return i
            return None
        ki_l = first_plateau(lps, win=5, eps=0.005)
        ki_p = first_plateau(psnrs, win=5, eps=0.3)
        print(f"  -> LPIPS plateau at rinse {ki_l} (5-window span < 0.005)")
        print(f"  -> PSNR plateau at rinse  {ki_p} (5-window span < 0.3 dB)")
        record["trajectories"][side + "_plateau"] = {
            "lpips_first_plateau": ki_l,
            "psnr_first_plateau": ki_p,
        }

    with open(OUT, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
