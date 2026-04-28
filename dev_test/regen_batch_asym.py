"""Batch *asymmetric* (WAVES-style) diffusive regeneration.

Reproduces regeneration/regen.py::DiffWMAttacker.attack faithfully:

* num_inference_steps = 50 (default 50-step inference schedule)
* pipe.scheduler.add_noise(latents, noise, timestep=N)  -- noise at training
  timestep N out of 1000
* head_start_step = 50 - max(N // 20, 1)  -- at most a handful of denoising
  iterations on a sparse stride-20 schedule

DDIMScheduler (replacing the default PNDM) so a few-step partial denoising is
well-defined regardless of the skipped warmup.

For each input image, writes one PNG to --out-dir.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from diffusers import ReSDPipeline, DDIMScheduler


def regen_asym(image: Image.Image, pipe, noise_step: int, seed: int = 1024) -> Image.Image:
    device = pipe.device
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(seed)

    arr = np.asarray(image.convert("RGB").resize((512, 512))) / 255.0
    arr = (arr - 0.5) * 2
    img_t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(img_t).latent_dist.sample(g)
    latents = latents * pipe.vae.config.scaling_factor

    noise = torch.randn(latents.shape, device=device, dtype=dtype, generator=g)
    nt = torch.tensor([noise_step], dtype=torch.long, device=device)
    latents = pipe.scheduler.add_noise(latents, noise, nt).to(dtype)

    head_start_step = 50 - max(noise_step // 20, 1)
    out = pipe(
        [""],
        head_start_latents=latents,
        head_start_step=head_start_step,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=g,
    )
    return out[0][0]


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB").resize((512, 512))).astype(np.float32)
    y = np.asarray(b.convert("RGB")).astype(np.float32)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/home/trabbani/WAVES/valid_512")
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/dev_test/regen_N020_asym")
    ap.add_argument("--noise-step", type=int, default=20, help="WAVES noise_step / strength")
    ap.add_argument("--ext", default="*.jpg")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = sorted(glob.glob(os.path.join(args.in_dir, args.ext)))
    if not inputs:
        raise SystemExit(f"No {args.ext} in {args.in_dir}")

    denoise_iters = max(args.noise_step // 20, 1)
    print(f"input:        {args.in_dir}  ({len(inputs)} files)")
    print(f"output:       {args.out_dir}")
    print(f"noise_step:   {args.noise_step}  (-> {denoise_iters} denoising iter(s) at head_start_step={50 - denoise_iters})")
    print(f"model:        {args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = ReSDPipeline.from_pretrained(args.model, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    print(f"pipeline loaded on {device}")

    psnrs = []
    skipped = 0
    t0 = time.time()
    for i, in_path in enumerate(inputs, 1):
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(args.out_dir, base + ".png")
        if os.path.exists(out_path):
            skipped += 1
            continue
        src = Image.open(in_path)
        out = regen_asym(src, pipe, noise_step=args.noise_step)
        out.save(out_path)
        p = psnr(src, out)
        psnrs.append(p)
        elapsed = time.time() - t0
        done = i - skipped
        rate = done / max(elapsed, 1e-9)
        eta = (len(inputs) - i) / max(rate, 1e-9)
        print(f"[{i:3d}/{len(inputs)}] PSNR={p:5.2f} dB  {base}  ({elapsed:5.1f}s, ETA {eta:5.0f}s)")

    print(f"done in {time.time() - t0:.1f}s; processed {len(psnrs)}, skipped {skipped}")
    if psnrs:
        print(f"PSNR  mean={np.mean(psnrs):.2f}  min={np.min(psnrs):.2f}  max={np.max(psnrs):.2f}  std={np.std(psnrs):.2f}")


if __name__ == "__main__":
    main()
