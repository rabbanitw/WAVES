"""Time the N=10 symmetric regen on CPU for one image.

CPU runs use fp32 (PyTorch fp16 on CPU is incomplete and triggers
fallbacks/errors for several ops). Same pipeline + regen logic as the
GPU path, just torch_dtype=float32 and pipe.to('cpu').
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from diffusers import ReSDPipeline, DDIMScheduler


def regen_cpu(image: Image.Image, pipe, n_steps: int, seed: int = 1024) -> Image.Image:
    device = pipe.device
    dtype = pipe.unet.dtype  # fp32 on CPU
    g = torch.Generator(device=device).manual_seed(seed)

    arr = np.asarray(image.convert("RGB").resize((512, 512))) / 255.0
    arr = (arr - 0.5) * 2
    img_t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(img_t).latent_dist.sample(g)
    latents = latents * pipe.vae.config.scaling_factor

    noise = torch.randn(latents.shape, device=device, dtype=dtype, generator=g)
    nt = torch.tensor([n_steps - 1], dtype=torch.long, device=device)
    latents = pipe.scheduler.add_noise(latents, noise, nt).to(dtype)

    nis = pipe.scheduler.config.num_train_timesteps
    out = pipe(
        [""],
        head_start_latents=latents,
        head_start_step=nis - n_steps,
        num_inference_steps=nis,
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
    ap.add_argument("--input", default="/home/trabbani/WAVES/valid_512/prompt_0_attempt_1_img_0_512x512.jpg")
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/dev_test/regen_N010_cpu")
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"torch threads: {torch.get_num_threads()}  (PyTorch will use this many CPU threads)")

    t = time.time()
    pipe = ReSDPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to("cpu")
    t_load = time.time() - t
    print(f"pipeline load: {t_load:.1f}s")

    src = Image.open(args.input)
    print(f"input: {args.input}")
    print(f"running regen N={args.n_steps} on CPU ...")

    t = time.time()
    out = regen_cpu(src, pipe, n_steps=args.n_steps)
    t_regen = time.time() - t

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.out_dir, f"{base}_cpu.png")
    out.save(out_path)
    p = psnr(src, out)
    print(f"regen wall time: {t_regen:.1f}s  ({t_regen/args.n_steps:.1f}s per denoising iter)")
    print(f"PSNR vs source: {p:.2f} dB  -> {out_path}")


if __name__ == "__main__":
    main()
