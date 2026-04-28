"""WAVES-asymmetric diffusive regeneration sweep on one image.

For each noise_step value N in --steps, applies the WAVES recipe:
  add_noise at training timestep N (out of 1000), num_inference_steps=50,
  head_start_step = 50 - max(N // 20, 1) -> max(N // 20, 1) denoising iter(s)
  on a sparse stride-20 schedule. DDIM scheduler, empty prompt, guidance 7.5,
  seed 1024. SD 1.4 fp16.

Note: for N in [1, 10], max(N // 20, 1) = 1, so all sweep points do exactly
one denoising iteration; the only thing that varies is the amount of noise
injected.
"""

import argparse
import os

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
    ap.add_argument("--input", default="/home/trabbani/WAVES/valid_512/prompt_0_attempt_1_img_0_512x512.jpg")
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/dev_test/regen_sweep_asym_light")
    ap.add_argument("--steps", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = ReSDPipeline.from_pretrained(args.model, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)

    src = Image.open(args.input)
    print(f"input: {args.input}  size={src.size}")
    print(f"sweep: noise_step in {args.steps}")
    print(f"  scheduler: DDIM, 50-step inference; head_start_step = 50 - max(N // 20, 1)")
    print()

    for n in args.steps:
        out = regen_asym(src, pipe, noise_step=n)
        denoise_iters = max(n // 20, 1)
        out_path = os.path.join(args.out_dir, f"prompt_0_regen_asym_N{n:03d}.png")
        out.save(out_path)
        p = psnr(src, out)
        print(f"  N={n:3d}  denoise_iters={denoise_iters}  PSNR={p:5.2f} dB  -> {out_path}")


if __name__ == "__main__":
    main()
