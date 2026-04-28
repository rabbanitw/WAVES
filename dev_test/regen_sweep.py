"""Symmetric diffusive regeneration sweep on one image.

For each N in --steps, noise the image latent for N forward steps and then
denoise it for N reverse steps. Implementation:

* Use DDIMScheduler with num_inference_steps = num_train_timesteps = 1000,
  so scheduler.timesteps = [999, 998, ..., 0] — each inference step covers
  exactly one training timestep.
* Add noise via scheduler.add_noise at training timestep N-1 (so the latent
  is at the noise level the reverse process expects when entering at
  scheduler.timesteps[1000 - N] = N - 1).
* Call ReSDPipeline with num_inference_steps=1000 and head_start_step=1000-N,
  which runs the loop over scheduler.timesteps[1000-N:] = [N-1, N-2, ..., 0].
  That's exactly N denoising iterations.

This is a strength sweep that grows monotonically with N: small N noises only
the bottom of the schedule (light regen), large N noises further up (heavier).
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from diffusers import ReSDPipeline, DDIMScheduler


def regen_symmetric(image: Image.Image, pipe, n_steps: int, seed: int = 1024) -> Image.Image:
    device = pipe.device
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(seed)

    img = np.asarray(image.convert("RGB").resize((512, 512))) / 255.0
    img = (img - 0.5) * 2
    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(img_t).latent_dist.sample(generator)
    latents = latents * pipe.vae.config.scaling_factor

    noise = torch.randn(latents.shape, device=device, dtype=dtype, generator=generator)
    noise_t = torch.tensor([n_steps - 1], dtype=torch.long, device=device)
    latents = pipe.scheduler.add_noise(latents, noise, noise_t).to(dtype)

    num_inference_steps = pipe.scheduler.config.num_train_timesteps  # 1000
    head_start_step = num_inference_steps - n_steps

    out = pipe(
        [""],
        head_start_latents=latents,
        head_start_step=head_start_step,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        generator=generator,
    )
    return out[0][0]


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB").resize((512, 512))).astype(np.float32)
    y = np.asarray(b.convert("RGB")).astype(np.float32)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="valid_512/prompt_0_attempt_1_img_0_512x512.jpg")
    ap.add_argument("--out-dir", default="dev_test/regen_sweep")
    ap.add_argument("--steps", type=int, nargs="+", default=[20, 40, 60, 80, 100])
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
    print(f"sweep: N (denoising iterations) in {args.steps}")
    print(f"  scheduler: DDIM, 1000-step dense schedule (one training timestep per inference step)")
    print()

    for n in args.steps:
        out = regen_symmetric(src, pipe, n_steps=n)
        out_path = os.path.join(args.out_dir, f"prompt_0_regen_N{n:03d}.png")
        out.save(out_path)
        p = psnr(src, out)
        print(f"  N={n:3d}  PSNR={p:5.2f} dB  -> {out_path}")


if __name__ == "__main__":
    main()
