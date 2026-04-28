"""Smoke test for the WAVES diffusion-regeneration attack.

Uses `from diffusers import ReSDPipeline` (the same import line as
`regeneration/regen.py`) so we exercise the real, vendored ReSDPipeline
class. Mirrors `DiffWMAttacker.attack` but inlined so we don't need to
import the (currently broken) `regeneration` package.
"""

import argparse

import numpy as np
import torch
from PIL import Image
from diffusers import ReSDPipeline


def regen_diffusion(image: Image.Image, pipe, noise_step: int = 60, seed: int = 1024) -> Image.Image:
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)

    img = np.asarray(image.convert("RGB").resize((512, 512))) / 255.0
    img = (img - 0.5) * 2
    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float16)

    latents = pipe.vae.encode(img_t).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    noise = torch.randn(latents.shape, device=device, dtype=torch.float16, generator=generator)
    timestep = torch.tensor([noise_step], dtype=torch.long, device=device)
    latents = pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)

    head_start_step = 50 - max(noise_step // 20, 1)
    out = pipe(
        [""],
        head_start_latents=latents,
        head_start_step=head_start_step,
        guidance_scale=7.5,
        generator=generator,
    )
    return out[0][0]  # (images, nsfw) -> first image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--noise-step", type=int, default=60)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, model={args.model}, noise_step={args.noise_step}")
    print(f"ReSDPipeline class: {ReSDPipeline}")

    pipe = ReSDPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, revision="fp16"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)

    img = Image.open(args.input)
    print(f"input: {args.input}  size={img.size}  mode={img.mode}")
    out = regen_diffusion(img, pipe, noise_step=args.noise_step)
    out.save(args.output)
    print(f"wrote: {args.output}  size={out.size}")


if __name__ == "__main__":
    main()
