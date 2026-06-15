"""Symmetric and asymmetric diffusive-regeneration attacks.

Both attacks: encode the input to the SD VAE latent space, add scheduler
noise at some training timestep, run a few denoising steps starting from
that noised latent, decode back to pixels. The two variants differ in
the noise level and denoising schedule:

* `regen_symmetric` — runs the SD 1000-step DDIM schedule densely, so each
  inference step undoes exactly one training timestep. Noise the latent
  to training step N-1, then take N denoising iterations back to t=0.
  "Noise for N, denoise for N." This is the cleanest perturbation budget
  to reason about and the variant we report numbers against.

* `regen_asym` — the recipe shipped in the WAVES paper code (their Table 3
  numbers). 50 inference steps over the full 1000-step schedule (stride
  ~20), noise at training timestep `noise_step`, then only
  `max(noise_step // 20, 1)` denoising iterations. With noise_step=20
  you noise to training step 20 but only get 1 denoising step at the
  bottom of the schedule, leaving residual noise in the output. Included
  for cross-referencing against published WAVES numbers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

from .resd_pipeline import ReSDPipeline


def build_pipeline(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> ReSDPipeline:
    """Load Stable Diffusion + DDIM scheduler, ready for either attack."""
    pipe = ReSDPipeline.from_pretrained(model_id, torch_dtype=dtype, revision="fp16")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    return pipe.to(device)


def regen_symmetric(
    image: Image.Image,
    pipe: ReSDPipeline,
    n_steps: int,
    seed: int = 1024,
) -> Image.Image:
    """Symmetric DDIM dense regen: noise N steps, denoise N steps.

    Resizes to 512x512 (SD 1.x native), adds noise at training timestep
    N-1, then runs N denoising iterations through training timesteps
    [N-1, N-2, ..., 0].
    """
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
    t = torch.tensor([n_steps - 1], dtype=torch.long, device=device)
    latents = pipe.scheduler.add_noise(latents, noise, t).to(dtype)

    full = pipe.scheduler.config.num_train_timesteps  # 1000 for SD 1.x
    out = pipe(
        [""],
        head_start_latents=latents,
        head_start_step=full - n_steps,
        num_inference_steps=full,
        guidance_scale=7.5,
        generator=g,
    )
    return out[0][0]


def regen_asym(
    image: Image.Image,
    pipe: ReSDPipeline,
    noise_step: int,
    seed: int = 1024,
) -> Image.Image:
    """WAVES-paper-style asymmetric regen.

    50-step inference schedule (stride ~20 training timesteps), noise to
    training timestep `noise_step`, then `max(noise_step // 20, 1)`
    denoising iterations. Leaves residual noise on the output relative to
    the symmetric variant.
    """
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
    t = torch.tensor([noise_step], dtype=torch.long, device=device)
    latents = pipe.scheduler.add_noise(latents, noise, t).to(dtype)

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
