"""Diffusive-regeneration attacks against SynthID.

Public API:
  regen_symmetric(image, pipe, n_steps)   symmetric DDIM dense regen
                                          (noise N steps, denoise N steps)
  regen_asym(image, pipe, noise_step)     the asymmetric 50-step-schedule
                                          variant the WAVES paper actually
                                          shipped — fewer denoise iters
                                          than noise level implies
  build_pipeline(model_id, device)        load SD + DDIM scheduler ready
                                          for either attack
  ReSDPipeline                            the vendored SD subclass

CLI: `python -m regen --src DIR --dst DIR --n-steps N [--asym]`
"""

from .core import regen_symmetric, regen_asym, build_pipeline
from .resd_pipeline import ReSDPipeline

__all__ = ["regen_symmetric", "regen_asym", "build_pipeline", "ReSDPipeline"]
