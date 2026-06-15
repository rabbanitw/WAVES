"""CLI: batch diffusive-regen on a directory of images.

Usage:
  python -m regen --src DIR --dst DIR --n-steps N
  python -m regen --src DIR --dst DIR --noise-step N --asym
"""

import argparse
import glob
import os
import time

from PIL import Image

from . import build_pipeline, regen_asym, regen_symmetric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of source images")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--n-steps", type=int, default=10,
                    help="for symmetric: number of noise/denoise iterations")
    ap.add_argument("--noise-step", type=int, default=None,
                    help="for asymmetric: training-timestep noise level "
                         "(overrides --n-steps when --asym is set)")
    ap.add_argument("--asym", action="store_true",
                    help="use the WAVES-paper-style asymmetric variant "
                         "(50-step inference schedule, sparse denoising)")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ext", default="*.jpg",
                    help="glob pattern for source files (default *.jpg)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.src, args.ext)))
    if not paths:
        raise SystemExit(f"no {args.ext} in {args.src}")

    print(f"loading {args.model} on {args.device} ...")
    pipe = build_pipeline(args.model, args.device)

    n = args.noise_step if (args.asym and args.noise_step is not None) else args.n_steps
    mode = "asym" if args.asym else "sym"
    print(f"running {mode} regen at N={n} on {len(paths)} images -> {args.dst}")

    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = Image.open(p).convert("RGB").resize((512, 512))
        if args.asym:
            out = regen_asym(img, pipe, noise_step=n)
        else:
            out = regen_symmetric(img, pipe, n_steps=n)
        base = os.path.splitext(os.path.basename(p))[0]
        out.save(os.path.join(args.dst, base + ".png"))
        if i % 10 == 0 or i == len(paths):
            el = time.time() - t0
            print(f"  [{i:3d}/{len(paths)}] {el:5.0f}s")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
