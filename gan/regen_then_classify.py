"""Test whether N-step diffusive regen disrupts the signal C_eval picks up.

For each test-split NB edit image:
  load -> resize 512 -> diffusive regen (symmetric, N steps) -> resize 256
  -> JPEG q=95 -> C_eval

Reports the fraction predicted as label=1 (= "this is a Nano-Banana edit").
Compare to the v5 eval baseline:
  c_eval_edit_acc on unmodified edits: 0.9400  (from runs/gan_v5/eval/metrics.json)

A drop = regen disrupts what C_eval was detecting.
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/trabbani/WAVES/gan")
sys.path.insert(0, "/home/trabbani/WAVES/dev_test")

from data import DatasetConfig, make_splits  # noqa: E402
from models import BinaryClassifier  # noqa: E402
from regen_sweep import regen_symmetric  # noqa: E402

from diffusers import ReSDPipeline, DDIMScheduler  # noqa: E402


def encode_match_codec(arr_neg1_1: np.ndarray, jpeg_q: int = 95) -> torch.Tensor:
    """arr in [-1, 1], H x W x 3 -> tensor [-1, 1] after JPEG q=95 round-trip."""
    img = Image.fromarray(((arr_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_q)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    out = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(out).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=300, help="number of test images to use")
    ap.add_argument("--c-eval-ckpt", default="/home/trabbani/WAVES/gan/runs/c_eval/best.pt")
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/regen_vs_c_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    # 1) Load C_eval (frozen)
    C_eval = BinaryClassifier(pretrained=False).to(device)
    sd = torch.load(args.c_eval_ckpt, map_location=device)
    C_eval.load_state_dict(sd["state_dict"])
    C_eval.eval()
    print(f"C_eval loaded (val_acc {sd.get('val_acc', '?')})")

    # 2) Load SD 1.4 + DDIM (same as our regen sweeps)
    pipe = ReSDPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16"
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    print(f"SD pipeline loaded (DDIM, fp16); regen N={args.n_steps}")

    # 3) Test split: same 300 pairs C_eval was eval'd on
    cfg = DatasetConfig()
    _, _, test_items, _ = make_splits(cfg)
    items = test_items[: args.limit]
    print(f"running on {len(items)} test edits...")

    # 4) Two paths per image:
    #      baseline:   PNG -> 256 -> JPEG q=95 -> C_eval         (matches training pipeline)
    #      regen:      PNG -> 512 -> regen_N -> 256 -> JPEG q=95 -> C_eval
    n_baseline_label1 = 0
    n_regen_label1 = 0
    n = 0
    t0 = time.time()
    rows = []  # for per-image CSV: slot, baseline_logit_diff, regen_logit_diff
    save_first_n = 4
    saved_grid = []

    with torch.no_grad():
        for i, item in enumerate(items):
            edit_path = os.path.join(cfg.root, item["edit_path"])
            img_pil = Image.open(edit_path).convert("RGB")

            # baseline: just resize to 256 and codec match
            arr256 = np.asarray(img_pil.resize((256, 256), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
            base_t = encode_match_codec(arr256).unsqueeze(0).to(device)
            base_logits = C_eval(base_t)
            base_pred = base_logits.argmax(dim=1).item()
            n_baseline_label1 += int(base_pred == 1)

            # regen: resize to 512, symmetric N-step DDIM regen, downsample to 256, codec match
            img512 = img_pil.resize((512, 512), Image.BICUBIC)
            regen_pil = regen_symmetric(img512, pipe, n_steps=args.n_steps)
            regen_arr = np.asarray(regen_pil.resize((256, 256), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
            regen_t = encode_match_codec(regen_arr).unsqueeze(0).to(device)
            regen_logits = C_eval(regen_t)
            regen_pred = regen_logits.argmax(dim=1).item()
            n_regen_label1 += int(regen_pred == 1)
            n += 1

            rows.append((item["slot"], base_pred, regen_pred,
                         base_logits[0].cpu().numpy().tolist(),
                         regen_logits[0].cpu().numpy().tolist()))

            if len(saved_grid) < save_first_n:
                saved_grid.append((base_t.squeeze(0).cpu(), regen_t.squeeze(0).cpu()))

            if (i + 1) % 20 == 0 or (i + 1) == len(items):
                el = time.time() - t0
                print(f"  [{i+1:3d}/{len(items)}] base_NBhit={n_baseline_label1}/{n} ({100*n_baseline_label1/n:.1f}%)  "
                      f"regen_NBhit={n_regen_label1}/{n} ({100*n_regen_label1/n:.1f}%)  "
                      f"({el:.0f}s)", flush=True)

    base_acc = n_baseline_label1 / n
    regen_acc = n_regen_label1 / n
    print(f"\n=== RESULT ===")
    print(f"  N (regen steps):                       {args.n_steps}")
    print(f"  test items:                            {n}")
    print(f"  baseline edit_acc (% predicted NB):    {base_acc:.4f}   ({n_baseline_label1}/{n})")
    print(f"  regen edit_acc   (% predicted NB):     {regen_acc:.4f}   ({n_regen_label1}/{n})")
    print(f"  delta:                                 {base_acc - regen_acc:+.4f}")

    # Save artifacts
    import json
    with open(os.path.join(args.out_dir, f"results_n{args.n_steps:03d}.json"), "w") as f:
        json.dump({
            "n_steps": args.n_steps,
            "n": n,
            "baseline_edit_acc": base_acc,
            "regen_edit_acc": regen_acc,
            "delta": base_acc - regen_acc,
        }, f, indent=2)

    # Save a small visual grid: base vs regen for the first few
    if saved_grid:
        from torchvision.utils import save_image
        grid = []
        for b, r in saved_grid:
            grid.append((b + 1) / 2)
            grid.append((r + 1) / 2)
        save_image(torch.stack(grid, dim=0),
                   os.path.join(args.out_dir, f"grid_n{args.n_steps:03d}.png"),
                   nrow=2)


if __name__ == "__main__":
    main()
