"""GAN evaluation: label-flipping rate via the held-out C_eval classifier.

Loads:
  * gan/runs/c_eval/best.pt   -> C_eval ResNet-18 binary classifier
  * gan/runs/<gan_run>/ckpt_last.pt -> trained Generator G

Reports on the test split:
  * Baseline accuracy of C_eval on (orig, edit) pairs
  * Label-flip rate: fraction of G(edit) predicted as label 0 (= "original")
  * Edit-fidelity LPIPS: mean LPIPS(G(edit), edit)
  * C_eval logit histograms before vs after G
"""

from __future__ import annotations

import argparse
import json
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import DatasetConfig, PicoBananaPaired, make_splits
from models import BinaryClassifier, GeneratorUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-eval-ckpt", default="/home/trabbani/WAVES/gan/runs/c_eval/best.pt")
    ap.add_argument("--gan-ckpt", default="/home/trabbani/WAVES/gan/runs/gan_v1/ckpt_last.pt")
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/gan_v1/eval")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DatasetConfig()
    _, _, test_items, _ = make_splits(cfg)
    test_ds = PicoBananaPaired(test_items, cfg)
    dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # C_eval
    C = BinaryClassifier(pretrained=False).to(device)
    C.load_state_dict(torch.load(args.c_eval_ckpt, map_location=device)["state_dict"])
    C.eval()

    # G
    G = GeneratorUNet().to(device)
    G.load_state_dict(torch.load(args.gan_ckpt, map_location=device)["G"])
    G.eval()

    lpips_net = lpips.LPIPS(net="vgg").to(device).eval()

    correct_orig = 0
    correct_edit = 0
    flipped = 0
    n_pairs = 0
    sum_lpips_gen_vs_edit = 0.0
    sum_lpips_orig_vs_edit = 0.0
    logit_orig = []
    logit_edit = []
    logit_gen = []

    with torch.no_grad():
        for batch in dl:
            orig = batch["orig"].to(device)
            edit = batch["edit"].to(device)
            gen = G(edit)

            log_o = C(orig)
            log_e = C(edit)
            log_g = C(gen)

            logit_orig.append(log_o.cpu().numpy())
            logit_edit.append(log_e.cpu().numpy())
            logit_gen.append(log_g.cpu().numpy())

            correct_orig += (log_o.argmax(dim=1) == 0).sum().item()
            correct_edit += (log_e.argmax(dim=1) == 1).sum().item()
            flipped += (log_g.argmax(dim=1) == 0).sum().item()
            n_pairs += orig.size(0)

            sum_lpips_gen_vs_edit += lpips_net(gen, edit).sum().item()
            sum_lpips_orig_vs_edit += lpips_net(orig, edit).sum().item()

    logit_orig = np.concatenate(logit_orig, axis=0)
    logit_edit = np.concatenate(logit_edit, axis=0)
    logit_gen = np.concatenate(logit_gen, axis=0)

    metrics = {
        "n_pairs": n_pairs,
        "c_eval_orig_acc": correct_orig / n_pairs,
        "c_eval_edit_acc": correct_edit / n_pairs,
        "c_eval_overall_acc": (correct_orig + correct_edit) / (2 * n_pairs),
        "label_flip_rate": flipped / n_pairs,
        "lpips_GxEdit_vs_edit": sum_lpips_gen_vs_edit / n_pairs,
        "lpips_orig_vs_edit": sum_lpips_orig_vs_edit / n_pairs,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(args.out_dir, "logits.npz"),
             orig=logit_orig, edit=logit_edit, gen=logit_gen)

    print("=== Eval results ===")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")

    # Save a small visual grid
    with torch.no_grad():
        for batch in dl:
            orig = batch["orig"][:8].to(device)
            edit = batch["edit"][:8].to(device)
            gen = G(edit)
            grid = torch.cat([orig, edit, gen], dim=0)
            grid = (grid + 1) / 2
            save_image(grid, os.path.join(args.out_dir, "grid.png"), nrow=8)
            break


if __name__ == "__main__":
    main()
