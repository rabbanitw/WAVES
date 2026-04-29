"""Eval for the v5 AdvGAN attack: how well does G transfer from C_train to C_eval?"""

import argparse
import json
import os

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import DatasetConfig, PicoBananaPaired, make_splits
from models import BinaryClassifier, FlatPerturbator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-eval-ckpt", default="/home/trabbani/WAVES/gan/runs/c_eval/best.pt")
    ap.add_argument("--c-train-ckpt", default="/home/trabbani/WAVES/gan/runs/c_train/best.pt")
    ap.add_argument("--gan-ckpt", default="/home/trabbani/WAVES/gan/runs/gan_v5/ckpt_last.pt")
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/gan_v5/eval")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DatasetConfig()
    _, _, test_items, _ = make_splits(cfg)
    test_ds = PicoBananaPaired(test_items, cfg)
    dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # Both classifiers (frozen, eval mode)
    C_eval = BinaryClassifier(pretrained=False).to(device)
    C_eval.load_state_dict(torch.load(args.c_eval_ckpt, map_location=device)["state_dict"])
    C_eval.eval()
    C_train = BinaryClassifier(pretrained=False).to(device)
    C_train.load_state_dict(torch.load(args.c_train_ckpt, map_location=device)["state_dict"])
    C_train.eval()

    # G with the args it was trained with
    gan_ckpt = torch.load(args.gan_ckpt, map_location=device)
    gan_args = gan_ckpt.get("args", {})
    G = FlatPerturbator(
        ngf=gan_args.get("ngf", 128),
        n_layers=gan_args.get("n_layers", 6),
        epsilon=gan_args.get("epsilon", 16.0 / 255.0),
    ).to(device)
    G.load_state_dict(gan_ckpt["G"])
    G.eval()
    print(f"G ckpt epoch={gan_ckpt.get('epoch', '?')}  args={gan_args}")

    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    correct_orig_eval = correct_edit_eval = 0
    correct_orig_train = correct_edit_train = 0
    flip_eval = flip_train = 0
    n = 0
    sum_lpips_gxedit_edit = 0.0
    sum_lpips_orig_edit = 0.0
    sum_linf = 0.0
    logit_orig_eval = []
    logit_edit_eval = []
    logit_gen_eval = []
    logit_gen_train = []

    with torch.no_grad():
        for batch in dl:
            orig = batch["orig"].to(device)
            edit = batch["edit"].to(device)
            gen = G(edit)

            # C_eval (the held-out judge)
            le_o = C_eval(orig)
            le_e = C_eval(edit)
            le_g = C_eval(gen)
            logit_orig_eval.append(le_o.cpu().numpy())
            logit_edit_eval.append(le_e.cpu().numpy())
            logit_gen_eval.append(le_g.cpu().numpy())
            correct_orig_eval += (le_o.argmax(1) == 0).sum().item()
            correct_edit_eval += (le_e.argmax(1) == 1).sum().item()
            flip_eval += (le_g.argmax(1) == 0).sum().item()

            # C_train (the surrogate G was trained against — sanity)
            lt_o = C_train(orig)
            lt_e = C_train(edit)
            lt_g = C_train(gen)
            logit_gen_train.append(lt_g.cpu().numpy())
            correct_orig_train += (lt_o.argmax(1) == 0).sum().item()
            correct_edit_train += (lt_e.argmax(1) == 1).sum().item()
            flip_train += (lt_g.argmax(1) == 0).sum().item()

            sum_lpips_gxedit_edit += lpips_net(gen, edit).sum().item()
            sum_lpips_orig_edit += lpips_net(orig, edit).sum().item()
            sum_linf += (gen - edit).abs().reshape(orig.size(0), -1).max(dim=1)[0].sum().item()
            n += orig.size(0)

    metrics = {
        "n_pairs": n,
        # C_eval (the headline judge)
        "c_eval_orig_acc": correct_orig_eval / n,
        "c_eval_edit_acc": correct_edit_eval / n,
        "c_eval_overall_acc": (correct_orig_eval + correct_edit_eval) / (2 * n),
        "c_eval_flip_rate": flip_eval / n,
        # C_train (sanity — should be ~1.0)
        "c_train_orig_acc": correct_orig_train / n,
        "c_train_edit_acc": correct_edit_train / n,
        "c_train_flip_rate": flip_train / n,
        # Perturbation magnitudes
        "lpips_GxEdit_vs_edit": sum_lpips_gxedit_edit / n,
        "lpips_orig_vs_edit": sum_lpips_orig_edit / n,
        "mean_linf": sum_linf / n,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(args.out_dir, "logits.npz"),
             orig_eval=np.concatenate(logit_orig_eval),
             edit_eval=np.concatenate(logit_edit_eval),
             gen_eval=np.concatenate(logit_gen_eval),
             gen_train=np.concatenate(logit_gen_train))

    print("\n=== v5 eval (G trained vs C_train, evaluated on C_eval) ===")
    print(f"  pairs evaluated:                {n}")
    print(f"  C_train flip rate (in-dist):    {metrics['c_train_flip_rate']:.4f}")
    print(f"  C_eval  flip rate (TRANSFER):   {metrics['c_eval_flip_rate']:.4f}   <-- HEADLINE")
    print(f"  C_eval baseline edit acc:       {metrics['c_eval_edit_acc']:.4f}")
    print(f"  LPIPS(G(edit), edit):           {metrics['lpips_GxEdit_vs_edit']:.4f}")
    print(f"  mean L_inf(G(edit) - edit):     {metrics['mean_linf']:.4f}")

    # Visual grid: orig | edit | G(edit) | 8x amplified delta
    with torch.no_grad():
        for batch in dl:
            orig = batch["orig"][:8].to(device); edit = batch["edit"][:8].to(device)
            gen = G(edit)
            delta = (gen - edit) * 8 * 0.5 + 0.5
            grid = torch.cat([orig, edit, gen, delta.clamp(0, 1) * 2 - 1], dim=0)
            grid = (grid + 1) / 2
            save_image(grid, os.path.join(args.out_dir, "grid.png"), nrow=8)
            break


if __name__ == "__main__":
    main()
