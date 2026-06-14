"""Pix2Pix-style watermark-removal GAN.

* Generator G:  residual U-Net.  G(x_edit) should look like an Open Images
  photo to D, while preserving the edit content per LPIPS(G(x_edit), x_edit).
* Discriminator D:  PatchGAN over Open Images (real, label 1) vs G output
  (fake, label 0).
* Loss:  L_adv (non-saturating) + lambda_preserve * LPIPS.

C_eval is NEVER used during training.
"""

from __future__ import annotations

import argparse
import os
import time

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import DatasetConfig, PicoBananaPaired, make_splits
from models import GeneratorUNet, PatchDiscriminator


def gan_loss_d(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    real_labels = torch.ones_like(d_real)
    fake_labels = torch.zeros_like(d_fake)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(d_real, real_labels)
        + F.binary_cross_entropy_with_logits(d_fake, fake_labels)
    )


def gan_loss_g(d_fake: torch.Tensor) -> torch.Tensor:
    real_labels = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, real_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/gan_v1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lambda-preserve", type=float, default=10.0,
                    help="weight on LPIPS(G(edit), edit)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="generator residual scale")
    ap.add_argument("--sample-every", type=int, default=200,
                    help="dump sample images every N steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DatasetConfig()
    train_items, val_items, test_items, _ = make_splits(cfg)
    print(f"splits: train={len(train_items)}  val={len(val_items)}  test={len(test_items)}")

    train_ds = PicoBananaPaired(train_items, cfg)
    val_ds = PicoBananaPaired(val_items[:32], cfg)  # small fixed sample for visuals

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    G = GeneratorUNet(alpha=args.alpha).to(device)
    D = PatchDiscriminator().to(device)
    lpips_net = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    opt_g = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    log_path = os.path.join(args.out_dir, "log.txt")
    log = open(log_path, "w")
    log.write("epoch\tstep\tloss_d\tloss_g_adv\tloss_g_lpips\tval_lpips\ttime\n")

    def save_samples(step: int, epoch: int):
        G.eval()
        with torch.no_grad():
            for batch in val_dl:
                edit = batch["edit"].to(device)
                gen = G(edit)
                # save first 4: edit, G(edit) side by side
                from torchvision.utils import save_image
                cat = torch.cat([edit[:4], gen[:4]], dim=0)
                cat = (cat + 1) / 2
                save_image(cat, os.path.join(args.out_dir, "samples", f"e{epoch:03d}_s{step:06d}.png"),
                           nrow=4)
                break
        G.train()

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            orig = batch["orig"].to(device, non_blocking=True)
            edit = batch["edit"].to(device, non_blocking=True)

            # ---- D step ----
            with torch.no_grad():
                fake = G(edit)
            d_real = D(orig)
            d_fake = D(fake)
            loss_d = gan_loss_d(d_real, d_fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- G step ----
            fake = G(edit)
            d_fake_for_g = D(fake)
            loss_g_adv = gan_loss_g(d_fake_for_g)
            loss_g_lpips = lpips_net(fake, edit).mean()
            loss_g = loss_g_adv + args.lambda_preserve * loss_g_lpips
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            if step % 20 == 0:
                el = time.time() - t0
                print(f"epoch {epoch:2d} step {step:5d}  "
                      f"loss_d={loss_d.item():.4f}  "
                      f"loss_g_adv={loss_g_adv.item():.4f}  "
                      f"loss_g_lpips={loss_g_lpips.item():.4f}  "
                      f"({el:.0f}s)", flush=True)
            step += 1

        # epoch end: val LPIPS + sample dump
        G.eval()
        with torch.no_grad():
            vp = []
            for vb in val_dl:
                e = vb["edit"].to(device)
                vp.append(lpips_net(G(e), e).mean().item())
            val_lpips = sum(vp) / len(vp)
        G.train()
        el = time.time() - t0
        print(f"--- epoch {epoch:2d} done   val_lpips_GxEdit_vs_edit={val_lpips:.4f}  ({el:.0f}s)", flush=True)
        log.write(f"{epoch}\t{step}\t{loss_d.item():.4f}\t{loss_g_adv.item():.4f}\t"
                  f"{loss_g_lpips.item():.4f}\t{val_lpips:.4f}\t{el:.0f}\n")
        log.flush()
        save_samples(step, epoch)
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch,
                    "args": vars(args)},
                   os.path.join(args.out_dir, f"ckpt_last.pt"))

    log.close()


if __name__ == "__main__":
    main()
