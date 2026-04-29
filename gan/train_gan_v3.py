"""GAN v3: classical-adversarial-style perturbator.

What's different from v1/v2:
* G is a FlatPerturbator: 6 conv layers, NO downsampling, L_inf-bounded
  output `clamp(x + epsilon * tanh(net(x)), -1, 1)`. epsilon=16/255 by
  default — every pixel can change by at most ~6% of full color range.
  This rules out the "localized black hole" failure mode v1 exhibited
  (U-Net's spatial bottleneck encouraged regional edits).
* Dual D (PatchGAN + Global), same as v2.
* lambda_preserve=1.0 (light, since epsilon already constrains the change).

C_eval is still frozen and unused during training.
"""

from __future__ import annotations

import argparse
import os
import time

import lpips
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import DatasetConfig, PicoBananaPaired, make_splits
from models import FlatPerturbator, PatchDiscriminator, GlobalDiscriminator


def bce_real(d_out): return F.binary_cross_entropy_with_logits(d_out, torch.ones_like(d_out))
def bce_fake(d_out): return F.binary_cross_entropy_with_logits(d_out, torch.zeros_like(d_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/gan_v3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lambda-preserve", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=16.0 / 255.0,
                    help="L_inf perturbation budget in [-1,1] units")
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=6)
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
    val_ds = PicoBananaPaired(val_items[:32], cfg)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    G = FlatPerturbator(ngf=args.ngf, n_layers=args.n_layers, epsilon=args.epsilon).to(device)
    Dp = PatchDiscriminator().to(device)
    Dg = GlobalDiscriminator().to(device)
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device)
    for p in lpips_net.parameters(): p.requires_grad_(False)

    n_params = sum(p.numel() for p in G.parameters()) / 1e6
    print(f"G FlatPerturbator: {n_params:.2f}M params, epsilon={args.epsilon:.4f} "
          f"(~{args.epsilon * 255:.1f}/255)")

    opt_g = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_dp = Adam(Dp.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_dg = Adam(Dg.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    log = open(os.path.join(args.out_dir, "log.txt"), "w")
    log.write("epoch\tstep\tloss_dp\tloss_dg\tloss_g_adv_p\tloss_g_adv_g\tloss_g_lpips\t"
              "val_lpips\tval_linf\ttime\n")

    def save_samples(step, epoch):
        G.eval()
        with torch.no_grad():
            for batch in val_dl:
                edit = batch["edit"].to(device)
                gen = G(edit)
                # 8x amplified delta for visibility
                delta_vis = ((gen - edit) / args.epsilon * 0.5 + 0.5).clamp(0, 1)
                from torchvision.utils import save_image
                cat = torch.cat([(edit[:4] + 1) / 2, (gen[:4] + 1) / 2, delta_vis[:4]], dim=0)
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

            # ---- D step (both Ds) ----
            with torch.no_grad():
                fake = G(edit)
            dp_real = Dp(orig); dp_fake = Dp(fake)
            loss_dp = 0.5 * (bce_real(dp_real) + bce_fake(dp_fake))
            opt_dp.zero_grad(set_to_none=True); loss_dp.backward(); opt_dp.step()
            dg_real = Dg(orig); dg_fake = Dg(fake)
            loss_dg = 0.5 * (bce_real(dg_real) + bce_fake(dg_fake))
            opt_dg.zero_grad(set_to_none=True); loss_dg.backward(); opt_dg.step()

            # ---- G step ----
            fake = G(edit)
            dp_fake = Dp(fake)
            dg_fake = Dg(fake)
            loss_g_adv_p = bce_real(dp_fake)
            loss_g_adv_g = bce_real(dg_fake)
            loss_g_adv = 0.5 * (loss_g_adv_p + loss_g_adv_g)
            loss_g_lpips = lpips_net(fake, edit).mean()
            loss_g = loss_g_adv + args.lambda_preserve * loss_g_lpips
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()

            if step % 20 == 0:
                el = time.time() - t0
                print(f"epoch {epoch:2d} step {step:5d}  "
                      f"dp={loss_dp.item():.3f} dg={loss_dg.item():.3f} "
                      f"g_adv_p={loss_g_adv_p.item():.3f} g_adv_g={loss_g_adv_g.item():.3f} "
                      f"g_lpips={loss_g_lpips.item():.3f}  ({el:.0f}s)", flush=True)
            step += 1

        # epoch end
        G.eval()
        with torch.no_grad():
            vp, vli = [], []
            for vb in val_dl:
                e = vb["edit"].to(device)
                g = G(e)
                vp.append(lpips_net(g, e).mean().item())
                vli.append((g - e).abs().max().item())
            val_lpips = sum(vp) / len(vp)
            val_linf = max(vli)
        G.train()
        el = time.time() - t0
        print(f"--- epoch {epoch:2d} done   val_lpips={val_lpips:.4f}  "
              f"val_linf={val_linf:.4f} (budget={args.epsilon:.4f})  ({el:.0f}s)", flush=True)
        log.write(f"{epoch}\t{step}\t{loss_dp.item():.4f}\t{loss_dg.item():.4f}\t"
                  f"{loss_g_adv_p.item():.4f}\t{loss_g_adv_g.item():.4f}\t"
                  f"{loss_g_lpips.item():.4f}\t{val_lpips:.4f}\t{val_linf:.4f}\t{el:.0f}\n")
        log.flush()
        save_samples(step, epoch)
        torch.save({"G": G.state_dict(), "Dp": Dp.state_dict(), "Dg": Dg.state_dict(),
                    "epoch": epoch, "args": vars(args)},
                   os.path.join(args.out_dir, "ckpt_last.pt"))

    log.close()


if __name__ == "__main__":
    main()
