"""GAN v5: AdvGAN-style — train G against a FROZEN pretrained classifier.

This abandons the standard-GAN pattern (which kept failing because our
from-scratch D's couldn't learn the binary task) for the adversarial-
example pattern that actually has a working precedent in this regime
(WAVES's own AdvCls attack):

* C_train  is a frozen ResNet-18 (ImageNet-pretrained, fine-tuned on the
  binary task with seed=99). Loaded from gan/runs/c_train/best.pt.
* G        is the FlatPerturbator (L_inf bounded, 0.6M params).
* Loss:    L_G = CE(C_train(G(edit)), label=0)  +  lambda * LPIPS(G(edit), edit)
           (G wants C_train to predict label 0 = "this is real")

C_eval (seed=42) remains the held-out evaluator. It is never used in
training, so a high label-flip rate on C_eval after this run = the
adversarial perturbation transfers across initializations of the same
architecture, which is the meaningful claim.
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
from models import FlatPerturbator, BinaryClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/trabbani/WAVES/gan/runs/gan_v5")
    ap.add_argument("--c-train-ckpt", default="/home/trabbani/WAVES/gan/runs/c_train/best.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lambda-preserve", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    ap.add_argument("--ngf", type=int, default=128)
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
    C_train = BinaryClassifier(pretrained=False).to(device)
    sd = torch.load(args.c_train_ckpt, map_location=device)
    C_train.load_state_dict(sd["state_dict"])
    C_train.eval()
    for p in C_train.parameters():
        p.requires_grad_(False)

    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device)
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    n_g = sum(p.numel() for p in G.parameters()) / 1e6
    print(f"G FlatPerturbator: {n_g:.2f}M params, ngf={args.ngf}, "
          f"epsilon={args.epsilon:.4f} (~{args.epsilon * 255:.1f}/255)")
    print(f"C_train loaded from {args.c_train_ckpt} (val_acc={sd.get('val_acc', '?')})  FROZEN")

    opt_g = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    log = open(os.path.join(args.out_dir, "log.txt"), "w")
    log.write("epoch\tstep\tloss_g_adv\tloss_g_lpips\tval_lpips\tval_linf\t"
              "val_flip_rate\ttime\n")

    def save_samples(step, epoch):
        G.eval()
        with torch.no_grad():
            for batch in val_dl:
                edit = batch["edit"].to(device)
                gen = G(edit)
                delta_vis = ((gen - edit) / args.epsilon * 0.5 + 0.5).clamp(0, 1)
                from torchvision.utils import save_image
                cat = torch.cat([(edit[:4] + 1) / 2, (gen[:4] + 1) / 2, delta_vis[:4]], dim=0)
                save_image(cat, os.path.join(args.out_dir, "samples", f"e{epoch:03d}_s{step:06d}.png"),
                           nrow=4)
                break
        G.train()

    print("\n=== AdvGAN training (G vs frozen C_train) ===")
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            edit = batch["edit"].to(device, non_blocking=True)
            target_label = torch.zeros(edit.size(0), dtype=torch.long, device=device)

            fake = G(edit)
            logits = C_train(fake)
            loss_g_adv = F.cross_entropy(logits, target_label)
            loss_g_lpips = lpips_net(fake, edit).mean()
            loss_g = loss_g_adv + args.lambda_preserve * loss_g_lpips

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            if step % 20 == 0:
                el = time.time() - t0
                # quick flip-rate snapshot on this batch
                with torch.no_grad():
                    flip = (logits.argmax(dim=1) == 0).float().mean().item()
                print(f"epoch {epoch:2d} step {step:5d}  "
                      f"adv={loss_g_adv.item():.4f} lpips={loss_g_lpips.item():.4f} "
                      f"flip(train)={flip:.3f}  ({el:.0f}s)", flush=True)
            step += 1

        # epoch end: val LPIPS, val L_inf, val flip rate (vs C_train, in-distribution)
        G.eval()
        with torch.no_grad():
            vp, vli, vflip, n_v = [], [], 0, 0
            for vb in val_dl:
                e = vb["edit"].to(device); g = G(e)
                vp.append(lpips_net(g, e).mean().item())
                vli.append((g - e).abs().max().item())
                vflip += (C_train(g).argmax(dim=1) == 0).sum().item()
                n_v += e.size(0)
            val_lpips = sum(vp) / len(vp); val_linf = max(vli)
            val_flip = vflip / n_v
        G.train()
        el = time.time() - t0
        print(f"--- epoch {epoch:2d} done   val_lpips={val_lpips:.4f}  val_linf={val_linf:.4f}  "
              f"val_flip_C_train={val_flip:.4f}  ({el:.0f}s)", flush=True)
        log.write(f"{epoch}\t{step}\t{loss_g_adv.item():.4f}\t{loss_g_lpips.item():.4f}\t"
                  f"{val_lpips:.4f}\t{val_linf:.4f}\t{val_flip:.4f}\t{el:.0f}\n")
        log.flush()
        save_samples(step, epoch)
        torch.save({"G": G.state_dict(), "epoch": epoch, "args": vars(args)},
                   os.path.join(args.out_dir, "ckpt_last.pt"))

    log.close()


if __name__ == "__main__":
    main()
