"""DP-FedAvg memorization sweep on a long-tailed Tiny-ImageNet subset.

K in {1,2,4,6,8,10} clients. Per-client clip C, server Gaussian noise sigma on
the *averaged delta*. Tracks per-class train and test accuracy of the global
model, plus a per-client per-class train-acc heatmap.

Outputs JSON metrics + checkpoint per K to OUT_DIR.
"""

import argparse, json, math, os, random, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# ---------------------- dataset ----------------------

TINY_ROOT = Path("/mnt/data/tiny-imagenet/tiny-imagenet-200")

NORM_MEAN = (0.4802, 0.4481, 0.3975)
NORM_STD  = (0.2770, 0.2691, 0.2821)


class TinyImageNetSubset(Dataset):
    """Long-tailed subset built from a chosen list of wnids and per-class quotas."""
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def build_long_tailed(num_classes, head_count, tail_count, seed=0):
    """Pick `num_classes` wnids and assign each an exponentially decaying sample
    count from `head_count` down to `tail_count`. Returns
    (train_samples, test_samples, wnid_list, per_class_train_count).
    """
    rng = random.Random(seed)
    all_wnids = sorted([d.name for d in (TINY_ROOT / "train").iterdir() if d.is_dir()])
    rng.shuffle(all_wnids)
    wnids = all_wnids[:num_classes]
    wnid_to_label = {w: i for i, w in enumerate(wnids)}

    # exponential decay of training counts head -> tail
    if num_classes == 1:
        counts = [head_count]
    else:
        ratio = (tail_count / head_count) ** (1.0 / (num_classes - 1))
        counts = [max(1, int(round(head_count * (ratio ** i)))) for i in range(num_classes)]

    train_samples = []
    val_anno = (TINY_ROOT / "val" / "val_annotations.txt").read_text().splitlines()
    val_map = {}
    for line in val_anno:
        parts = line.split("\t")
        val_map[parts[0]] = parts[1]

    test_samples = []
    for w in wnids:
        wn_train_dir = TINY_ROOT / "train" / w / "images"
        files = sorted(wn_train_dir.iterdir())
        rng.shuffle(files)
        n = counts[wnid_to_label[w]]
        for f in files[:n]:
            train_samples.append((str(f), wnid_to_label[w]))

    val_dir = TINY_ROOT / "val" / "images"
    for fname, w in val_map.items():
        if w in wnid_to_label:
            test_samples.append((str(val_dir / fname), wnid_to_label[w]))

    return train_samples, test_samples, wnids, counts


def dirichlet_partition_synced(train_samples, test_samples, num_clients, alpha,
                                num_classes, seed=0):
    """Partition train and test indices with the *same* per-class Dirichlet
    proportions, so each client receives an i.i.d. (train_k, test_k) pair from
    the client-specific class distribution. Returns (client_train, client_test).
    """
    rng = np.random.default_rng(seed)
    train_by_class, test_by_class = defaultdict(list), defaultdict(list)
    for idx, (_, y) in enumerate(train_samples):
        train_by_class[y].append(idx)
    for idx, (_, y) in enumerate(test_samples):
        test_by_class[y].append(idx)
    client_train = [[] for _ in range(num_clients)]
    client_test = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        if num_clients == 1:
            client_train[0].extend(train_by_class[c])
            client_test[0].extend(test_by_class[c])
            continue
        proportions = rng.dirichlet([alpha] * num_clients)
        for samples_by_class, client_lists in (
            (train_by_class, client_train), (test_by_class, client_test)
        ):
            idxs = list(samples_by_class[c])
            if not idxs:
                continue
            rng.shuffle(idxs)
            cuts = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            chunks = np.split(np.array(idxs), cuts)
            for k, chunk in enumerate(chunks):
                client_lists[k].extend(chunk.tolist())
    for ci in client_train:
        rng.shuffle(ci)
    for ci in client_test:
        rng.shuffle(ci)
    return client_train, client_test


# ---------------------- model ----------------------

def _replace_bn_with_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            ng = num_groups if child.num_features % num_groups == 0 else 1
            setattr(module, name, nn.GroupNorm(ng, child.num_features))
        else:
            _replace_bn_with_gn(child, num_groups)


def make_model(num_classes):
    """ResNet-18 adapted for 64x64 with GroupNorm (BN running stats are not
    transferred through parameter-only flat copies, and GroupNorm is the
    standard normalization choice for FedAvg/DP-FedAvg)."""
    m = resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    _replace_bn_with_gn(m, num_groups=8)
    return m


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


# ---------------------- training ----------------------

def local_train(model, loader, lr, momentum, weight_decay, epochs, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()


@torch.no_grad()
def per_class_acc(model, loader, num_classes, device):
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    total = torch.zeros(num_classes, dtype=torch.long, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        for c in range(num_classes):
            mask = y == c
            total[c] += mask.sum()
            correct[c] += ((pred == y) & mask).sum()
    acc = correct.float() / total.clamp(min=1).float()
    return acc.cpu().numpy(), total.cpu().numpy()


def run_one(args, K, train_samples, test_samples, num_classes, counts, device, out_dir):
    seed = args.seed + K
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_ds = TinyImageNetSubset(train_samples, train_tf)
    train_eval_ds = TinyImageNetSubset(train_samples, eval_tf)
    test_ds = TinyImageNetSubset(test_samples, eval_tf)

    client_train_idx, client_test_idx = dirichlet_partition_synced(
        train_samples, test_samples, K, args.dirichlet_alpha, num_classes, seed=seed)
    print(f"[K={K}] client train sizes:", [len(c) for c in client_train_idx])
    print(f"[K={K}] client  test sizes:", [len(c) for c in client_test_idx])

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2,
                             pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False,
                                   num_workers=2, pin_memory=True)
    client_train_eval_loaders = [
        DataLoader(Subset(train_eval_ds, idx), batch_size=256, shuffle=False,
                   num_workers=1, pin_memory=True) if len(idx) > 0 else None
        for idx in client_train_idx
    ]
    client_test_loaders = [
        DataLoader(Subset(test_ds, idx), batch_size=256, shuffle=False,
                   num_workers=1, pin_memory=True) if len(idx) > 0 else None
        for idx in client_test_idx
    ]

    global_model = make_model(num_classes).to(device)
    history = {"K": K, "rounds": [],
               "test_acc_by_class": [], "train_acc_by_class": [],
               "client_train_acc_by_class_final": None,
               "client_test_acc_by_class_final": None,
               "counts": counts}

    for r in range(args.rounds):
        global_flat = flat_params(global_model).clone()
        agg_delta = torch.zeros_like(global_flat)
        active = 0
        for k in range(K):
            idx = client_train_idx[k]
            if len(idx) == 0:
                continue
            local_model = make_model(num_classes).to(device)
            set_flat_params(local_model, global_flat)
            loader = DataLoader(Subset(train_ds, idx), batch_size=args.batch_size,
                                shuffle=True, num_workers=1, pin_memory=True,
                                drop_last=False)
            local_train(local_model, loader, args.lr, args.momentum,
                        args.weight_decay, args.local_epochs, device)
            delta = flat_params(local_model) - global_flat
            # clip per-client delta to L2 norm C
            nrm = delta.norm()
            scale = (args.clip_C / nrm).clamp(max=1.0)
            delta.mul_(scale)
            agg_delta.add_(delta)
            active += 1
        if active == 0:
            continue
        agg_delta.div_(K)  # uniform average over K (matches theorem)
        if args.sigma > 0:
            agg_delta.add_(torch.randn_like(agg_delta) * args.sigma)
        new_flat = global_flat + agg_delta
        set_flat_params(global_model, new_flat)

        if (r + 1) % args.eval_every == 0 or r == args.rounds - 1:
            te_acc, te_tot = per_class_acc(global_model, test_loader, num_classes, device)
            tr_acc, tr_tot = per_class_acc(global_model, train_eval_loader, num_classes, device)
            history["rounds"].append(r + 1)
            history["test_acc_by_class"].append(te_acc.tolist())
            history["train_acc_by_class"].append(tr_acc.tolist())
            print(f"[K={K} r={r+1:3d}] train_acc_mean={tr_acc.mean():.3f}  "
                  f"test_acc_mean={te_acc.mean():.3f}  "
                  f"head5_train={tr_acc[:5].mean():.3f}  head5_test={te_acc[:5].mean():.3f}  "
                  f"tail5_train={tr_acc[-5:].mean():.3f}  tail5_test={te_acc[-5:].mean():.3f}")

    # final per-client per-class train and test accuracy of the *global* model
    def grid(loaders):
        out_grid, totals = [], []
        for ldr in loaders:
            if ldr is None:
                out_grid.append([float("nan")] * num_classes)
                totals.append([0] * num_classes)
                continue
            acc, tot = per_class_acc(global_model, ldr, num_classes, device)
            out_grid.append(acc.tolist())
            totals.append(tot.tolist())
        return out_grid, totals

    tr_grid, tr_totals = grid(client_train_eval_loaders)
    te_grid, te_totals = grid(client_test_loaders)
    history["client_train_acc_by_class_final"] = tr_grid
    history["client_test_acc_by_class_final"] = te_grid
    history["client_train_class_counts"] = tr_totals
    history["client_test_class_counts"] = te_totals
    history["client_train_sizes"] = [len(c) for c in client_train_idx]
    history["client_test_sizes"] = [len(c) for c in client_test_idx]

    out = out_dir / f"K{K:02d}.json"
    out.write_text(json.dumps(history))
    torch.save(global_model.state_dict(), out_dir / f"K{K:02d}_global.pt")
    print(f"[K={K}] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=50)
    ap.add_argument("--head-count", type=int, default=500)
    ap.add_argument("--tail-count", type=int, default=5)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--clip-C", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=3e-4)
    ap.add_argument("--Ks", type=str, default="1,2,4,6,8,10")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="/home/trabbani/fedavg_exp/runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, test_samples, wnids, counts = build_long_tailed(
        args.num_classes, args.head_count, args.tail_count, seed=args.seed)
    print(f"# classes={args.num_classes}, total_train={len(train_samples)}, "
          f"total_test={len(test_samples)}, head_count={counts[0]}, tail_count={counts[-1]}")

    cfg = {k: v for k, v in vars(args).items()}
    cfg["wnids"] = wnids
    cfg["counts"] = counts
    cfg["total_train"] = len(train_samples)
    cfg["total_test"] = len(test_samples)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    Ks = [int(k) for k in args.Ks.split(",") if k.strip()]
    t0 = time.time()
    for K in Ks:
        run_one(args, K, train_samples, test_samples, args.num_classes, counts,
                device, out_dir)
        print(f"  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
