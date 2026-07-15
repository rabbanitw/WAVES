"""Pre-generate adversarial copies of the OOD COCO eval sets at a fixed
ε ladder, attacking best_ood_balanced_r50.pt with the true label as
target (i.e., the perturbation flips the prediction). Saved as PNGs so
evaluation can later read them through the same matched-codec
preprocessing the detector trained on."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import _load_and_match_codec
from models import BinaryClassifier

SRC_EDIT = Path("/mnt/data/coco_ood_v2_500/edit")
SRC_NAT = Path("/mnt/data/coco_ood_v2_500/orig")
ATTACKER_CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
OUT_ROOT = Path("/mnt/data/coco_ood_v2_500")
EPS_LADDER = [1, 2, 4, 8, 16]   # /255
ALPHA = 1.0 / 255.0
K = 10
IMAGE_SIZE = 384


def pil_to_chw01(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def _resize_and_jpeg(pil_native):
    """Match-codec preprocessing: Lanczos to 384x384 + JPEG q=95 roundtrip
    so the attack lives in the same color/freq space as the deployed eval."""
    img = pil_native.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pgd(model, x, y, eps, alpha, k):
    """x in [-1,1] (already normalized for the model). True-label attack."""
    eps_b = eps * 2.0
    alpha_b = alpha * 2.0
    x_orig = x.detach()
    delta = torch.zeros_like(x_orig)
    for _ in range(k):
        delta.requires_grad_(True)
        x_pert = (x_orig + delta).clamp(-1.0, 1.0)
        logits = model(x_pert)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha_b * grad.sign()
            delta = delta.clamp(-eps_b, eps_b)
            delta = ((x_orig + delta).clamp(-1.0, 1.0) - x_orig)
    return (x_orig + delta).clamp(-1.0, 1.0).detach()


def attack_dir(model, src_dir, label, eps_units, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in os.listdir(src_dir) if p.endswith(".jpg"))
    eps = eps_units / 255.0
    n_flipped = 0
    for i, name in enumerate(files):
        # 1. matched-codec preprocess to 384x384 JPEG-q95-decoded
        pil384 = _load_and_match_codec(str(src_dir / name), IMAGE_SIZE, jpeg_q=95)
        x01 = pil_to_chw01(pil384).to(device)
        x_n11 = (x01 * 2.0 - 1.0)
        y = torch.tensor([label], device=device, dtype=torch.long)

        x_adv = pgd(model, x_n11, y, eps, ALPHA, K)
        # convert back to [0, 1] PNG
        x_adv01 = (x_adv + 1.0) / 2.0
        arr = (x_adv01[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        adv_pil = Image.fromarray(arr)
        adv_pil.save(out_dir / name.replace(".jpg", ".png"), "PNG", optimize=True)

        # Track whether attacker is now confused
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1).item()
        if pred != label:
            n_flipped += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] flipped_so_far={n_flipped}", flush=True)
    return {"n": len(files), "n_flipped_on_attacker": n_flipped}


def main():
    device = "cuda"
    print(f"loading attacker {ATTACKER_CKPT}...", flush=True)
    model = BinaryClassifier(pretrained=False, backbone="resnet50").to(device).eval()
    ckpt = torch.load(ATTACKER_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)

    results = {}
    for eps_units in EPS_LADDER:
        print(f"\n=== ε = {eps_units}/255 ===", flush=True)
        edit_out = OUT_ROOT / f"edit_adv_eps{eps_units:03d}"
        nat_out = OUT_ROOT / f"orig_adv_eps{eps_units:03d}"
        print(f"  edits -> {edit_out}", flush=True)
        edit_stats = attack_dir(model, SRC_EDIT, label=1, eps_units=eps_units,
                                out_dir=edit_out, device=device)
        print(f"  naturals -> {nat_out}", flush=True)
        nat_stats = attack_dir(model, SRC_NAT, label=0, eps_units=eps_units,
                               out_dir=nat_out, device=device)
        results[f"eps{eps_units:03d}"] = {
            "edit_flipped_pct": edit_stats["n_flipped_on_attacker"] / edit_stats["n"],
            "nat_flipped_pct": nat_stats["n_flipped_on_attacker"] / nat_stats["n"],
        }
        print(f"  flip rate on attacker: edits={results[f'eps{eps_units:03d}']['edit_flipped_pct']*100:.1f}%  "
              f"naturals={results[f'eps{eps_units:03d}']['nat_flipped_pct']*100:.1f}%", flush=True)

    print("\nDone.")
    for k, v in results.items():
        print(f"  {k}: edit flip = {v['edit_flipped_pct']*100:5.1f}%   "
              f"nat flip = {v['nat_flipped_pct']*100:5.1f}%")


if __name__ == "__main__":
    main()
