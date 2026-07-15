"""Test-time JPEG defense: re-encode the input at lower JPEG quality
before classifying. JPEG quantization is non-differentiable AND destroys
the high-frequency noise that PGD attacks live in. We expect:
  - clean accuracy: drops slightly with lower q
  - adv accuracy: recovers a lot (the perturbation is partly washed out)

Evaluates best_ood_balanced_r50.pt against:
  - clean OOD-COCO edits + naturals
  - 5 levels of pre-generated adv OOD-COCO (ε ∈ {1,2,4,8,16}, both sides)
At JPEG qualities q ∈ {95, 75, 50, 30, 10}."""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from data import to_tensor_neg1_1
from models import BinaryClassifier

CKPT = "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt"
COCO_ROOT = Path("/mnt/data/coco_ood_v2_500")
OUT = Path("/home/trabbani/WAVES/detector/runs/jpeg_defense_r50_balanced.json")

IMAGE_SIZE = 384
EPS_LEVELS = [0, 1, 2, 4, 8, 16]   # 0 = clean
JPEG_QUALITIES = [95, 75, 50, 30, 10]


def jpeg_roundtrip(pil, q):
    pil = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JPEGDefenseDataset(Dataset):
    """Reads images from a directory, applies the matched-codec preprocessing
    but at the SPECIFIED JPEG quality (not the training default of 95)."""

    def __init__(self, dir_, label, jpeg_q):
        self.paths = sorted(os.listdir(dir_))
        self.dir = dir_
        self.label = label
        self.q = jpeg_q

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        path = os.path.join(self.dir, self.paths[k])
        pil = Image.open(path)
        x = jpeg_roundtrip(pil, self.q)
        return {"img": to_tensor_neg1_1(x), "label": self.label,
                "fname": self.paths[k]}


@torch.no_grad()
def eval_set(model, dl, device):
    correct = total = 0
    for batch in dl:
        x = batch["img"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def main():
    device = "cuda"
    print(f"loading {CKPT} ...", flush=True)
    model = BinaryClassifier(pretrained=False, backbone="resnet50").to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device)["state_dict"])

    results = {}  # jpeg_q -> {eps -> {edit_tpr, nat_tnr, balanced}}
    for q in JPEG_QUALITIES:
        results[q] = {}
        print(f"\n========== JPEG q = {q} ==========", flush=True)
        for eps in EPS_LEVELS:
            if eps == 0:
                edit_dir = COCO_ROOT / "edit"
                nat_dir = COCO_ROOT / "orig"
            else:
                edit_dir = COCO_ROOT / f"edit_adv_eps{eps:03d}"
                nat_dir = COCO_ROOT / f"orig_adv_eps{eps:03d}"
            if not edit_dir.is_dir() or not nat_dir.is_dir():
                continue
            edit_ds = JPEGDefenseDataset(str(edit_dir), label=1, jpeg_q=q)
            nat_ds = JPEGDefenseDataset(str(nat_dir), label=0, jpeg_q=q)
            edit_dl = DataLoader(edit_ds, batch_size=64, shuffle=False,
                                 num_workers=4, pin_memory=True)
            nat_dl = DataLoader(nat_ds, batch_size=64, shuffle=False,
                                num_workers=4, pin_memory=True)
            edit_tpr = eval_set(model, edit_dl, device)
            nat_tnr = eval_set(model, nat_dl, device)
            bal = edit_tpr + nat_tnr - 1.0
            results[q][eps] = {"edit_tpr": edit_tpr, "nat_tnr": nat_tnr,
                               "balanced": bal}
            tag = "clean" if eps == 0 else f"adv-eps{eps}"
            print(f"  {tag:>11s}  edit-TPR={edit_tpr:.4f}  nat-TNR={nat_tnr:.4f}  "
                  f"bal={bal:+.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"ckpt": CKPT, "results_by_q": results}, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)

    print("\n========== SUMMARY: balanced score by (q, eps) ==========")
    print(f"{'q':>4s} | " + "  ".join(f"eps={e:<3d}" for e in EPS_LEVELS))
    for q in JPEG_QUALITIES:
        cells = []
        for eps in EPS_LEVELS:
            if eps in results[q]:
                cells.append(f"{results[q][eps]['balanced']:>+6.3f}")
            else:
                cells.append("   -  ")
        print(f"{q:>4d} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
