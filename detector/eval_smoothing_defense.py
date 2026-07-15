"""Randomized smoothing defense: at inference, add Gaussian noise sigma to the
input and average predictions over N samples. Cohen-Rosenfeld-Kolter (ICML'19).

The attacker can't precompute a perturbation that's robust across many random
noise realizations -- the attack must work in EXPECTATION over noise, which is
much harder. Sigma is in [-1,1] normalized space (i.e. sigma=0.05 corresponds
to ~6/255 pixel-space std).

Evaluates best_ood_balanced_r50.pt + best_adv_mixed_r18.pt against:
  - clean OOD-COCO edits + naturals
  - 5 levels of pre-generated adv OOD-COCO (eps in {1,2,4,8,16})
At sigma in {0.0, 0.025, 0.05, 0.10, 0.15, 0.25}, N samples averaged.
"""

import io
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from data import to_tensor_neg1_1
from models import BinaryClassifier

COCO_ROOT = Path("/mnt/data/coco_ood_v2_500")
IMAGE_SIZE = 384
EPS_LEVELS = [0, 1, 2, 4, 8, 16]
N_SAMPLES = 20

CKPTS = [
    ("noiseaug_s53_r18",
     "/home/trabbani/WAVES/detector/best_adv_mixed_noiseaug_r18.pt", "resnet18"),
]
# Focus on the matched-training sigma (0.025) since that's what the model
# was trained for, plus a couple of bracketing values.
SIGMAS = [0.0, 0.025, 0.05]


class CodecDataset(Dataset):
    def __init__(self, dir_, label):
        self.paths = sorted(os.listdir(dir_))
        self.dir = dir_
        self.label = label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        path = os.path.join(self.dir, self.paths[k])
        pil = Image.open(path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        # matched-codec roundtrip at q=95
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        return {"img": to_tensor_neg1_1(pil), "label": self.label}


@torch.no_grad()
def smoothed_eval(model, dl, device, sigma, n_samples):
    correct = total = 0
    for batch in dl:
        x = batch["img"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        B = x.size(0)
        if sigma == 0.0:
            probs = F.softmax(model(x), dim=1)
        else:
            # average softmax probs over n_samples Gaussian-noised copies
            probs_sum = torch.zeros(B, 2, device=device)
            for _ in range(n_samples):
                noise = torch.randn_like(x) * sigma
                x_noisy = (x + noise).clamp(-1.0, 1.0)
                probs_sum += F.softmax(model(x_noisy), dim=1)
            probs = probs_sum / n_samples
        pred = probs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def run_for_ckpt(name, ckpt_path, backbone, device):
    print(f"\n========================================================")
    print(f"  {name}  ({ckpt_path}, {backbone})")
    print(f"========================================================", flush=True)
    model = BinaryClassifier(pretrained=False, backbone=backbone).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])

    results = {}
    for sigma in SIGMAS:
        results[sigma] = {}
        n = 1 if sigma == 0.0 else N_SAMPLES
        print(f"\n  sigma={sigma:.3f}  N={n}", flush=True)
        for eps in EPS_LEVELS:
            if eps == 0:
                edit_dir = COCO_ROOT / "edit"
                nat_dir = COCO_ROOT / "orig"
            else:
                edit_dir = COCO_ROOT / f"edit_adv_eps{eps:03d}"
                nat_dir = COCO_ROOT / f"orig_adv_eps{eps:03d}"
            if not edit_dir.is_dir() or not nat_dir.is_dir():
                continue
            edit_ds = CodecDataset(str(edit_dir), label=1)
            nat_ds = CodecDataset(str(nat_dir), label=0)
            edit_dl = DataLoader(edit_ds, batch_size=32, shuffle=False,
                                 num_workers=4, pin_memory=True)
            nat_dl = DataLoader(nat_ds, batch_size=32, shuffle=False,
                                num_workers=4, pin_memory=True)
            edit_tpr = smoothed_eval(model, edit_dl, device, sigma, n)
            nat_tnr = smoothed_eval(model, nat_dl, device, sigma, n)
            bal = edit_tpr + nat_tnr - 1.0
            results[sigma][eps] = {"edit_tpr": edit_tpr,
                                   "nat_tnr": nat_tnr, "balanced": bal}
            tag = "clean" if eps == 0 else f"adv-eps{eps}"
            print(f"    {tag:>11s}  edit-TPR={edit_tpr:.4f}  nat-TNR={nat_tnr:.4f}  "
                  f"bal={bal:+.4f}", flush=True)
    return results


def main():
    device = "cuda"
    all_results = {}
    for name, ckpt_path, backbone in CKPTS:
        all_results[name] = run_for_ckpt(name, ckpt_path, backbone, device)

    out = Path("/home/trabbani/WAVES/detector/runs/smoothing_defense.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        # convert tuple keys to str
        serial = {}
        for name, r in all_results.items():
            serial[name] = {str(s): {str(e): v for e, v in d.items()}
                            for s, d in r.items()}
        json.dump(serial, f, indent=2)
    print(f"\nwrote {out}", flush=True)

    print("\n========== SUMMARY: balanced score by (sigma, eps) ==========")
    for name in all_results:
        print(f"\n=== {name} ===")
        print(f"{'sigma':>7s} | " + "  ".join(f"eps={e:<3d}" for e in EPS_LEVELS))
        for sigma in SIGMAS:
            cells = []
            for eps in EPS_LEVELS:
                if eps in all_results[name].get(sigma, {}):
                    cells.append(f"{all_results[name][sigma][eps]['balanced']:>+6.3f}")
                else:
                    cells.append("   -  ")
            print(f"{sigma:>7.3f} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
