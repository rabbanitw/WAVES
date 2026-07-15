"""Quick test: does the diffusion fingerprint have low-frequency content?

Apply a low-pass filter (Gaussian blur of varying sigma) to OOD-COCO edit +
natural inputs, then evaluate existing detectors. If clean OOD-COCO-edit
detection holds up at moderate blur, the fingerprint has enough low-freq
content to survive -- which means a frequency-domain (low-pass) classifier
should preserve detection while killing the bulk of pixel-PGD attack energy.

If detection collapses with blur, the fingerprint is concentrated in
high frequencies and a low-pass defense kills the signal too."""

import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from data import to_tensor_neg1_1
from models import BinaryClassifier

COCO_ROOT = Path("/mnt/data/coco_ood_v2_500")
IMAGE_SIZE = 384
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
TARGETS = [
    ("baseline_r50",     "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt",      "resnet50"),
    ("mixed_s48_r18",    "/home/trabbani/WAVES/detector/best_adv_mixed_r18.pt",         "resnet18"),
    ("noiseaug_s53_r18", "/home/trabbani/WAVES/detector/best_adv_mixed_noiseaug_r18.pt","resnet18"),
]
OUT = Path("/home/trabbani/WAVES/detector/runs/lowpass_input_eval.json")


class BlurredDataset(Dataset):
    def __init__(self, dir_, label, blur_sigma):
        self.paths = sorted(p for p in Path(dir_).iterdir())
        self.label = label
        self.sigma = blur_sigma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        pil = Image.open(self.paths[k]).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        # matched-codec roundtrip
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        if self.sigma > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=self.sigma))
        return {"img": to_tensor_neg1_1(pil), "label": self.label}


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


def run_one(name, ckpt_path, backbone, device):
    print(f"\n========================================================")
    print(f"  {name}  (backbone={backbone})")
    print(f"========================================================", flush=True)
    model = BinaryClassifier(pretrained=False, backbone=backbone).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])

    rows = {}
    for sigma in BLUR_SIGMAS:
        edit_ds = BlurredDataset(str(COCO_ROOT / "edit"), label=1, blur_sigma=sigma)
        nat_ds = BlurredDataset(str(COCO_ROOT / "orig"), label=0, blur_sigma=sigma)
        edit_dl = DataLoader(edit_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)
        nat_dl = DataLoader(nat_ds, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)
        edit_tpr = eval_set(model, edit_dl, device)
        nat_tnr = eval_set(model, nat_dl, device)
        bal = edit_tpr + nat_tnr - 1.0
        rows[sigma] = {"edit_tpr": edit_tpr, "nat_tnr": nat_tnr, "bal": bal}
        print(f"  sigma={sigma:.1f}  edit-TPR={edit_tpr:.4f}  nat-TNR={nat_tnr:.4f}  "
              f"bal={bal:+.4f}", flush=True)
    return rows


def main():
    device = "cuda"
    results = {}
    for name, ckpt_path, backbone in TARGETS:
        results[name] = run_one(name, ckpt_path, backbone, device)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    serial = {n: {str(s): v for s, v in r.items()} for n, r in results.items()}
    with open(OUT, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)

    print(f"\n========== SUMMARY: balanced score by (sigma) ==========")
    print(f"{'sigma':>6s} | " + "  ".join(f"{n:<22s}" for n, _, _ in TARGETS))
    for sigma in BLUR_SIGMAS:
        cells = []
        for name, _, _ in TARGETS:
            cells.append(f"{results[name][sigma]['bal']:>+6.3f} "
                         f"({results[name][sigma]['edit_tpr']:.2f}/"
                         f"{results[name][sigma]['nat_tnr']:.2f})")
        print(f"{sigma:>6.1f} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
