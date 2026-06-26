"""Evaluate best_ood.pt on a directory of regen'd NB images. All inputs
are assumed to be label=1 (NB-origin), so detection rate = mean prediction
of class 1."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from data import _load_and_match_codec, to_tensor_neg1_1
from models import BinaryClassifier


class DirSingle(Dataset):
    def __init__(self, dir_, image_size=384, jpeg_q=95):
        self.dir = dir_
        self.paths = sorted(p for p in os.listdir(dir_) if p.endswith((".jpg", ".png")))
        self.image_size = image_size
        self.jpeg_q = jpeg_q

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        img = _load_and_match_codec(os.path.join(self.dir, self.paths[k]),
                                    self.image_size, self.jpeg_q)
        return {"img": to_tensor_neg1_1(img), "fname": self.paths[k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/trabbani/WAVES/detector/best_ood.pt")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BinaryClassifier(pretrained=False, backbone="resnet18").to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}", flush=True)

    ds = DirSingle(args.dir, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
    print(f"n={len(ds)} from {args.dir}", flush=True)

    pred_1 = 0
    confs = []
    misses = []
    with torch.no_grad():
        for batch in dl:
            x = batch["img"].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs > 0.5).long().cpu().numpy()
            for i, p in enumerate(probs.cpu().numpy()):
                confs.append(float(p))
                if pred[i] == 1:
                    pred_1 += 1
                else:
                    misses.append(batch["fname"][i])

    confs = np.array(confs)
    print(f"\nNB-detection rate: {pred_1}/{len(ds)}  = {pred_1/len(ds):.4f}")
    print(f"mean P(synthetic):  {confs.mean():.4f}")
    print(f"P>0.9 count:        {(confs > 0.9).sum()}")
    print(f"P<0.1 count:        {(confs < 0.1).sum()}")
    if misses:
        print(f"\nMissed ({len(misses)}):")
        for m in misses[:20]:
            idx = ds.paths.index(m)
            print(f"  {m}  P(synthetic)={confs[idx]:.4f}")


if __name__ == "__main__":
    main()
