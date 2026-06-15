"""Pico-Banana paired dataset for the watermark-removal GAN.

Yields (orig, edit) tensors at 256x256 in [-1, 1].

* Filters to the photoreal->photoreal subset of edit_types (~3447 pairs of
  the 5084 we have on disk).
* **Matched-codec preprocessing**: both the JPEG original and the PNG edit
  are re-encoded as JPEG q=95 in-memory before decoding, so the classifier
  cannot learn JPEG-vs-PNG compression noise as the dominant signal.
* Deterministic train/val/test split by slot index.
"""

from __future__ import annotations

import io
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_STYLIZED = frozenset({
    "Strong artistic style transfer (e.g., Van Gogh/anime/etc.)",
    "Photo to cartoon/sketch/comic",
    "Line-art ink sketch of the person",
    "Funko-Pop–style toy figure of the person",
    "Sticker-ify the person with bold outline and white border",
    "LEGO-minifigure rendition of the person",
    "Simpsonize the person (yellow-skin cartoon style)",
    "Convert person to 2D anime/manga style (identity-preserving)",
    "Convert person to Pixar/Disney-like 3D cartoon look",
    "Convert person to Western comic cel-shaded style",
    "Caricature with mild feature exaggeration (keep identity)",
})
_LIGHTLY_STYLIZED = frozenset({
    "Add film grain or vintage filter",
    "Modern ↔ historical style/look",
    "Apply seasonal transformation (summer ↔ winter)",
})


def is_photoreal(edit_type: str) -> bool:
    return edit_type not in _STYLIZED and edit_type not in _LIGHTLY_STYLIZED


def load_metadata(path: str, photoreal_only: bool = True) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if photoreal_only and not is_photoreal(d.get("edit_type", "")):
                continue
            out.append(d)
    return out


def split_by_slot(items: list[dict], val_n: int = 200, test_n: int = 300, seed: int = 17):
    rng = random.Random(seed)
    items = sorted(items, key=lambda d: d["slot"])
    shuffled = items[:]
    rng.shuffle(shuffled)
    test = shuffled[:test_n]
    val = shuffled[test_n: test_n + val_n]
    train = shuffled[test_n + val_n:]
    return train, val, test


def _load_and_match_codec(path: str, size: int = 256, jpeg_q: int = 95) -> Image.Image:
    """Load, resize square, and re-encode as JPEG q=95 to normalize codec noise."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def to_tensor_neg1_1(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@dataclass
class DatasetConfig:
    # Point at your local copy of Pico-Banana-400K SFT (see
    # https://github.com/apple/pico-banana-400k for the manifest +
    # download instructions). The dataloader expects `metadata.jsonl`
    # with `slot`, `src_path`, `edit_path`, `edit_type` keys; the
    # download_sample.py reference script in upstream WAVES produces
    # one in this exact layout.
    # Point at your local copy of Pico-Banana-400K SFT (see
    # https://github.com/apple/pico-banana-400k for the manifest +
    # download instructions). The dataloader expects metadata.jsonl
    # with `slot`, `src_path`, `edit_path`, `edit_type` keys.
    root: str = "/path/to/pico-banana-400k/sample"
    metadata: str = "/path/to/pico-banana-400k/sample/metadata.jsonl"
    image_size: int = 256
    jpeg_q: int = 95
    photoreal_only: bool = True



class PicoBananaSingle(Dataset):
    """Returns a single image (orig OR edit) with a binary label for C_eval.

    Label 0 = original Open Images, label 1 = Nano-Banana edit.
    """

    def __init__(self, items: Iterable[dict], cfg: DatasetConfig):
        self.items = list(items)
        self.cfg = cfg
        # interleaved: each pair contributes 2 examples
        self.idx = []
        for i in range(len(self.items)):
            self.idx.append((i, 0))
            self.idx.append((i, 1))

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int) -> dict:
        i, label = self.idx[k]
        d = self.items[i]
        rel = d["src_path"] if label == 0 else d["edit_path"]
        img = _load_and_match_codec(
            os.path.join(self.cfg.root, rel),
            self.cfg.image_size, self.cfg.jpeg_q,
        )
        return {
            "img": to_tensor_neg1_1(img),
            "label": int(label),
            "slot": d["slot"],
        }


# Convenience entrypoints
def make_splits(cfg: DatasetConfig | None = None):
    cfg = cfg or DatasetConfig()
    items = load_metadata(cfg.metadata, photoreal_only=cfg.photoreal_only)
    train, val, test = split_by_slot(items)
    return train, val, test, cfg


if __name__ == "__main__":
    cfg = DatasetConfig()
    train, val, test, _ = make_splits(cfg)
    print(f"photoreal pairs:  total={len(train)+len(val)+len(test)}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    ds = PicoBananaSingle(train[:2], cfg)
    for i in range(2):
        item = ds[i]
        print(f"  [{i}] img={tuple(item['img'].shape)}  "
              f"range=[{item['img'].min():.3f},{item['img'].max():.3f}]  "
              f"label={item['label']}  slot={item['slot']}")
