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
    root: str = "/home/trabbani/pico-banana-400k/sample_6k"
    metadata: str = "/home/trabbani/pico-banana-400k/sample_6k/metadata.jsonl"
    image_size: int = 256
    jpeg_q: int = 95
    photoreal_only: bool = True


class PicoBananaPaired(Dataset):
    """Returns dicts: {"orig": [3,H,W] in [-1,1], "edit": [3,H,W], "slot": int}."""

    def __init__(self, items: Iterable[dict], cfg: DatasetConfig):
        self.items = list(items)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        d = self.items[i]
        orig = _load_and_match_codec(
            os.path.join(self.cfg.root, d["src_path"]),
            self.cfg.image_size, self.cfg.jpeg_q,
        )
        edit = _load_and_match_codec(
            os.path.join(self.cfg.root, d["edit_path"]),
            self.cfg.image_size, self.cfg.jpeg_q,
        )
        return {
            "orig": to_tensor_neg1_1(orig),
            "edit": to_tensor_neg1_1(edit),
            "slot": d["slot"],
            "edit_type": d.get("edit_type", ""),
        }


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


# ----- v2: original v1 splits + photoreal_v2 download -> bigger train set -----

@dataclass
class DatasetConfigV2:
    """Combined dataset across the existing sample_6k (now also preprocessed
    to 384-JPEG-q95 with slot offset +100000) and the new photoreal_v2
    download. Test/val are inherited from v1 unchanged so accuracy numbers
    are directly comparable to the original C_eval (88.5% test_acc baseline)."""
    v2_root: str = "/mnt/data/pico-banana-400k/photoreal_v2"
    v2_metadata_download: str = "/mnt/data/pico-banana-400k/photoreal_v2/metadata.jsonl"
    v2_metadata_v1_resized: str = "/mnt/data/pico-banana-400k/photoreal_v2/v1_resized_metadata.jsonl"
    image_size: int = 256
    jpeg_q: int = 95


def _attach_root(items, root):
    for d in items:
        d["root"] = root
    return items


def make_splits_v2(cfg: DatasetConfigV2 | None = None):
    cfg = cfg or DatasetConfigV2()

    # Load v1-resized (the existing photoreal pairs preprocessed to 384 JPG)
    v1_resized = load_metadata(cfg.v2_metadata_v1_resized, photoreal_only=False)
    v1_resized = _attach_root(v1_resized, cfg.v2_root)

    # Recover the v1 train/val/test split: undo the +100000 slot offset and
    # apply split_by_slot with the same seed=17 the original v1 used.
    for d in v1_resized:
        d["v1_slot"] = d["slot"] - 100000
    # Sort by v1_slot so split_by_slot's deterministic shuffle matches v1.
    by_v1_slot = sorted(v1_resized, key=lambda d: d["v1_slot"])
    # split_by_slot uses d["slot"] for sorting; temporarily swap.
    for d in by_v1_slot:
        d["_orig_slot"] = d["slot"]
        d["slot"] = d["v1_slot"]
    v1_train, v1_val, v1_test = split_by_slot(by_v1_slot)
    for d in by_v1_slot:
        d["slot"] = d.pop("_orig_slot")
        del d["v1_slot"]

    # Load new photoreal_v2 downloads (all already photoreal)
    new_items = []
    if os.path.exists(cfg.v2_metadata_download):
        new_items = load_metadata(cfg.v2_metadata_download, photoreal_only=False)
        new_items = _attach_root(new_items, cfg.v2_root)
    # All new items go into train.
    train = v1_train + new_items
    return train, v1_val, v1_test, cfg


class PicoBananaPairedV2(Dataset):
    """Like PicoBananaPaired but reads images from per-item d['root']
    (which is set in make_splits_v2). Same matched-codec preprocessing."""

    def __init__(self, items, cfg: DatasetConfigV2):
        self.items = list(items)
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        orig = _load_and_match_codec(
            os.path.join(d["root"], d["src_path"]),
            self.cfg.image_size, self.cfg.jpeg_q,
        )
        edit = _load_and_match_codec(
            os.path.join(d["root"], d["edit_path"]),
            self.cfg.image_size, self.cfg.jpeg_q,
        )
        return {
            "orig": to_tensor_neg1_1(orig),
            "edit": to_tensor_neg1_1(edit),
            "slot": d["slot"],
            "edit_type": d.get("edit_type", ""),
        }


class PicoBananaSingleV2(Dataset):
    """Single-image (orig OR edit) variant of PicoBananaPairedV2."""

    def __init__(self, items, cfg: DatasetConfigV2):
        self.items = list(items)
        self.cfg = cfg
        self.idx = []
        for i in range(len(self.items)):
            self.idx.append((i, 0))
            self.idx.append((i, 1))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i, label = self.idx[k]
        d = self.items[i]
        rel = d["src_path"] if label == 0 else d["edit_path"]
        img = _load_and_match_codec(
            os.path.join(d["root"], rel),
            self.cfg.image_size, self.cfg.jpeg_q,
        )
        return {
            "img": to_tensor_neg1_1(img),
            "label": int(label),
            "slot": d["slot"],
        }


if __name__ == "__main__":
    cfg = DatasetConfig()
    train, val, test, _ = make_splits(cfg)
    print(f"photoreal pairs:  total={len(train)+len(val)+len(test)}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    ds = PicoBananaPaired(train[:4], cfg)
    for i in range(2):
        item = ds[i]
        print(f"  [{i}] orig={tuple(item['orig'].shape)}  edit={tuple(item['edit'].shape)}  "
              f"orig.range=[{item['orig'].min():.3f},{item['orig'].max():.3f}]  "
              f"slot={item['slot']}  edit_type={item['edit_type'][:40]}")
