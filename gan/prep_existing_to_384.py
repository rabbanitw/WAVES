"""Preprocess the existing 3447 photoreal pairs (from sample_6k, raw 1024px)
into the same 384-JPEG-q95 format as the photoreal_v2 download.

Writes to photoreal_v2/originals/ and photoreal_v2/edits/ with slot offset
+100000 to avoid collision with the 0..34999 download slots. Metadata goes
to a separate file (v1_resized_metadata.jsonl) so it doesn't race with the
in-progress download appending to metadata.jsonl.
"""

import io
import json
import os
import sys

from PIL import Image

V1_ROOT = "/home/trabbani/pico-banana-400k/sample_6k"
V1_META = "/home/trabbani/pico-banana-400k/sample_6k/metadata.jsonl"
V2_ROOT = "/mnt/data/pico-banana-400k/photoreal_v2"
NEW_META = "/mnt/data/pico-banana-400k/photoreal_v2/v1_resized_metadata.jsonl"

SLOT_OFFSET = 100000
SIZE = 384
JPEG_Q = 95

_STYLIZED = {
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
}
_LIGHTLY_STYLIZED = {
    "Add film grain or vintage filter",
    "Modern ↔ historical style/look",
    "Apply seasonal transformation (summer ↔ winter)",
}


def is_photoreal(edit_type: str) -> bool:
    return edit_type not in _STYLIZED and edit_type not in _LIGHTLY_STYLIZED


def preprocess(in_path: str, out_path: str) -> int:
    img = Image.open(in_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    img.save(out_path, format="JPEG", quality=JPEG_Q, optimize=True)
    return os.path.getsize(out_path)


def main():
    os.makedirs(os.path.join(V2_ROOT, "originals"), exist_ok=True)
    os.makedirs(os.path.join(V2_ROOT, "edits"), exist_ok=True)

    with open(V1_META) as f:
        items = [json.loads(line) for line in f]
    photoreal = [i for i in items if is_photoreal(i.get("edit_type", ""))]
    print(f"sample_6k: {len(items)} total -> {len(photoreal)} photoreal")

    out_lines = []
    n_new = 0
    n_cached = 0
    for item in photoreal:
        slot = item["slot"]
        new_slot = slot + SLOT_OFFSET
        src_in = os.path.join(V1_ROOT, item["src_path"])
        edit_in = os.path.join(V1_ROOT, item["edit_path"])
        src_out = os.path.join(V2_ROOT, "originals", f"{new_slot:05d}.jpg")
        edit_out = os.path.join(V2_ROOT, "edits", f"{new_slot:05d}.jpg")

        if os.path.exists(src_out) and os.path.exists(edit_out):
            ss = os.path.getsize(src_out); es = os.path.getsize(edit_out)
            n_cached += 1
        else:
            try:
                ss = preprocess(src_in, src_out)
                es = preprocess(edit_in, edit_out)
                n_new += 1
            except Exception as e:
                print(f"  skip slot {slot}: {e}", file=sys.stderr)
                continue
        rec = {
            "slot": new_slot,
            "src_path": f"originals/{new_slot:05d}.jpg",
            "edit_path": f"edits/{new_slot:05d}.jpg",
            "src_size": ss,
            "edit_size": es,
            "open_image_input_url": item.get("open_image_input_url", ""),
            "output_image_rel": item.get("output_image_rel", ""),
            "edit_type": item.get("edit_type", ""),
            "text": item.get("text", ""),
            "summarized_text": item.get("summarized_text", ""),
            "origin": "v1_resized",
        }
        out_lines.append(json.dumps(rec))
        if (n_new + n_cached) % 500 == 0:
            print(f"  ... {n_new + n_cached} done ({n_new} new, {n_cached} cached)", flush=True)

    with open(NEW_META, "w") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"\nwrote {len(out_lines)} entries to {NEW_META}")
    print(f"  preprocessed {n_new} new pairs (existed before: {n_cached})")


if __name__ == "__main__":
    main()
