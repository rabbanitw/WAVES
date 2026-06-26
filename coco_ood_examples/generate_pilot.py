"""Pilot OOD-generation pipeline: 10 COCO val2017 images spanning
diverse categories. For each:
  1. Ask Gemini Vision (gemini-2.5-flash) for a realistic image-specific
     edit instruction in Pico-Banana style.
  2. Apply the edit via Gemini's image-editing endpoint (gemini-2.5-flash-image).
  3. Save orig + edit + metadata for inspection.

Output: /home/trabbani/WAVES/coco_ood_examples/{orig,edit}/000XX.jpg
        /home/trabbani/WAVES/coco_ood_examples/metadata.jsonl
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv("/home/trabbani/WAVES/.env")

COCO_IMG_DIR = "/mnt/data/coco_val2017/val2017"
COCO_INSTANCES = "/mnt/data/coco_val2017/annotations/instances_val2017.json"
OUT_DIR = Path("/home/trabbani/WAVES/coco_ood_examples/v2")
ORIG_DIR = OUT_DIR / "orig"
EDIT_DIR = OUT_DIR / "edit"
META = OUT_DIR / "metadata.jsonl"

VISION_MODEL = "gemini-2.5-flash"
EDIT_MODEL = "gemini-2.5-flash-image"
SLEEP = 18
N_IMAGES = 10
SEED = 42

ORIG_DIR.mkdir(parents=True, exist_ok=True)
EDIT_DIR.mkdir(parents=True, exist_ok=True)

VISION_PROMPT = """You are looking at a single photograph. Suggest ONE realistic, photoreal edit to apply to it, written in the verbose art-direction style used by professional image-editing briefs.

Constraints:
- The edit must reference specific content visible in this image (an object, the lighting, the weather, a color you can see, a region of the frame).
- The edit must be feasible by a state-of-the-art image-editor (weather change, object add/remove/replace, lighting shift, color tone, zoom, outpainting, attribute change).
- The result must remain photoreal -- NOT stylized, NOT anime, NOT painting.
- Aim for ~40-60 words across 3-6 clauses. Include:
  * material / texture specifics (e.g. "rich mahogany wood tone with a satin finish")
  * color and lighting details (specific colors, light direction, contrast goals)
  * one or more "while preserving X" or "ensuring Y" clauses for elements that should stay unchanged
- Output ONLY the imperative edit instruction itself, as a single dense run-on sentence or 2-3 sentences. No preamble, no "here is", no "I suggest"."""


def load_diverse_image_ids():
    """Pick N_IMAGES image_ids that each contain a different primary
    COCO category, so the pilot spans a variety of subjects."""
    with open(COCO_INSTANCES) as f:
        ann = json.load(f)
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    by_cat = {}
    for a in ann["annotations"]:
        by_cat.setdefault(a["category_id"], []).append(a["image_id"])

    rng = random.Random(SEED)
    chosen = []
    used_imgs = set()
    cat_ids = list(by_cat.keys())
    rng.shuffle(cat_ids)
    for cid in cat_ids:
        if len(chosen) == N_IMAGES:
            break
        # pick one image with this primary cat, that we haven't seen
        candidates = [iid for iid in by_cat[cid] if iid not in used_imgs]
        if not candidates:
            continue
        iid = rng.choice(candidates)
        chosen.append((iid, cats[cid]))
        used_imgs.add(iid)
    return chosen


def main():
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    pairs = load_diverse_image_ids()
    print(f"selected {len(pairs)} diverse COCO images:", flush=True)
    for iid, cat in pairs:
        print(f"  {iid:>12d}  -> {cat}", flush=True)

    t0 = time.time()
    for i, (img_id, primary_cat) in enumerate(pairs):
        src_path = Path(COCO_IMG_DIR) / f"{img_id:012d}.jpg"
        slot = i
        print(f"\n[{slot+1}/{len(pairs)}] img_id={img_id} primary={primary_cat}", flush=True)

        src = Image.open(src_path).convert("RGB")
        # save original as JPEG q=95 (no resize -- keep COCO native res)
        src.save(ORIG_DIR / f"{slot:05d}.jpg", "JPEG", quality=95, optimize=True)
        print(f"  src size: {src.size}", flush=True)

        # 1. Vision: ask for an edit suggestion
        try:
            resp = client.models.generate_content(
                model=VISION_MODEL,
                contents=[src, VISION_PROMPT],
            )
            edit_text = resp.text.strip()
            print(f"  edit_instruction: {edit_text[:200]}", flush=True)
        except Exception as e:
            print(f"  [VISION FAIL] {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue

        time.sleep(SLEEP)

        # 2. Apply the edit
        try:
            resp = client.models.generate_content(
                model=EDIT_MODEL,
                contents=[src, edit_text],
            )
            png_bytes = None
            for cand in (resp.candidates or []):
                for part in (cand.content.parts or []):
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        png_bytes = part.inline_data.data
                        break
                if png_bytes is not None:
                    break
            if png_bytes is None:
                print(f"  [EDIT FAIL] no inline image in response", flush=True)
                continue
            edited = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            print(f"  edited size: {edited.size}", flush=True)
            edited.save(EDIT_DIR / f"{slot:05d}.jpg", "JPEG", quality=95, optimize=True)
        except Exception as e:
            print(f"  [EDIT FAIL] {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue

        with open(META, "a") as f:
            f.write(json.dumps({
                "slot": slot,
                "coco_img_id": img_id,
                "primary_cat": primary_cat,
                "src_native_size": list(src.size),
                "edit_native_size": list(edited.size),
                "edit_instruction": edit_text,
            }) + "\n")

        el = time.time() - t0
        print(f"  done. elapsed {el:.0f}s", flush=True)
        if slot < len(pairs) - 1:
            time.sleep(SLEEP)

    print(f"\ndone. outputs in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
