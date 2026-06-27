"""Full OOD-set generation: scaled-up v2 pipeline.

For N diverse COCO val2017 images, run the two-stage Gemini Vision -> Gemini
edit pipeline with the verbose PB-style prompt template (see generate_pilot.py
for the prompt itself), saving each (orig, edit) pair at NB native resolution.

Resumable: skips any slot whose metadata is already recorded. Outputs are
saved outside the repo (to /mnt/data) so git stays clean."""

from __future__ import annotations

import argparse
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

VISION_MODEL = "gemini-2.5-flash"
EDIT_MODEL = "gemini-2.5-flash-image"

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


def diverse_image_ids(n, seed=42):
    with open(COCO_INSTANCES) as f:
        ann = json.load(f)
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    by_cat = {}
    for a in ann["annotations"]:
        by_cat.setdefault(a["category_id"], []).append(a["image_id"])

    rng = random.Random(seed)
    chosen = []
    used = set()
    # Cycle through shuffled cat_ids over and over, picking one fresh image
    # per cat per cycle, until we have N.
    cat_ids = list(by_cat.keys())
    rng.shuffle(cat_ids)
    cycle = 0
    while len(chosen) < n:
        added_this_cycle = 0
        for cid in cat_ids:
            if len(chosen) == n:
                break
            cands = [iid for iid in by_cat[cid] if iid not in used]
            if not cands:
                continue
            iid = rng.choice(cands)
            chosen.append((iid, cats[cid]))
            used.add(iid)
            added_this_cycle += 1
        if added_this_cycle == 0:
            break  # exhausted all images
        cycle += 1
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out-dir", default="/mnt/data/coco_ood_v2_500")
    ap.add_argument("--sleep", type=float, default=18.0,
                    help="seconds between API calls (18 = free-tier-safe, 6 = paid-tier-tight)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    orig_dir = out / "orig"
    edit_dir = out / "edit"
    orig_dir.mkdir(parents=True, exist_ok=True)
    edit_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    fail_path = out / "failures.jsonl"

    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    pairs = diverse_image_ids(args.n, seed=args.seed)
    print(f"selected {len(pairs)} diverse COCO images", flush=True)

    # Resume: skip slots already in metadata
    done = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slot"])
                except Exception:
                    pass
    print(f"already done: {len(done)} slots", flush=True)

    t0 = time.time()
    new_ok = new_fail = 0
    for slot, (img_id, primary_cat) in enumerate(pairs):
        if slot in done:
            continue
        src_path = Path(COCO_IMG_DIR) / f"{img_id:012d}.jpg"
        try:
            src = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"[{slot:5d}] open fail: {e}", flush=True)
            new_fail += 1
            continue

        # 1. vision-prompt suggestion
        try:
            resp = client.models.generate_content(
                model=VISION_MODEL,
                contents=[src, VISION_PROMPT],
            )
            edit_text = resp.text.strip()
        except Exception as e:
            err = f"vision_{type(e).__name__}: {str(e)[:160]}"
            with open(fail_path, "a") as f:
                f.write(json.dumps({"slot": slot, "img_id": img_id, "error": err}) + "\n")
            new_fail += 1
            time.sleep(args.sleep)
            continue
        time.sleep(args.sleep)

        # 2. edit
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
                raise RuntimeError("no inline image data in response")
            edited = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception as e:
            err = f"edit_{type(e).__name__}: {str(e)[:160]}"
            with open(fail_path, "a") as f:
                f.write(json.dumps({"slot": slot, "img_id": img_id, "error": err,
                                    "edit_text": edit_text}) + "\n")
            new_fail += 1
            time.sleep(args.sleep)
            continue

        # Save (no resize -- both at native resolution; downstream code can resize)
        src.save(orig_dir / f"{slot:05d}.jpg", "JPEG", quality=95, optimize=True)
        edited.save(edit_dir / f"{slot:05d}.jpg", "JPEG", quality=95, optimize=True)
        with open(meta_path, "a") as f:
            f.write(json.dumps({
                "slot": slot,
                "coco_img_id": img_id,
                "primary_cat": primary_cat,
                "src_native_size": list(src.size),
                "edit_native_size": list(edited.size),
                "edit_instruction": edit_text,
            }) + "\n")
        new_ok += 1

        if (new_ok + new_fail) % 10 == 0 or slot == len(pairs) - 1:
            el = time.time() - t0
            done_count = new_ok + new_fail
            rate = done_count / max(el, 1e-9)
            remaining = len(pairs) - len(done) - done_count
            eta_min = remaining / max(rate, 1e-9) / 60
            print(f"[{slot+1:5d}/{len(pairs)}] new_ok={new_ok} new_fail={new_fail}  "
                  f"{rate*60:5.2f}/min  ETA {eta_min:6.1f} min", flush=True)

        time.sleep(args.sleep)

    print(f"\ndone. new_ok={new_ok}  new_fail={new_fail}", flush=True)


if __name__ == "__main__":
    main()
