"""Generate 512x512 photoreal images via Gemini 3.1 Flash Image Preview,
using prompts derived from Pico-Banana edit instructions, for use as
extra OOD training/eval data.

Pipeline:
  1. Sample N entries from sft.jsonl (seed=200, distinct from any other run)
  2. Build a prompt:  "<photoreal_prefix>. <scene_phrase derived from entry>"
  3. Call gemini-3.1-flash-image-preview with response_modalities=['IMAGE']
  4. Validate output: must be a non-degenerate image, both dims >= 512
  5. Center-crop to square -> resize to 512x512 -> save JPEG q=95
  6. Append metadata.jsonl per success; log failures separately

The photoreal prefix is intentionally heavy on camera+lens cues — Gemini
tends to default to stylized illustration without strong photoreal forcing.
"""

import argparse
import io
import json
import os
import random
import sys
import time

import dotenv
from PIL import Image, UnidentifiedImageError

# Load API key from project .env
dotenv.load_dotenv("/home/trabbani/WAVES/.env")

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402


PHOTOREAL_PREFIX = (
    "Photorealistic photograph captured with a Nikon D850 DSLR, 50mm prime lens at f/2.8, "
    "natural daylight, sharp focus, high detail, fine film grain. No illustration, no painting, no cartoon. "
    "Subject:"
)


def build_prompt(entry: dict) -> str:
    """Translate a Pico-Banana entry into a self-contained image-generation prompt.

    `summarized_text` is short and reads like an edit description ('Remove flag,
    extend sky and dune'). We feed it to Gemini as the subject specification —
    Gemini will render *the resulting scene* as a photo.
    """
    scene = entry.get("summarized_text") or entry.get("text", "")
    return f"{PHOTOREAL_PREFIX} {scene}"


def square_resize(img: Image.Image, size: int = 512) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def generate_one(client, model: str, prompt: str, timeout: int = 60):
    """Returns (PIL.Image | None, error_str | None)."""
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
    except Exception as e:
        return None, f"api:{type(e).__name__}: {str(e)[:160]}"

    if not resp.candidates:
        return None, "no_candidates"

    for c in resp.candidates:
        if not c.content or not c.content.parts:
            continue
        for p in c.content.parts:
            if p.inline_data is not None and p.inline_data.data:
                try:
                    img = Image.open(io.BytesIO(p.inline_data.data)).convert("RGB")
                    return img, None
                except (UnidentifiedImageError, Exception) as e:
                    return None, f"decode:{type(e).__name__}: {str(e)[:120]}"
    return None, "no_image_part_in_response"


def validate(img: Image.Image, min_side: int = 512) -> str | None:
    """Return None if OK, else error string."""
    if img is None:
        return "img_is_none"
    w, h = img.size
    if w < min_side or h < min_side:
        return f"too_small_{w}x{h}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--jsonl", default="/home/trabbani/pico-banana-400k/jsonl/sft.jsonl")
    ap.add_argument("--out-dir", default="/mnt/data/synth_gemini_300")
    ap.add_argument("--model", default="gemini-3.1-flash-image-preview")
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    metadata_path = os.path.join(args.out_dir, "metadata.jsonl")
    failures_path = os.path.join(args.out_dir, "failures.jsonl")

    completed_slots = set()
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            for line in f:
                try:
                    completed_slots.add(json.loads(line)["slot"])
                except Exception:
                    pass
    print(f"already completed: {len(completed_slots)}")

    print(f"loading {args.jsonl} ...")
    with open(args.jsonl) as f:
        all_entries = [json.loads(l) for l in f]
    print(f"  {len(all_entries)} SFT entries")
    random.seed(args.seed)
    sampled = random.sample(all_entries, args.n)
    print(f"sampled {args.n} entries (seed={args.seed})")

    pending = [(i, e) for i, e in enumerate(sampled) if i not in completed_slots]
    print(f"pending: {len(pending)}; model={args.model}; size={args.size}")

    n_ok = 0
    n_fail = 0
    t0 = time.time()
    for idx, (slot, entry) in enumerate(pending):
        prompt = build_prompt(entry)
        last_err = None
        img = None
        for attempt in range(args.max_retries + 1):
            img, err = generate_one(client, args.model, prompt)
            if img is not None and validate(img, args.size) is None:
                last_err = None
                break
            last_err = err or validate(img, args.size)
            if attempt < args.max_retries:
                time.sleep(2 + 2 * attempt)

        if img is None or last_err is not None:
            n_fail += 1
            with open(failures_path, "a") as f:
                f.write(json.dumps({"slot": slot, "prompt": prompt[:200], "error": last_err}) + "\n")
            print(f"[{idx + 1:3d}/{len(pending)}] FAIL slot={slot}: {last_err}", flush=True)
            continue

        out = square_resize(img, args.size)
        out_path = os.path.join(args.out_dir, f"{slot:04d}.jpg")
        out.save(out_path, format="JPEG", quality=95, optimize=True)
        n_ok += 1
        with open(metadata_path, "a") as f:
            f.write(json.dumps({
                "slot": slot,
                "out_path": f"{slot:04d}.jpg",
                "prompt": prompt,
                "raw_size": list(img.size),
                "saved_size": [args.size, args.size],
                "edit_type": entry.get("edit_type", ""),
                "summarized_text": entry.get("summarized_text", ""),
            }) + "\n")
        elapsed = time.time() - t0
        rate = (idx + 1) / max(elapsed, 1e-9)
        eta = (len(pending) - idx - 1) / max(rate, 1e-9)
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"[{idx + 1:3d}/{len(pending)}] ok={n_ok} fail={n_fail}  "
                  f"({elapsed:5.0f}s, {rate:4.2f}/s, ETA {eta/60:5.1f} min)", flush=True)

    print(f"\ndone. ok={n_ok} fail={n_fail}  total time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
