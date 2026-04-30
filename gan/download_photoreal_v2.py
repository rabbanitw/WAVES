"""Download a large photoreal-only sample with on-the-fly preprocessing.

Differences from the throttled v1:
* Pre-filter the SFT manifest to photoreal edit_types BEFORE sampling
  (so the 30K we attempt are all photoreal candidates, not 35K SFT of
  which only ~68% are photoreal).
* Download both images, resize to 384x384 (Lanczos), encode JPEG q=95,
  and save THAT — never store the raw 1-2 MB PNG/JPEG. Disk per pair
  drops from ~3 MB to ~120 KB. 30K pairs ≈ 3.5 GB.
* Same throttling and resumability as v1: 4 workers, 1s pacing per
  Flickr subdomain, exp-backoff on 429, incremental metadata.
"""

import argparse
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from PIL import Image

CDN_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset download)"}

_STYLIZED_EXCLUDE = {
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
    "Add film grain or vintage filter",
    "Modern ↔ historical style/look",
    "Apply seasonal transformation (summer ↔ winter)",
}


MIN_INTERVAL_FLICKR = 1.0
MIN_INTERVAL_CDN = 0.05
_host_last = {}
_host_lock = threading.Lock()


def pace(host: str) -> None:
    interval = MIN_INTERVAL_FLICKR if "flickr" in host else MIN_INTERVAL_CDN
    while True:
        with _host_lock:
            now = time.monotonic()
            last = _host_last.get(host, 0.0)
            wait = (last + interval) - now
            if wait <= 0:
                _host_last[host] = now
                return
        time.sleep(wait)


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    backoff = 60
    last_err = None
    for attempt in range(5):
        host = urlparse(url).netloc
        pace(host)
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
        except requests.exceptions.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            last_err = "429"
            continue
        if 400 <= r.status_code < 500:
            raise requests.HTTPError(f"{r.status_code} {url}")
        time.sleep(5 * (attempt + 1))
        last_err = f"{r.status_code}"
    raise requests.HTTPError(f"giving up: {last_err} {url}")


def preprocess(raw: bytes, size: int, jpeg_q: int) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=jpeg_q, optimize=True)
    return out.getvalue()


def download_pair(slot: int, entry: dict, out_dir: str, size: int, jpeg_q: int):
    src_url = entry["open_image_input_url"]
    edit_url = CDN_BASE + entry["output_image"]
    src_path = os.path.join(out_dir, "originals", f"{slot:05d}.jpg")
    edit_path = os.path.join(out_dir, "edits", f"{slot:05d}.jpg")

    try:
        src_raw = fetch_bytes(src_url)
        src_jpg = preprocess(src_raw, size, jpeg_q)
        with open(src_path, "wb") as f:
            f.write(src_jpg)
        edit_raw = fetch_bytes(edit_url)
        edit_jpg = preprocess(edit_raw, size, jpeg_q)
        with open(edit_path, "wb") as f:
            f.write(edit_jpg)
        return (slot, True, "ok", len(src_jpg), len(edit_jpg))
    except Exception as e:
        for p in (src_path, edit_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return (slot, False, f"{type(e).__name__}: {str(e)[:120]}", 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/trabbani/pico-banana-400k/jsonl/sft.jsonl")
    ap.add_argument("--out-dir", default="/mnt/data/pico-banana-400k/photoreal_v2")
    ap.add_argument("--n", type=int, default=35000)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--jpeg-q", type=int, default=95)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "originals"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "edits"), exist_ok=True)

    metadata_path = os.path.join(args.out_dir, "metadata.jsonl")
    failures_path = os.path.join(args.out_dir, "failures.jsonl")

    print(f"loading {args.jsonl} ...")
    photoreal_entries = []
    total = 0
    with open(args.jsonl) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            if d.get("edit_type", "") not in _STYLIZED_EXCLUDE:
                photoreal_entries.append(d)
    print(f"  total entries: {total}")
    print(f"  photoreal entries: {len(photoreal_entries)} ({100*len(photoreal_entries)/total:.1f}%)")

    n = min(args.n, len(photoreal_entries))
    random.seed(args.seed)
    sampled = random.sample(photoreal_entries, n)
    print(f"sampled {n} photoreal candidates (seed={args.seed})")

    completed_slots = set()
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            for line in f:
                try:
                    completed_slots.add(json.loads(line)["slot"])
                except Exception:
                    pass
    print(f"already completed: {len(completed_slots)}")

    pending = [(i, e) for i, e in enumerate(sampled) if i not in completed_slots]
    print(f"pending: {len(pending)}; workers={args.workers}")

    write_lock = threading.Lock()
    new_ok = 0
    new_fail = 0
    bytes_ok = 0
    t0 = time.time()
    last_print = t0

    def write_meta(rec):
        with write_lock:
            with open(metadata_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def write_fail(rec):
        with write_lock:
            with open(failures_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_pair, i, e, args.out_dir, args.size, args.jpeg_q):
                   (i, e) for i, e in pending}
        completed = 0
        for fut in as_completed(futures):
            slot, ok, msg, ss, es = fut.result()
            i, entry = futures[fut]
            completed += 1
            if ok:
                new_ok += 1
                bytes_ok += ss + es
                write_meta({
                    "slot": slot,
                    "src_path": f"originals/{slot:05d}.jpg",
                    "edit_path": f"edits/{slot:05d}.jpg",
                    "src_size": ss,
                    "edit_size": es,
                    "open_image_input_url": entry["open_image_input_url"],
                    "output_image_rel": entry["output_image"],
                    "edit_type": entry.get("edit_type", ""),
                    "text": entry.get("text", ""),
                    "summarized_text": entry.get("summarized_text", ""),
                })
            else:
                new_fail += 1
                write_fail({"slot": slot, "error": msg,
                            "src_url": entry["open_image_input_url"]})
            now = time.time()
            if now - last_print >= 30 or completed == len(pending):
                el = now - t0
                rate = completed / max(el, 1e-9)
                eta = (len(pending) - completed) / max(rate, 1e-9)
                gb = bytes_ok / 1024**3
                print(f"[{completed:5d}/{len(pending)}] new_ok={new_ok} new_fail={new_fail} "
                      f"{gb:5.2f} GB  {rate:5.2f} pairs/s  ETA {eta/60:5.1f} min",
                      flush=True)
                last_print = now

    print(f"\ndone. new ok={new_ok} new fail={new_fail}")


if __name__ == "__main__":
    main()
