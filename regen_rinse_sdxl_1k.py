"""Rinsed-regen sweep: same 1000 sample_6k pairs as before, but the
attack is *two* successive SDXL DDIM regens at the same N. Configs
match the WAVES rinse notation: 10x10, 20x20, 40x40 == regen N=10
twice, regen N=20 twice, regen N=40 twice. Round 1 uses SEED, round 2
uses SEED+1 so the second pass adds genuinely new noise (same seed
would just re-walk the same DDIM trajectory).

Same per-pair metrics (PSNR/SSIM/MS-SSIM/LPIPS/CLIP-sim) plus FID, and
the same atomic 25-pair JSON checkpoints + 100-pair running summary.
Caches *final* rinsed images to /mnt/data/regen_cache_sdxl_rinse_1k/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf_cache")
os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")
os.environ.setdefault("TORCH_HOME", "/mnt/data/torch_cache")
os.makedirs("/mnt/data/hf_cache", exist_ok=True)
os.makedirs("/mnt/data/torch_cache", exist_ok=True)

import lpips                                                       # noqa: E402
import open_clip                                                   # noqa: E402
from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline  # noqa: E402
from pytorch_msssim import ms_ssim, ssim                           # noqa: E402
from skimage.metrics import peak_signal_noise_ratio                # noqa: E402
from torchmetrics.image.fid import FrechetInceptionDistance        # noqa: E402

META = "/mnt/data/pico-banana-400k/sample_6k/metadata.jsonl"
ROOT = "/mnt/data/pico-banana-400k/sample_6k"
CACHE = Path("/mnt/data/regen_cache_sdxl_rinse_1k")
OUT = "/home/trabbani/WAVES/regen_rinse_sdxl_1k.json"

MODEL_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

N_PAIRS = 1000
# (n_steps_per_round, n_rounds) — label = f"{n}x{n}" in display
RINSE_CONFIGS = [(10, 2), (20, 2), (40, 2)]
LABELS = [f"{n}x{n}" for n, k in RINSE_CONFIGS]
NUM_INFERENCE_STEPS = 1000
SEED = 1024
GUIDANCE = 1.0
CKPT_EVERY = 25
REPORT_EVERY = 100

CACHE.mkdir(parents=True, exist_ok=True)


def round8(x):
    return int(x) - (int(x) % 8)


def center_crop(img, w, h):
    W, H = img.size
    left, top = (W - w) // 2, (H - h) // 2
    return img.crop((left, top, left + w, top + h))


def pil_chw01(img, device):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def pil_uint8_299(img, device):
    r = img.convert("RGB").resize((299, 299), Image.BICUBIC)
    a = np.asarray(r, dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def psnr_np(a_pil, b_pil):
    A = np.asarray(a_pil.convert("RGB"), dtype=np.uint8)
    B = np.asarray(b_pil.convert("RGB"), dtype=np.uint8)
    return float(peak_signal_noise_ratio(A, B, data_range=255))


@torch.no_grad()
def ssim_torch(a01, b01):
    return float(ssim(a01, b01, data_range=1.0).item())


@torch.no_grad()
def msssim_torch(a01, b01):
    return float(ms_ssim(a01, b01, data_range=1.0).item())


@torch.no_grad()
def lpips_torch(a01, b01, net):
    return float(net(a01 * 2 - 1, b01 * 2 - 1).mean().item())


@torch.no_grad()
def clip_cos(a_pil, b_pil, clip_model, clip_preprocess, device):
    a = clip_preprocess(a_pil).unsqueeze(0).to(device)
    b = clip_preprocess(b_pil).unsqueeze(0).to(device)
    fa = clip_model.encode_image(a)
    fb = clip_model.encode_image(b)
    fa = fa / fa.norm(dim=-1, keepdim=True)
    fb = fb / fb.norm(dim=-1, keepdim=True)
    return float((fa * fb).sum().item())


def regen_xl_one(pipe, img, n_steps, seed):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt="", prompt_2="", image=img,
        strength=n_steps / NUM_INFERENCE_STEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    ).images[0]


def regen_rinse(pipe, img, n_steps, n_rounds, seed_base):
    out = img
    for r in range(n_rounds):
        out = regen_xl_one(pipe, out, n_steps, seed_base + r)
    return out


def atomic_dump(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, path)


def print_running_summary(rows, fids):
    print(f"\n  n = {len(rows)}", flush=True)
    print(f"  {'metric':<10s}  {'config':>6s}  "
          f"{'mean orig':>10s}  {'mean edit':>10s}  {'edit-orig':>10s}",
          flush=True)
    for m in ["psnr", "ssim", "msssim", "lpips", "clipsim"]:
        for lab in LABELS:
            lo = np.array([r[f"{m}_orig_{lab}"] for r in rows])
            le = np.array([r[f"{m}_edit_{lab}"] for r in rows])
            print(f"  {m:<10s}  {lab:>6s}  {lo.mean():>10.4f}  {le.mean():>10.4f}  "
                  f"{(le-lo).mean():>+10.4f}", flush=True)
    for lab in LABELS:
        try:
            fo = float(fids[("orig", lab)].compute())
            fe = float(fids[("edit", lab)].compute())
            print(f"  {'fid':<10s}  {lab:>6s}  {fo:>10.3f}  {fe:>10.3f}  "
                  f"{fe-fo:>+10.3f}", flush=True)
        except Exception as e:
            print(f"  fid         {lab:>6s}  -- ({e})", flush=True)
    print(flush=True)


def main():
    device = "cuda"
    print(f"loading SDXL ({MODEL_SDXL}) ...", flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_SDXL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print("SDXL loaded", flush=True)

    print("loading LPIPS ...", flush=True)
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    print(f"loading CLIP ({CLIP_MODEL_NAME}/{CLIP_PRETRAINED}) ...", flush=True)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)
    clip_model.eval()

    print(f"loading FID InceptionV3 ({2*len(LABELS)}x) ...", flush=True)
    fids = {(side, lab): FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            for side in ["orig", "edit"] for lab in LABELS}
    for fid in fids.values():
        fid.eval()
    torch.cuda.synchronize()
    print(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    items = []
    skipped = 0
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            try:
                eW, eH = Image.open(os.path.join(ROOT, d["edit_path"])).size
                oW, oH = Image.open(os.path.join(ROOT, d["src_path"])).size
            except Exception:
                skipped += 1
                continue
            tW, tH = round8(eW), round8(eH)
            if tW < 512 or tH < 512 or oW < tW or oH < tH:
                skipped += 1
                continue
            d["_tW"], d["_tH"] = tW, tH
            items.append(d)
            if len(items) == N_PAIRS:
                break
    print(f"selected {len(items)} pairs (skipped {skipped})", flush=True)

    rows = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        tW, tH = item["_tW"], item["_tH"]
        try:
            orig_full = Image.open(os.path.join(ROOT, item["src_path"])).convert("RGB")
            edit_full = Image.open(os.path.join(ROOT, item["edit_path"])).convert("RGB")
        except Exception as e:
            print(f"  [skip {i}] open failed: {e}", flush=True)
            continue
        orig = center_crop(orig_full, tW, tH)
        edit = center_crop(edit_full, tW, tH)

        rec = {"slot": item["slot"], "edit_type": item.get("edit_type", ""),
               "size": [tW, tH]}

        orig01 = pil_chw01(orig, device)
        edit01 = pil_chw01(edit, device)
        orig_299 = pil_uint8_299(orig, device)
        edit_299 = pil_uint8_299(edit, device)

        for (n_steps, n_rounds), lab in zip(RINSE_CONFIGS, LABELS):
            r_orig = regen_rinse(pipe, orig, n_steps, n_rounds, SEED)
            r_edit = regen_rinse(pipe, edit, n_steps, n_rounds, SEED)
            r_orig.save(CACHE / f"{item['slot']:06d}_orig_{lab}.jpg", "JPEG", quality=95)
            r_edit.save(CACHE / f"{item['slot']:06d}_edit_{lab}.jpg", "JPEG", quality=95)

            r_orig01 = pil_chw01(r_orig, device)
            r_edit01 = pil_chw01(r_edit, device)

            rec[f"psnr_orig_{lab}"]    = psnr_np(orig, r_orig)
            rec[f"psnr_edit_{lab}"]    = psnr_np(edit, r_edit)
            rec[f"ssim_orig_{lab}"]    = ssim_torch(orig01, r_orig01)
            rec[f"ssim_edit_{lab}"]    = ssim_torch(edit01, r_edit01)
            rec[f"msssim_orig_{lab}"]  = msssim_torch(orig01, r_orig01)
            rec[f"msssim_edit_{lab}"]  = msssim_torch(edit01, r_edit01)
            rec[f"lpips_orig_{lab}"]   = lpips_torch(orig01, r_orig01, lpips_net)
            rec[f"lpips_edit_{lab}"]   = lpips_torch(edit01, r_edit01, lpips_net)
            rec[f"clipsim_orig_{lab}"] = clip_cos(orig, r_orig, clip_model, clip_preprocess, device)
            rec[f"clipsim_edit_{lab}"] = clip_cos(edit, r_edit, clip_model, clip_preprocess, device)

            fids[("orig", lab)].update(orig_299, real=True)
            fids[("orig", lab)].update(pil_uint8_299(r_orig, device), real=False)
            fids[("edit", lab)].update(edit_299, real=True)
            fids[("edit", lab)].update(pil_uint8_299(r_edit, device), real=False)
        rows.append(rec)

        if i % CKPT_EVERY == 0 or i == len(items):
            atomic_dump(rows, OUT)
            el = time.time() - t0
            rate = i / el
            print(f"  [{i:4d}/{len(items)}]  {el:7.0f}s  ETA {(len(items)-i)/rate/60:7.1f} min  "
                  f"last={tW}x{tH}  ckpt", flush=True)

        if i % REPORT_EVERY == 0 or i == len(items):
            print_running_summary(rows, fids)

    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
