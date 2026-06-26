"""Rinse-to-plateau across N pairs from sample_6k. For each pair, rinse
each side at N=10 for up to MAX_RINSES, recording LPIPS+PSNR to start.
Report per-pair and population-level statistics."""

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

import lpips                                                          # noqa: E402
from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio                   # noqa: E402

META = "/mnt/data/pico-banana-400k/sample_6k/metadata.jsonl"
ROOT = "/mnt/data/pico-banana-400k/sample_6k"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
OUT = "/home/trabbani/WAVES/rinse_until_plateau_multi.json"

N_STEPS_PER_RINSE = 10
NUM_INFERENCE_STEPS = 1000
SEED_BASE = 1024
GUIDANCE = 1.0
MAX_RINSES = 50
N_PAIRS = 10

PLATEAU_WIN = 5
PLATEAU_LPIPS_TIGHT = 0.005
PLATEAU_LPIPS_LOOSE = 0.010
PLATEAU_PSNR_TIGHT = 0.3
PLATEAU_PSNR_LOOSE = 0.6


def round8(x):
    return int(x) - (int(x) % 8)


def center_crop(img, w, h):
    W, H = img.size
    return img.crop(((W - w) // 2, (H - h) // 2,
                     (W - w) // 2 + w, (H - h) // 2 + h))


def pil_chw01(img, device):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def regen_once(pipe, img, n_steps, seed):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt="", prompt_2="", image=img,
                strength=n_steps / NUM_INFERENCE_STEPS,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE,
                generator=gen).images[0]


def first_plateau(seq, win, eps):
    for i in range(win, len(seq) + 1):
        window = seq[i - win:i]
        if max(window) - min(window) < eps:
            return i
    return None


def rinse_side(pipe, net, start, device):
    start01 = pil_chw01(start, device)
    A = np.asarray(start.convert("RGB"), dtype=np.uint8)
    current = start
    traj = []
    for k in range(1, MAX_RINSES + 1):
        current = regen_once(pipe, current, N_STEPS_PER_RINSE, SEED_BASE + k)
        cur01 = pil_chw01(current, device)
        with torch.no_grad():
            lp = float(net(start01 * 2 - 1, cur01 * 2 - 1).mean().item())
        B = np.asarray(current.convert("RGB"), dtype=np.uint8)
        psnr = float(peak_signal_noise_ratio(A, B, data_range=255))
        traj.append({"rinse": k, "lpips": lp, "psnr": psnr})
    return traj


def main():
    device = "cuda"
    print(f"loading SDXL ({MODEL}) ...", flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print("SDXL loaded", flush=True)

    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    items = []
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            try:
                eW, eH = Image.open(os.path.join(ROOT, d["edit_path"])).size
                oW, oH = Image.open(os.path.join(ROOT, d["src_path"])).size
            except Exception:
                continue
            tW, tH = round8(eW), round8(eH)
            if tW >= 512 and tH >= 512 and oW >= tW and oH >= tH:
                d["_tW"], d["_tH"] = tW, tH
                items.append(d)
                if len(items) == N_PAIRS:
                    break
    print(f"selected {len(items)} pairs", flush=True)

    records = []
    for idx, d in enumerate(items, 1):
        tW, tH = d["_tW"], d["_tH"]
        orig = center_crop(
            Image.open(os.path.join(ROOT, d["src_path"])).convert("RGB"), tW, tH)
        edit = center_crop(
            Image.open(os.path.join(ROOT, d["edit_path"])).convert("RGB"), tW, tH)

        print(f"\n[{idx}/{len(items)}] slot {d['slot']} {tW}x{tH}  "
              f"edit_type='{d.get('edit_type', '')}'", flush=True)
        t0 = time.time()
        traj_o = rinse_side(pipe, net, orig, device)
        t1 = time.time()
        traj_e = rinse_side(pipe, net, edit, device)
        t2 = time.time()

        lps_o = [t["lpips"] for t in traj_o]
        lps_e = [t["lpips"] for t in traj_e]
        psnrs_o = [t["psnr"] for t in traj_o]
        psnrs_e = [t["psnr"] for t in traj_e]

        # Plateau detection
        plat = {
            "orig_lpips_tight": first_plateau(lps_o, PLATEAU_WIN, PLATEAU_LPIPS_TIGHT),
            "orig_lpips_loose": first_plateau(lps_o, PLATEAU_WIN, PLATEAU_LPIPS_LOOSE),
            "edit_lpips_tight": first_plateau(lps_e, PLATEAU_WIN, PLATEAU_LPIPS_TIGHT),
            "edit_lpips_loose": first_plateau(lps_e, PLATEAU_WIN, PLATEAU_LPIPS_LOOSE),
            "orig_psnr_tight":  first_plateau(psnrs_o, PLATEAU_WIN, PLATEAU_PSNR_TIGHT),
            "orig_psnr_loose":  first_plateau(psnrs_o, PLATEAU_WIN, PLATEAU_PSNR_LOOSE),
            "edit_psnr_tight":  first_plateau(psnrs_e, PLATEAU_WIN, PLATEAU_PSNR_TIGHT),
            "edit_psnr_loose":  first_plateau(psnrs_e, PLATEAU_WIN, PLATEAU_PSNR_LOOSE),
        }
        print(f"  orig: end LPIPS={lps_o[-1]:.4f} PSNR={psnrs_o[-1]:.2f}dB  "
              f"plateau(LPIPS<0.01)={plat['orig_lpips_loose']}  ({t1-t0:.0f}s)",
              flush=True)
        print(f"  edit: end LPIPS={lps_e[-1]:.4f} PSNR={psnrs_e[-1]:.2f}dB  "
              f"plateau(LPIPS<0.01)={plat['edit_lpips_loose']}  ({t2-t1:.0f}s)",
              flush=True)

        records.append({
            "slot": d["slot"],
            "size": [tW, tH],
            "edit_type": d.get("edit_type", ""),
            "traj_orig": traj_o,
            "traj_edit": traj_e,
            "plateau": plat,
        })

        # Save incremental
        with open(OUT, "w") as f:
            json.dump({"max_rinses": MAX_RINSES, "n_steps_per_rinse": N_STEPS_PER_RINSE,
                       "records": records}, f, indent=2)

    # Population summary
    print(f"\n\n========== POPULATION SUMMARY (n={len(records)}) ==========", flush=True)
    print(f"{'slot':>6s}  {'oLPIPS_end':>10s}  {'eLPIPS_end':>10s}  "
          f"{'o_plat':>7s}  {'e_plat':>7s}  {'oPSNR_end':>9s}  {'ePSNR_end':>9s}",
          flush=True)
    for r in records:
        op = r["plateau"]["orig_lpips_loose"]; ep = r["plateau"]["edit_lpips_loose"]
        print(f"{r['slot']:>6d}  "
              f"{r['traj_orig'][-1]['lpips']:>10.4f}  {r['traj_edit'][-1]['lpips']:>10.4f}  "
              f"{str(op):>7s}  {str(ep):>7s}  "
              f"{r['traj_orig'][-1]['psnr']:>9.2f}  {r['traj_edit'][-1]['psnr']:>9.2f}",
              flush=True)

    def stats(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        if len(a) == 0:
            return "all None"
        return (f"n={len(a)}/{len(vals)}  mean={a.mean():.1f}  med={np.median(a):.1f}  "
                f"min={a.min():.0f}  max={a.max():.0f}")

    o_plats_loose = [r["plateau"]["orig_lpips_loose"] for r in records]
    e_plats_loose = [r["plateau"]["edit_lpips_loose"] for r in records]
    o_plats_tight = [r["plateau"]["orig_lpips_tight"] for r in records]
    e_plats_tight = [r["plateau"]["edit_lpips_tight"] for r in records]
    o_end_lpips = np.array([r["traj_orig"][-1]["lpips"] for r in records])
    e_end_lpips = np.array([r["traj_edit"][-1]["lpips"] for r in records])
    o_end_psnr = np.array([r["traj_orig"][-1]["psnr"] for r in records])
    e_end_psnr = np.array([r["traj_edit"][-1]["psnr"] for r in records])

    print(f"\nLPIPS-plateau (window=5, span<0.01) rinses:")
    print(f"  orig:  {stats(o_plats_loose)}")
    print(f"  edit:  {stats(e_plats_loose)}")
    print(f"LPIPS-plateau (window=5, span<0.005) rinses:")
    print(f"  orig:  {stats(o_plats_tight)}")
    print(f"  edit:  {stats(e_plats_tight)}")

    print(f"\nEndpoint LPIPS @ rinse {MAX_RINSES}:")
    print(f"  orig:  mean={o_end_lpips.mean():.4f}  std={o_end_lpips.std():.4f}")
    print(f"  edit:  mean={e_end_lpips.mean():.4f}  std={e_end_lpips.std():.4f}")
    print(f"  edit - orig gap: mean={ (e_end_lpips-o_end_lpips).mean():+.4f}  "
          f"#pairs(edit>orig): {(e_end_lpips>o_end_lpips).sum()}/{len(records)}")

    print(f"\nEndpoint PSNR @ rinse {MAX_RINSES}:")
    print(f"  orig:  mean={o_end_psnr.mean():.2f}  std={o_end_psnr.std():.2f}")
    print(f"  edit:  mean={e_end_psnr.mean():.2f}  std={e_end_psnr.std():.2f}")
    print(f"  edit - orig gap: mean={ (e_end_psnr-o_end_psnr).mean():+.2f}")


if __name__ == "__main__":
    main()
