# WAVES — minimal SynthID-robustness repro

Stripped-down branch of `umd-huang-lab/WAVES` containing just what's needed to:

1. Run the **diffusive-regeneration** attack on the 104-image SynthID test set (`valid_512/`).
2. Train and evaluate the **v1 watermark-removal GAN** (Pix2Pix-style U-Net generator + PatchGAN discriminator with LPIPS edit-preservation).
3. Inspect what each attack does visually (`examples/regen_progression/` shows 3 source images at N = 0, 10, 20, 40, 80 regen depths side by side).

Full WAVES paper: arXiv 2401.08573. Upstream repo: <https://github.com/umd-huang-lab/WAVES>.

---

## Repository layout

| path | what |
|---|---|
| `valid_512/` | 104 Nano-Banana-generated, SynthID-watermarked images at 512×512 (the original WAVES SynthID test set we use as the robustness target) |
| `regeneration/` | Vendored `ReSDPipeline` (a `StableDiffusionPipeline` subclass that resumes denoising from a pre-noised latent) plus the `DiffWMAttacker` from WAVES that drives it |
| `gan/` | The v1 watermark-removal GAN: `models.py` (U-Net generator, PatchGAN discriminator, ResNet-18 binary classifier), `data.py` (paired Pico-Banana dataloader with matched-codec preprocessing), `train_gan.py`, `eval.py` |
| `pico_banana_samples/` | 15 paired (original Open Images photo, Nano-Banana edit) examples from Apple's Pico-Banana-400K, spanning 15 different `edit_type`s — for sanity-checking the data pipeline without downloading the full dataset |
| `examples/regen_progression/` | Pre-rendered 5-panel collages showing prompts 0, 50, 100 from `valid_512/` at N = 0 (original), 10, 20, 40, 80 symmetric-regen depths |
| `examples/make_collages.py` | Script that regenerates the collages from `valid_512/` and the four regen-output directories (which are NOT in this branch — see "Reproducing the collages" below) |
| `requirements.txt` | Minimal pin set, see "Install" |

## Install

The vendored `ReSDPipeline` is wired against an older `diffusers` API; the pins below are the combination we have end-to-end-verified. CUDA 11.8 wheels are what we used:

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The Stable Diffusion 1.4 weights (used as the surrogate model for regeneration) get auto-downloaded by HuggingFace on first run (~5 GB into your HF cache).

## Reproducing the collages

The collages in `examples/regen_progression/` were rendered from the four regen-output directories below. We don't ship those in this branch because they're large (~50 MB each); regenerate them on your end:

```bash
# Generate regen'd versions of valid_512 at N = 10, 20, 40, 80 symmetric DDIM steps.
# See regeneration/regen.py for the symmetric-regen function (we use
# DDIMScheduler + num_inference_steps = 1000 + a head_start_step that yields
# exactly N denoising iterations at training-timestep N-1 noise level).
for N in 10 20 40 80; do
  python -m regeneration.regen \
      --src valid_512/ \
      --dst dev_test/regen_N$(printf %03d $N)/ \
      --n-steps $N
done

# Build the 5-panel collages.
python examples/make_collages.py --prompts 0 50 100
```

(N=10 takes ~1 sec per image on an A100; N=80 takes ~5 sec/image. CPU is ~60× slower.)

## Quick experiments

**Train the v1 watermark-removal GAN** (requires the full Pico-Banana-400K SFT split — see <https://github.com/apple/pico-banana-400k> — preprocessed under `gan/data.py::DatasetConfig`'s expected layout):

```bash
python gan/train_gan.py --epochs 40 --batch 16 --lambda-preserve 10
```

**Evaluate a trained checkpoint** (computes label-flip rate against a held-out C_eval classifier, edit-fidelity LPIPS, and writes a 24-image visual grid):

```bash
python gan/eval.py --gan-ckpt path/to/ckpt_last.pt --c-eval-ckpt path/to/c_eval.pt
```

## Reference numbers from our run

The v1 GAN at λ_preserve=10, 40 epochs, on the photoreal-only subset of Pico-Banana:

- **label-flip rate on held-out C_eval = 11.3%** (only 11% of NB edits get flipped to "real")
- **LPIPS(G(edit), edit) = 0.031** (perturbations are small but visually noticeable as localized "black blobs")

This was the experiment that motivated the AdvGAN-style v5 redesign (FlatPerturbator + L∞ ≤ 16/255) which is *not* in this branch — see the parent repo for the full thread.

## License

WAVES is MIT-licensed; this branch inherits that. Pico-Banana-400K samples in `pico_banana_samples/` are subject to its CC BY-NC-ND 4.0 license (and the underlying Open Images CC BY 2.0). The 104 SynthID images in `valid_512/` come from the original WAVES paper's evaluation set.
