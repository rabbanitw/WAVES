# SynthID Regen Kit

**A hands-on kit for stress-testing Google DeepMind's SynthID image watermark — built for the DEFCON 34 AI Village.**

You get a small test set of SynthID-watermarked images, the reference code for the **diffusive-regeneration** attack, an attempted **GAN-based watermark eraser** (which mostly doesn't work — included as a worked negative example), and pre-rendered "before / regen-at-N-steps" comparison strips so you can eyeball what each attack actually costs the image.

If you came here to find a quick way to scrub SynthID off an image, the honest answer is in the table at the bottom of this file: diffusive regen at moderate strength cracks the watermark but visibly degrades the picture, and the GAN approach (which leaves the picture pristine) doesn't actually remove the watermark.

---

## What's in the box

| path | what |
|---|---|
| `images/` | **104** SynthID-watermarked Nano-Banana images at 512×512 (`image_0.jpg` … `image_103.jpg`) plus `prompts.txt` (the 104 generation prompts, one per line, indexed by image number). Held-out target set every experiment scores against. |
| `images/regen_{10,20,40,80}/` | Pre-rendered regen-attacked copies of all 104 test images at symmetric N=10, 20, 40, 80 (JPEG q=95, same `image_N.jpg` naming as sources). Skip the GPU run and use these directly for downstream analysis. |
| `regen/` | The diffusive-regeneration attack. A small, self-contained library: a vendored `ReSDPipeline` (a `StableDiffusionPipeline` subclass that lets you resume denoising from a pre-noised latent) plus the symmetric N-step regen function. |
| `gan/` | The GAN-based attack attempt. U-Net generator + PatchGAN discriminator + LPIPS edit-preservation. **This is the "GAN that doesn't quite work"** — useful as a worked example of why naive generator-based watermark removal is harder than it looks. |
| `pico_banana_pairs/` | 15 (real-photo, Nano-Banana-edit) example pairs spanning 15 different edit categories. Sourced from Apple's Pico-Banana-400K. Lets you sanity-check the data pipeline without downloading the 400K-image full set. |
| `examples/regen_progression/` | Pre-rendered 5-panel "before / after" strips for prompts 0, 25, 100 of the test set. Each strip shows: original \| N=10 regen \| N=20 \| N=40 \| N=80. **Open one and look at it before doing anything else.** |
| `examples/make_collages.py` | Rebuilds the strips from a regenerated test set. |
| `requirements.txt` | Verified pin set. The vendored pipeline is tied to a specific `diffusers` version; if you upgrade, you'll break it. |

---

## Setup

```bash
# CUDA 11.8 wheels — what we tested against
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Stable Diffusion 1.4 (the surrogate model the regen attack runs through) auto-downloads on first call (~5 GB into your HuggingFace cache).

A GPU is strongly recommended. CPU works but is ~60× slower; a single N=10 regen takes ~40 s on a 12-core Xeon vs ~0.7 s on an A100.

---

## Quick run — regenerate one image, see the watermark survive or not

```python
from PIL import Image
from regen import build_pipeline, regen_symmetric

pipe = build_pipeline("CompVis/stable-diffusion-v1-4", device="cuda")
src = Image.open("images/image_0.jpg")
attacked = regen_symmetric(src, pipe, n_steps=20)  # try 10, 20, 40, 80
attacked.save("/tmp/attacked.png")
```

Run the image you actually care about through Google's SynthID Detector (Vertex AI / AI Studio) before and after. At N≈10-20 most images come back as "no watermark detected" but still look close to the original; at N=80 you've destroyed the watermark and most of the image's fine detail too.

---

## Reproducing the comparison strips

The 5-panel strips in `examples/regen_progression/` were generated from regen-attacked copies of the test set at four depths. They're not shipped because they're large; regenerate yourself:

```bash
# the regen attack outputs are already in images/regen_{10,20,40,80}/,
# so rebuilding the strips is just one step:
python examples/make_collages.py --prompts 0 25 100

# if you ever want to re-run the attack yourself (with a different model,
# scheduler, seed, etc.):
for N in 10 20 40 80; do
  python -m regen --src images/ --dst /tmp/regen_$N/ --n-steps $N
done
```

On an A100 the full sweep takes ~25 min for all four depths × 104 images.

---

## The GAN attack (worked negative example)

If you want to try the GAN path:

```bash
# Requires the full Pico-Banana-400K SFT split — see https://github.com/apple/pico-banana-400k.
# Point gan/data.py::DatasetConfig at where you put it.
python gan/train_gan.py --epochs 40 --batch 16 --lambda-preserve 10

# Score: label-flip rate on a held-out binary classifier + LPIPS edit-fidelity
python gan/eval.py --gan-ckpt path/to/ckpt.pt --c-eval-ckpt path/to/c_eval.pt
```

**What you get from this:** the GAN converges, the output looks pixel-perfect (LPIPS ≈ 0.031, perturbations as visible "black blobs" on close inspection), and it only flips **11.3%** of held-out classifier predictions. Run the same outputs through actual SynthID and detection stays high. The GAN learned to fool the classifier, not to remove the watermark.

Why include it then? Because the experiment **is the lesson**: surrogate-classifier-based adversarial attacks don't transfer to a real watermark detector when you don't have decoder access — the same finding the WAVES paper (arXiv 2401.08573, §F.3.2) reports for their AdvCls family. Trying it yourself is faster than reading the paper.

---

## What we actually found

| attack | how visible | held-out classifier detection rate | matches real SynthID behavior? |
|---|---|---|---|
| (no attack — baseline) | — | **91.3%** flagged as AI-generated | — |
| diffusive regen, N=10 (sym DDIM) | mild softening | 64.4% | yes — also breaks real SynthID at ~similar rate |
| diffusive regen, N=20 | noticeable | 58.7% | yes |
| diffusive regen, N=40 | obvious blur | 53.8% | yes |
| diffusive regen, N=80 | heavily degraded | **45.2%** | yes — strongest tested |
| WAVES-asymmetric N=20 (sparse denoise) | similar to sym N=20 | 60.6% | yes |
| Pix2Pix-style GAN (this kit, λ=10) | invisible to eye | flips 89% of classifier preds | **no** — bypasses classifier without touching watermark |

Numbers are on the 104-image `images/` set. The "real SynthID" column is qualitative — we ran a handful of attacked images through the actual detector and the regen results lined up with the classifier story; the GAN results did not.

---

## License / credits

- The diffusive regeneration pipeline and `ReSDPipeline` are adapted from Zhao et al., "Invisible Image Watermarks Are Provably Removable Using Generative AI" (NeurIPS 2023), as bundled by the WAVES benchmark (UMD Huang Lab, arXiv 2401.08573, MIT). This kit cherry-picks just the relevant attack code, no upstream dependency.
- Nano-Banana paired samples in `pico_banana_pairs/` are from Apple's Pico-Banana-400K (CC BY-NC-ND 4.0), built on Open Images (CC BY 2.0).
- The 104 test images in `images/` were produced by Nano-Banana for SynthID stress-testing and are included for research/demo use only.
- This kit is MIT-licensed.
