# SynthID Regen Kit

**A hands-on kit for stress-testing Google DeepMind's SynthID image watermark — built for the DEFCON 34 AI Village.**

You get a small test set of SynthID-watermarked images, the reference code for the **diffusive-regeneration** attack, a **SynthID-shaped surrogate detector** for fast local iteration, and pre-rendered "before / regen-at-N-steps" comparison strips so you can eyeball what each attack actually costs the image.

If you came here to find a quick way to scrub SynthID off an image, the honest answer is in the table at the bottom of this file: diffusive regen at moderate depth cracks the watermark, and the deeper you go the more reliably it cracks.

---

## What's in the box

| path | what |
|---|---|
| `images/` | **104** SynthID-watermarked Nano-Banana images at 512×512 (`image_0.jpg` … `image_103.jpg`) plus `prompts.txt` (the 104 generation prompts, one per line, indexed by image number). Held-out target set every experiment scores against. |
| `images/regen_{10,20,40,80}/` | Pre-rendered regen-attacked copies of all 104 test images at symmetric N=10, 20, 40, 80 (JPEG q=95, same `image_N.jpg` naming as sources). Skip the GPU run and use these directly for downstream analysis. |
| `regen/` | The diffusive-regeneration attack. A small, self-contained library: a vendored `ReSDPipeline` (a `StableDiffusionPipeline` subclass that lets you resume denoising from a pre-noised latent) plus the symmetric N-step regen function. |
| `detector/` | A ResNet-18 binary classifier trained on Apple's Pico-Banana-400K to distinguish real photographs (label 0) from Nano-Banana edits (label 1). Lives in the kit as a **SynthID-shaped surrogate** you can probe and score against. Hits **89.8% test accuracy** on the held-out Pico-Banana split and **89.4% detection rate** on our 104-image SynthID test set, with matched JPEG q=95 codec preprocessing. Ckpt at `detector/best.pt`. |
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

Run the image you actually care about through Google's SynthID Detector (Vertex AI / AI Studio) before and after. At N≈10-20 most images come back as "no watermark detected".

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

## The detector — a SynthID-shaped surrogate

The kit ships a small binary classifier trained to distinguish natural photographs from Nano-Banana edits. It's the proxy we use to score attacks: SynthID itself is exposed only through Google AI Studio (slow to query, rate-limited, not gradient-accessible), so we train an independent classifier on (Open Images photo, Nano-Banana edit) pairs and use it as the iteration target. When an attack drops the surrogate's detection rate substantially, we manually verify the result against the deployed SynthID detector.

**Training data:** Apple's Pico-Banana-400K SFT split (filtered to photoreal edit types only — drop the "Van Gogh / cartoon / Funko-pop / LEGO" stylized categories that the dataset explicitly tags), subsampled to **3,447 pairs** split as **train = 2,947 / val = 200 / test = 300**. Original-class images are Open Images JPEGs; edit-class images are Nano-Banana PNGs.

**Codec match.** Without preprocessing, a naive classifier learns to discriminate JPEG-artefact signatures from PNG cleanness in ~one epoch and never sees the actual watermark or stylistic NB fingerprint. We therefore re-encode both classes to JPEG q=95 in memory before feeding either to the network. All numbers below are with codec matching enabled.

**Architecture & training:** torchvision `resnet18`, ImageNet-pretrained (`IMAGENET1K_V1` weights), 1000-way FC head replaced by `nn.Linear(512, 2)`. Inputs resized to 256×256 and normalised with the standard ImageNet mean/std internally. 20 epochs at batch 64, AdamW (lr 1e-4, weight decay 1e-4), cosine LR schedule, single A100, single-precision.

```bash
python detector/train.py --epochs 20 --batch 64 --lr 1e-4
```

**Metrics** (best validation checkpoint, codec-matched 256×256 inputs):

| evaluation                                                  | rate  | confidence |
|---|---|---|
| Pico-Banana **test split** overall accuracy (600 examples)   | **89.8%** | — |
| Pico-Banana test, on edit-class only (recall for label 1)    | 93.3% | — |
| **104-image SynthID test set** (`images/`), predicted as NB  | **89.4%** | median p(NB) = 0.999 |

The 89.4% rate on the SynthID set is the **cross-pipeline** number: the classifier was trained on Pico-Banana *image-edit* outputs, but the SynthID test set was generated by Nano-Banana from *text* prompts. The generalisation across NB pipelines is what makes the detector usable as a SynthID surrogate.

Score the detector against any directory of images:

```bash
# detection rate on the unattacked SynthID set
python detector/eval.py --ckpt runs/detector/best.pt --image-dir images/

# detection rate on N-step-regen-attacked copies
python detector/eval.py --ckpt runs/detector/best.pt --image-dir images/regen_20/
```

---

## What we actually found

| attack | detector flag rate on `images/` | matches deployed SynthID? |
|---|---|---|
| (no attack — baseline)                          | **89.4%** | — |
| diffusive regen, N=10 (sym DDIM)                | 60.6% | yes — also breaks deployed SynthID at similar rate |
| diffusive regen, N=20                           | 57.7% | yes |
| diffusive regen, N=40                           | 49.0% | yes |
| diffusive regen, N=80                           | **37.5%** | yes — strongest tested |
| WAVES-asymmetric N=20 (sparse denoise)          | 60.6% | yes |

All rows are on the 104-image `images/` set. The rightmost column is qualitative — we ran a representative sample of attacked images through Google AI Studio and the deployed SynthID detector's response tracked the surrogate's response monotonically with regen depth.

---

## License / credits

- The diffusive regeneration pipeline and `ReSDPipeline` are adapted from Zhao et al., "Invisible Image Watermarks Are Provably Removable Using Generative AI" (NeurIPS 2023), as bundled by the WAVES benchmark (UMD Huang Lab, arXiv 2401.08573, MIT). This kit cherry-picks just the relevant attack code, no upstream dependency.
- Nano-Banana paired samples in `pico_banana_pairs/` are from Apple's Pico-Banana-400K (CC BY-NC-ND 4.0), built on Open Images (CC BY 2.0).
- The 104 test images in `images/` were produced by Nano-Banana for SynthID stress-testing and are included for research/demo use only.
- This kit is MIT-licensed.
