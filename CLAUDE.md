# CLAUDE.md — session notes for the next agent

**Branch:** `experiments-sdxl-rinse-ood` — **all session work is here**.
Other branches: `main` is the FedAvg memorization paper (unrelated); `synthid-regen-kit` is the older
SynthID-detection kit (earlier detector ckpt + technical report, pre-session).

**Session dates:** 2026-06-27 through 2026-07-02
**Working directory:** `/home/trabbani/WAVES`

## What this session investigated

Whether a post-hoc pixel-space classifier (`C_eval`) can rival Google SynthID's
adversarial robustness at detecting Nano-Banana (Gemini 2.5 Flash Image) edits
vs. real photographs. The high-level story that emerged:

- `C_eval` (a ResNet classifier trained on Pico-Banana photoreal pairs) is
  **trivially defeated** in pixel space by white-box PGD (1-2 steps at
  ε≤2/255), by everyday photo edits (brightness ×1.5), and by very light
  SDXL img2img rinse (strength=0.010).
- SynthID (via Gemini's public UI) **survives** the darkroom edits that break
  `C_eval` — but was **also** defeated by our light SDXL regens (empirical
  finding, stronger than published WAVES results suggested).
- The `deep-research` skill's investigation confirmed SynthID Image is
  itself a post-hoc pixel-space encoder+decoder pair (NOT an in-processing
  latent watermark like Tree-Ring), so its extra robustness is not
  structural — it's due to adversarial/perturbation-augmented joint training
  of the encoder+detector.

## Recommended checkpoints (all at `detector/*.pt`)

| ckpt | size | in-dist test | DiffDB (OOD) | Clean COCO-edit | Clean COCO-nat-TNR | Distortion flip rate | Note |
|---|---|---|---|---|---|---|---|
| `best_ood_balanced_r50.pt` | 90 MB | **96.16%** | 92.2% | **87.30%** | 90.86% | 18.5% | Baseline reference. ResNet-50, no adv training. |
| `best_adv_mixed_r18.pt` | 43 MB | 97.18% | 81.1% | 60.28% | 93.95% | – | s48 winner of mixed-batch frozen-attacker sweep. Best "pre-gen adv" score (+0.41 EXP-bal), but broken by fresh EOT-PGD in 1 step. |
| `best_adv_blur_r18.pt` | 43 MB | 89.96% | 92.49% | 82.06% | 78.63% | – | s48 winner with in-model Gaussian blur σ=1.0. Only ckpt where per-eps adv stays at chance not −1.0 (gentle curve). |
| `best_regen_aug_r18.pt` | 43 MB | 96.77% | **99.63%** | 74.80% | 93.15% | **13.8%** | s55 winner of SDXL 1×10 regen-augmented sweep. **Best practical distortion robustness.** Deployed detector recommendation. |
| `best_psnr25_r18.pt` | 43 MB | 92% | 58% | 56% | 72% | – | s48 winner of PSNR≥25 filtered training (bal +0.28 at ep2). Most theoretically interesting: forced to learn substrate-level signal because semantic content differences filtered out. |

All are `BinaryClassifier(backbone=...)` from `detector/models.py`; class-0 = natural, class-1 = synth. Input pipeline: resize 384×384 + JPEG q=95 matched-codec + normalize to [-1,1] via `data.to_tensor_neg1_1`.

Loading example:
```python
from detector.models import BinaryClassifier
import torch
m = BinaryClassifier(pretrained=False, backbone='resnet18').cuda().eval()
m.load_state_dict(torch.load('detector/best_regen_aug_r18.pt', map_location='cuda')['state_dict'])
```

## Key experimental findings

### 1. Adversarial-robustness training doesn't work for `C_eval`

Every adv-training variant we tried gets broken by white-box EOT-PGD in 1-2 sign-gradient steps at max|δ|=1-2/255:

| defense | EOT steps to flip | max\|δ\| |
|---|---|---|
| baseline R50 (no defense) | 2 | 2/255 |
| mixed_s48 R18 + Gaussian noise smoothing σ=0.025 | 1 | 1/255 |
| noiseaug_s53 R18 + smoothing | 1 | 1/255 |
| 4-model noise-aug ensemble + smoothing | 1 | 1/255 |
| multi-attacker s48 R18 + smoothing | 1 | 1/255 |
| blur-trained s48 R18 + blur preprocessing | 1 | 1/255 |
| jpeg-aug s48 R18 + JPEG TTA (BPDA) | 1 | 1/255 |

The "pre-gen adv eval" (perturbations crafted against a frozen R50 attacker, saved to disk) can *look* like defenses work — mixed_s48 gets +0.77 balanced score at σ=0.025 smoothing on the pre-gen eval — but this is **entirely a transfer-attack artifact**. When EOT recomputes gradients against the actual defended model, it wins in 1 step. Every time.

**Do not repeat these experiments.** If the next agent wants to try adv defense again, it must be with a fundamentally different substrate — see "open questions" below.

### 2. Non-adversarial "darkroom" attacks trivially break `C_eval` baseline

On a strongly-synth-classified COCO edit (00001.jpg, baseline P(synth)=0.996):

| attack | baseline_r50 P(nat) after |
|---|---|
| brightness ×1.5 (Instagram-style) | 0.984 (flipped) |
| brightness ×1.8 | 0.999 (flipped) |
| sharpen ×8 | 0.785 (flipped) |
| Gaussian noise σ=20 | 0.997 (flipped) |
| Gaussian noise σ=40 | 1.000 (flipped) |

An average user's photo-editing pipeline defeats the detector without any adversarial intent.

### 3. Regen-augmented training (`best_regen_aug_r18.pt`) gives real distortion generalization

Trained on 2500 photoreal pairs plus SDXL 1×10 strength=0.01 regens of the same pairs, labels preserved. Result across 65-op darkroom sweep on 8 confidently-classified COCO edits:

| detector | flip rate |
|---|---|
| baseline_r50 | 18.5% |
| **regen_aug_s55** | **13.8%** ← best |
| regen_geom_aug_s48 (regen + crop/rotate/downsample aug) | 24.1% |

Regen-aug wins on **intensity/photometric** attacks (brightness −15.6pp, sharpen −6.3pp, noise −6.2pp) but regresses on **geometric** attacks (down_up +30pp). Adding geometric augmentation didn't fix this and hurt in other categories — seed-to-seed variance dominated the effect.

### 4. Light SDXL regeneration defeats both `C_eval` AND SynthID

Both `baseline_r50` and Gemini-side SynthID failed on our SDXL 1×10 regens at strength=0.010, which is far below the strength usually reported as watermark-breaking in the WAVES paper. This is a **stronger empirical result than the published SynthID robustness numbers suggest** and warrants proper measurement (e.g. 20-image ladder from strength 0.001 to 0.020).

Files: `regen_attack_flips/`, `regen_attack_flips_00000/`.

### 5. PSNR-filtered training discovers a *transferable diffusion substrate*

When we filtered training to pairs with PSNR(natural, edit) ≥ 25 dB (only 6% of pairs, "near-identical" edits), the model *cannot* rely on semantic content differences. It found a signal that transferred to non-NB diffusion models: DiffDB (SDXL-based) jumped to **83.5%** at the peak ep2 vs. ~60-75% typical. Trade-off: nat-TNR = 41% — model calls a lot of natural photos synth too. `best_psnr25_r18.pt` is the s48-multi-seed sweep winner (bal +0.28).

## Data dependencies

Everything is at `/mnt/data`. If migrating, these paths need to be updated:

| path | contents |
|---|---|
| `/mnt/data/pico-banana-400k/photoreal_v2/` | 147k photoreal Pico-Banana pairs (main training set) |
| `/mnt/data/pico-banana-400k/photoreal_v2/metadata.jsonl` | pair index |
| `/mnt/data/pico_pair_psnr_photoreal_v2.json` | pre-computed PSNR for every pair (dict: slot→psnr, ~10 MB) — from `detector/compute_pair_psnr.py` |
| `/mnt/data/pico_regen_10step_v1/` | 5000 SDXL 1×10 regen'd training images + 500 regen'd OOD-COCO — from `detector/pregen_sdxl_regen_train.py` |
| `/mnt/data/coco_ood_v2_500/` | 496 OOD-COCO NB edits + 496 natural COCO. Also has adv-attacked variants at ε ∈ {1,2,4,8,16}. |
| `/mnt/data/coco_ood_v2_500/{edit,orig}_adv_eps{001..016}/` | Pre-generated adv OOD-COCO from `detector/gen_adv_ood_coco.py` |
| `/mnt/data/hf_cache/` | HuggingFace + SDXL cache |

Training entry points (each imports data + models from `detector/`):

| script | purpose |
|---|---|
| `detector/adv_train.py` | Main adv-training script. Flags: `--mixed-batch`, `--attacker-ckpt`, `--noise-aug-sigma`, `--blur-sigma`, `--train-jpeg-q-choices` |
| `detector/regen_aug_train.py` | Regen-augmented training (produced `best_regen_aug_r18.pt`) |
| `detector/psnr_filtered_train.py` | PSNR-filtered training (produced `best_psnr25_r18.pt`) |
| `detector/train_full.py` | Original plain-CE trainer (produced `best_ood_balanced_r50.pt`) |

Sweep bash scripts alongside each trainer follow the pattern `*_sweep.sh` — 4 seeds each, ~15-80 min total depending on training set size.

## Failed approaches — do not repeat

1. **Online PGD adversarial training** — collapses to random after warmup, regardless of LR / warmup length. Any online PGD against R50/R101 targeted at the trainee's own gradients destabilizes.
2. **TRADES loss with online PGD** — same collapse, one epoch later.
3. **Randomized smoothing at inference** on baseline_r50 — clean accuracy collapses at σ ≥ 0.025 (model was trained at q=95 clean, noise pushes everything to natural).
4. **Gaussian blur at inference** on any model — differentiable, EOT trivially routes around it.
5. **JPEG TTA at inference** with BPDA-aware attack — non-differentiable but attacker uses identity backward, breaks in 1 step anyway.
6. **Multi-attacker training** (R50 + R18-s48 + R101-s44 as attackers) — hurt clean OOD without giving real EOT robustness.
7. **Multi-epoch (8-20) adv training with any variant** — always overfits to in-dist by ep2-3, dropping OOD to near-zero. Peak is always ep0 or ep1.

Every one of these is documented with a run in `runs/` and results in `runs/*.json`. Grep for the specific approach to find the ckpts / logs.

## Suggested next research directions

Ordered by likely payoff:

1. **Systematic SynthID-robustness measurement.** The user has Gemini access; ladder Nano-Banana edits through SDXL regens at strengths {0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100} on 30-50 different images and record SynthID detection rate. If SynthID collapses at strength 0.01 as we saw on 2 images, this is a paper-worthy result.
2. **Frequency-domain classifier from scratch.** Train R18 that consumes 2D DCT coefficients directly with the top 50% of frequency bands zeroed. Pixel-PGD has no path to attack DCT features it can't reach. Would test whether the diffusion fingerprint has enough low-frequency content to survive.
3. **DDIM-inversion residual features.** Run each image through an open-source SDXL DDIM inverter, extract noise-reconstruction residual, classify on that. Closest to what an in-processing watermark verifier does. ~1 day to prototype.
4. **Detect-and-reject adversarial-input classifier.** Train a *separate* binary "is this PGD-attacked?" model on the pre-gen adv eval set. Sidesteps the robustness question — pipeline flags attacked inputs and refuses to classify them.
5. **Combine PSNR-filtered training with regen-augmentation.** `best_psnr25_r18.pt` and `best_regen_aug_r18.pt` learn different substrate signals; stacking may give both diffusion-fingerprint transfer AND distortion robustness. ~30 min sweep.
6. **Two-stage detector.** Stage 1: `best_psnr25_r18.pt` (high recall via synth-happy bias) says "diffusion-artifact-like." Stage 2: `best_ood_balanced_r50.pt` (high precision) confirms. Then adv-attack the ensemble.

## User preferences (from memory, `feedback-keep-iterating`)

**Don't propose stopping, don't propose "writing up the negative result" unprompted.** After a failure, immediately propose and launch the next angle. This user wants to keep iterating on research problems until they explicitly say stop. Frame failures as "X didn't work, trying Y next" not "we've exhausted the options."

## How to resume

1. Read this file.
2. Verify data at `/mnt/data/pico-banana-400k/photoreal_v2/` still exists.
3. Run `git log --oneline -20` on branch `experiments-sdxl-rinse-ood` for recent commits.
4. To reproduce any result: the sweep script name matches the ckpt name (e.g. `best_regen_aug_r18.pt` ← `regen_aug_sweep.sh`).
5. To try a new defense angle: pick one from "suggested next directions" and just launch. The user's feedback memory is loaded automatically.

## Failure ledger (attack images shipped)

All at repo root:

| dir | contents |
|---|---|
| `regen_attack_flips/` | SDXL rinse ladder on 00001.jpg — 7 variants, baseline_r50 breaks at 1×40 (strength=0.04). Push these to SynthID to confirm the cross-detector break. |
| `regen_attack_flips_00000/` | Same ladder on 00000.jpg for cross-check |
| `darkroom_attack_flips/` | Brightness/sharpen/noise variants on 00001.jpg that flip baseline_r50. SynthID robust to these (per user's manual test). |
| `adv_attack_vs_s48/`, `adv_attack_eot_vs_*/`, `adv_attack_eot_bpda_jpeg_s48/` | Various EOT/BPDA attack outputs against adv-trained models. All broke in 1-2 steps. |
