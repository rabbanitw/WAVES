# v5 SynthID A/B test samples

8 pairs from the `valid_512/` SynthID set. Both images per pair are 256×256 PNG, **after** matched JPEG q=95 preprocessing — same pipeline our classifier was trained under, so they are a fair A/B.

| pair | files |
|---|---|
| prompt_000 | `prompt_000_orig.png` &nbsp;&nbsp; `prompt_000_attacked.png` |
| prompt_001 | `prompt_001_orig.png` &nbsp;&nbsp; `prompt_001_attacked.png` |
| prompt_010 | `prompt_010_orig.png` &nbsp;&nbsp; `prompt_010_attacked.png` |
| prompt_042 | `prompt_042_orig.png` &nbsp;&nbsp; `prompt_042_attacked.png` |
| prompt_050 | `prompt_050_orig.png` &nbsp;&nbsp; `prompt_050_attacked.png` |
| prompt_077 | `prompt_077_orig.png` &nbsp;&nbsp; `prompt_077_attacked.png` |
| prompt_090 | `prompt_090_orig.png` &nbsp;&nbsp; `prompt_090_attacked.png` |
| prompt_100 | `prompt_100_orig.png` &nbsp;&nbsp; `prompt_100_attacked.png` |

The `_attacked` version is `G(orig)` where G is the v5 FlatPerturbator (trained against C_train, evaluated on C_eval). Per-pixel L_∞ distance ≤ 16/255 on every pair (saturated at the budget). LPIPS distance ≈ 0.012 — visually imperceptible.

Important caveats for SynthID interpretation:
- These are **resized to 256×256 then JPEG q=95 re-encoded** before G is applied. Resize alone is a known watermark-disrupting transform, so the *baseline* `_orig.png` may already have weakened SynthID compared to the raw 512×512 source.
- The fair A/B is `_orig.png` vs `_attacked.png` — both have been through the same preprocessing, only `_attacked.png` has the additional G perturbation applied. So if SynthID detects on `_orig.png` but not on `_attacked.png`, that's the GAN's effect, isolated from preprocessing.
- The raw 512×512 sources are at `valid_512/prompt_<N>_attempt_1_img_0_512x512.jpg` if you also want to compare to those.
