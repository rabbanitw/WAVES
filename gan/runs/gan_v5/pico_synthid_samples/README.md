# v5 Pico-Banana A/B test samples

8 (NB edit, G(NB edit)) pairs from the **Pico-Banana-400K test split** (G's in-distribution data — these specific images were never seen during training, but they come from the same generation pipeline G learned to attack). Each pair spans a different edit_type for diversity.

| slot | edit_type | files |
|---:|---|---|
| 5036 | Replace one object category with another | `slot_05036_orig.png` / `slot_05036_attacked.png` |
| 5883 | Relocate an object (change spatial relation) | `slot_05883_orig.png` / `slot_05883_attacked.png` |
| 1301 | Change an object's attribute (color/material) | `slot_01301_orig.png` / `slot_01301_attacked.png` |
| 5150 | Add a new object to the scene | `slot_05150_orig.png` / `slot_05150_attacked.png` |
| 0636 | Zoom in | `slot_00636_orig.png` / `slot_00636_attacked.png` |
| 3669 | Change weather conditions | `slot_03669_orig.png` / `slot_03669_attacked.png` |
| 0975 | Outpainting (extend canvas beyond boundaries) | `slot_00975_orig.png` / `slot_00975_attacked.png` |
| 3089 | Remove an existing object | `slot_03089_orig.png` / `slot_03089_attacked.png` |

Both versions are 256×256 PNG, after the matched JPEG q=95 codec preprocessing.

* `_orig.png` = the unmodified Nano-Banana edited image (presumed SynthID-watermarked).
* `_attacked.png` = G(orig) where G is the v5 FlatPerturbator. Per-pixel L_∞ distance saturated at 16/255 on every pair; LPIPS ≈ 0.012, visually imperceptible.

If SynthID flags `_orig.png` but not `_attacked.png`, that's evidence the watermark itself was disrupted by the GAN attack (not just the NB stylistic fingerprint that C_eval also picked up on).
