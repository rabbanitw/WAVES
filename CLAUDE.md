# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WAVES (Watermark Analysis via Enhanced Stress-testing) is a benchmark for image-watermark robustness. The pipeline is: take watermarked images → apply an attack (distortion / regeneration / adversarial) → try to decode the watermark → measure detection accuracy and image-quality degradation. Three watermarking methods are first-class: `tree_ring`, `stable_sig`, `stegastamp`, evaluated against three image sources: `diffusiondb`, `mscoco`, `dalle3`.

## Environment setup

```bash
bash shell_scripts/install_dependencies.sh   # creates ./venv, pins to CUDA 11.8 / PyTorch 2.1.0
```

The script ends by symlinking `libnvrtc-*.so.11.2 -> libnvrtc.so` inside the venv to work around a torch+CUDNN bug — keep that step if you rebuild the env.

There are four split requirements files. Pick by role: `requirements_cli.txt` (analysis/CLI), `requirements_attack.txt` (running attacks/training surrogates), `requirements_space.txt` (Gradio app only), `requirements_all.txt` (everything).

`pip install -e .` (driven by `setup.py`) installs the package as `wmbench` and exposes the `wmbench` console script bound to `cli:cli`. Version is hardcoded in both `setup.py` and `cli.py` (`version` command) — keep them in sync.

### Required env vars (loaded via `python-dotenv`)

- `DATA_DIR` — root for all image directories. Every script that touches images calls `parse_image_dir_path`, which **rejects any path not under `DATA_DIR`**.
- `RESULT_DIR` — root for all JSON outputs (`*-status.json`, `*-reverse.json`, `*-decode.json`, `*-metric.json`).
- `GITHUB_TOKEN`, `REPO_URL`, `BRANCH_NAME` — only needed by `app.py` (the Gradio space pulls results from a git repo).

A `.env` at the repo root is the expected place to set these.

## Directory layout convention (enforced, not just documented)

`dev/find.py::parse_image_dir_path` parses paths by splitting on `/` and reading the last three components, so the layout is mandatory:

```
$DATA_DIR/main/<dataset>/<source>/{0..4999}.png
$DATA_DIR/attacked/<dataset>/<attack_name>-<attack_strength>-<source>/{0..4999}.png
$RESULT_DIR/<dataset>/<source>-<result_type>.json                    # for main/
$RESULT_DIR/<dataset>/<attack>-<strength>-<source>-<result_type>.json # for attacked/
```

Where `<dataset> ∈ {diffusiondb, mscoco, dalle3}`, `<source> ∈ {real, tree_ring, stable_sig, stegastamp, real_*}`, `<result_type> ∈ {status, reverse, decode, metric}`, and `<attack_strength>` must parse as a positive float.

`LIMIT = 5000` (full eval) and `SUBSET_LIMIT = 1000` (quick eval) in `dev/constants.py` are baked into how every script iterates and how completeness is checked. Don't change these without understanding that JSON results are keyed by string indices `"0".."4999"`.

## CLI

```bash
python cli.py <subcommand> [--all] [--dry] -- [script-specific args]
# or, after pip install -e .:
wmbench <subcommand> ...
```

Subcommands `status`, `reverse`, `decode`, `metric` all delegate to a per-script CLI; `--all` iterates every directory found by `get_all_image_dir_paths()` in parallel (`ProcessPoolExecutor`, 16 workers). When `--all` is set, `--path` is rejected. Without `--all`, the script's own `--path` flag (default cwd) is used.

**Known footgun:** `cli.py` builds subprocess paths as `dev_scripts/<name>.py`, but the scripts actually live in `scripts/`. If `wmbench status`/`reverse`/`decode`/`metric`/`chmod` fail with "no such file," that's why — invoke `python scripts/<name>.py` directly, or rename/symlink `scripts/` to `dev_scripts/`. The non-delegating commands (`version`, `space`) work fine.

`python cli.py space` launches the Gradio UI in `app.py`.

## Code architecture

The codebase splits into three layers.

### `dev/` — analysis library

Re-exported through `dev/__init__.py` so other code does `from dev import ...`. This is the only "library" surface:

- `constants.py` — canonical names for datasets, watermark methods, attacks, metrics, and the `GROUND_TRUTH_MESSAGES` (gzip+base64-encoded numpy arrays decoded at import time). Treat these dicts as the source of truth; new attacks/methods/metrics need to be registered here to flow through aggregation and plotting.
- `find.py` — path parsing/validation and directory enumeration. All path-shape rules live here.
- `io.py` — JSON read/write (orjson), `chmod_group_write`, and the gzip+base64 codecs used to embed numpy arrays and PNGs inside JSON.
- `eval.py` — bit-error rate, complex L1 (for tree-ring), detection metrics (AUC, TPR@1%FPR, TPR@0.1%FPR).
- `aggregate.py` — pulls per-image results out of the per-directory JSONs and rolls them up; has its own cache (`clear_aggregated_cache`).
- `parse.py`, `plot.py` — readers for individual JSONs and Plotly figures used by `app.py`.

### `scripts/` — pipeline stages (each is a self-contained `click` script)

Run in this order against an image directory:

1. **`status.py`** — verifies `{i}.png` exists for each `i ∈ [0, LIMIT)` and writes thumbnails for indices `[0, 1, 10, 100]`. Output: `<...>-status.json`.
2. **`reverse.py`** — runs DDIM inversion (Stable Diffusion) to produce `{i}_reversed.pkl` next to the images. Only meaningful for `tree_ring` (which decodes from latents, not pixels).
3. **`decode.py`** — runs the appropriate decoder per source. ONNX models in `decoders/` for `stable_sig` and `stegastamp`; tree-ring decodes from the reversed latents. The `mode` argument selects `tree_ring | stable_sig | stegastamp`. Output: `<...>-decode.json` keyed by index, with the decoded message base64-encoded.
4. **`metric.py`** — computes image-quality metrics (FID, CLIP-FID, PSNR, SSIM, NMI, LPIPS, Watson-DFT, aesthetics, artifacts, CLIP-score) by comparing the attacked directory against its un-attacked source. Output: `<...>-metric.json`.

`scripts/chmod.py` is the implementation behind `wmbench chmod` (group-writable so multiple lab users can share `$DATA_DIR`/`$RESULT_DIR`).

### Attack implementations (consumed by external pipelines, not by `cli.py`)

- `distortions/` — JPEG, blur, crop, rotation, brightness/contrast, erasing, noise, plus combos. Names registered in `ATTACK_NAMES`.
- `regeneration/` — diffusion-based and VAE-based regeneration (single, 2x, 4x).
- `adversarial/` — `embedding.py` (adversarial perturbation in feature space), `surrogate.py` + `train.py` (training surrogate watermark detectors), with feature extractors (CLIP / ResNet18 / VAEs) and pretrained surrogate weights under their respective subdirectories.
- `metrics/` — wraps LPIPS, clean-FID, aesthetic scorer, CLIP score; called from `scripts/metric.py`.

The expected workflow is: an external attack run writes images to `$DATA_DIR/attacked/.../`, then `wmbench --all status / reverse / decode / metric` populates `$RESULT_DIR`, then `dev/aggregate.py` (via notebooks or `app.py`) builds the comparison tables and plots.

### External code (vendored, mostly untouched)

`guided_diffusion/`, `ldm/`, `tree_ring/` are vendored from upstream watermarking/diffusion projects. Treat them as third-party — patch only what's needed, don't refactor.

## Adding things

- **New attack:** add an entry to `ATTACK_NAMES` in `dev/constants.py`, write the attack code under `distortions/`, `regeneration/`, or `adversarial/`, and emit images into `$DATA_DIR/attacked/<dataset>/<attack>-<strength>-<source>/`. The rest of the pipeline picks it up automatically once those files exist.
- **New watermark method:** add to `WATERMARK_METHODS` and `GROUND_TRUTH_MESSAGES` in `dev/constants.py`, drop a decoder in `decoders/`, and extend the `mode` branches in `scripts/decode.py`.
- **New CLI subcommand:** add a `@click.command` in `cli.py` and `cli.add_command(...)`. If it dispatches to a per-directory script, follow the `call_script` pattern — but note the `dev_scripts/` vs `scripts/` mismatch above.
