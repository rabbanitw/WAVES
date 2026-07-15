#!/bin/bash
# R18 + mixed-batch + noise-aug + GAUSSIAN BLUR PREPROCESSING.
# Blur sigma=1.0 baked into the classifier (both trainee and attacker).
# Adversarial perturbations must SURVIVE blur, i.e. live in low frequencies
# the natural-image distribution actually occupies.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local seed=$1
    local outdir=runs/adv_blur_r18/s${seed}
    echo
    echo "==================== seed=$seed (R18 + blur sigma=1.0) ===================="
    mkdir -p $outdir
    set -o pipefail
    $PY -u adv_train.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        --backbone resnet18 --constant-lr --seed $seed --lr 1e-4 \
        --eps-ladder 1,2,4,8,16 --adv-k 3 --adv-alpha-units 1.0 \
        --warmup-clean-steps 200 \
        --attacker-ckpt /home/trabbani/WAVES/detector/best_ood_balanced_r50.pt \
        --attacker-backbone resnet50 \
        --mixed-batch \
        --noise-aug-sigma 0.025 \
        --blur-sigma 1.0 \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep --line-buffered -E '^  ep|^epoch |saved best|per-eps|^---|wrote|Error|Traceback|OutOfMemory|WRAPPING'
    set +o pipefail
}

for SEED in 44 48 53 55; do
    run $SEED
done

echo
echo "==================== BLUR SWEEP SUMMARY ===================="
$PY -u <<'EOF'
import json, os, shutil
rows = []
best = None
for name in sorted(os.listdir("runs/adv_blur_r18")):
    summ = f"runs/adv_blur_r18/{name}/summary.json"
    ckpt = f"runs/adv_blur_r18/{name}/best.pt"
    if not os.path.exists(summ): continue
    s = json.load(open(summ))
    rows.append((name, s, ckpt))
    if best is None or s["best_score"] > best[1]["best_score"]:
        best = (name, s, ckpt)

print(f"{'seed':<8s}  {'expanded_bal':>12s}")
for name, s, _ in rows:
    print(f"{name:<8s}  {s['best_score']:>+12.4f}")

if best:
    name, s, ckpt = best
    shutil.copy(ckpt, "best_adv_blur_r18.pt")
    print(f"\n-> copied {ckpt} -> detector/best_adv_blur_r18.pt "
          f"(expanded-bal = {s['best_score']:+.4f})")
EOF
