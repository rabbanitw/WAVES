#!/bin/bash
# R18 trained on ORIGINAL train set + PRE-GEN SDXL 1x10 regens of a
# 2500-pair subset. Regen'd images inherit their source label. Standard CE.
# Eval includes clean and regen'd OOD-COCO.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local seed=$1
    local outdir=runs/regen_aug_r18/s${seed}
    echo
    echo "==================== seed=$seed (R18 + regen-aug) ===================="
    mkdir -p $outdir
    set -o pipefail
    $PY -u regen_aug_train.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        --backbone resnet18 --constant-lr --seed $seed --lr 1e-4 \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep --line-buffered -E '^  ep|^epoch |CLEAN|REGEN|saved best|Error|Traceback|OutOfMemory'
    set +o pipefail
}

for SEED in 44 48 53 55; do
    run $SEED
done

echo
echo "==================== REGEN-AUG SWEEP SUMMARY ===================="
$PY -u <<'EOF'
import json, os, shutil
rows = []
best = None
for name in sorted(os.listdir("runs/regen_aug_r18")):
    summ = f"runs/regen_aug_r18/{name}/summary.json"
    ckpt = f"runs/regen_aug_r18/{name}/best.pt"
    if not os.path.exists(summ): continue
    s = json.load(open(summ))
    rows.append((name, s, ckpt))
    if best is None or s["best_score"] > best[1]["best_score"]:
        best = (name, s, ckpt)

print(f"{'seed':<8s}  {'combined':>10s}")
for name, s, _ in rows:
    print(f"{name:<8s}  {s['best_score']:>+10.4f}")

if best:
    name, s, ckpt = best
    shutil.copy(ckpt, "best_regen_aug_r18.pt")
    print(f"\n-> copied {ckpt} -> detector/best_regen_aug_r18.pt "
          f"(combined = {s['best_score']:+.4f})")
EOF
