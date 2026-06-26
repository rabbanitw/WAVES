#!/bin/bash
# Determinism + seed-variance check.
# Runs cfg0 (cosine over 1 epoch) and cfg7 (constant LR) at two seeds each
# with full deterministic-CUDA flags enabled.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local name=$1; shift
    local outdir=runs/sweep_det/$name
    echo
    echo "==================== $name ===================="
    mkdir -p $outdir
    $PY -u train_full.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        --out-dir $outdir "$@" \
        2>&1 | tee $outdir/full.log | grep -E '^epoch |wrote|in-distribution|OOD synthetic|n=|loaded |splits|^opt:'
    echo
}

run cfg0_baseline_s42       --seed 42
run cfg0_baseline_s43       --seed 43
run cfg7_constlr_s42        --constant-lr --seed 42
run cfg7_constlr_s43        --constant-lr --seed 43

echo
echo "==================== determinism summary ===================="
$PY -u <<'EOF'
import json, os
print(f"{'config':<28s}  {'test':>8s}  {'TPR':>7s}  {'TNR':>7s}  {'OOD':>7s}")
for name in sorted(os.listdir("runs/sweep_det")):
    path = f"runs/sweep_det/{name}/summary.json"
    if not os.path.exists(path):
        print(f"{name:<28s}  -- missing --"); continue
    s = json.load(open(path))
    t, o = s["test_id"], s["test_ood"]
    print(f"{name:<28s}  {t['acc']:>8.4f}  {t['tpr']:>7.4f}  {t['tnr']:>7.4f}  {o['tpr']:>7.4f}")
EOF
