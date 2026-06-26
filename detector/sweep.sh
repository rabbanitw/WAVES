#!/bin/bash
# Hyperparameter sweep, 1-epoch budget, with fixed augmentation pipeline.
# cfg0 and cfg2 already completed in the previous (broken) sweep — they
# don't depend on aug so we leave their results in place and skip.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local name=$1; shift
    local outdir=runs/sweep/$name
    echo
    echo "==================== $name ===================="
    mkdir -p $outdir
    $PY -u train_full.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        --out-dir $outdir "$@" \
        2>&1 | tee $outdir/full.log | grep -E '^epoch |^  ep|wrote|saved best|in-distribution|OOD synthetic|n=|loaded |splits|^opt:'
    echo
}

# Re-run cfg1 with fixed aug
run cfg1_aug                      --aug

# Re-run aug+LS combinations
run cfg3_aug_ls10                 --aug --label-smoothing 0.1
run cfg4_aug_ls10_lr3e4_warm      --aug --label-smoothing 0.1 --lr 3e-4 --warmup-pct 0.1
run cfg5_aug_ls10_lr3e4_wd1e3     --aug --label-smoothing 0.1 --lr 3e-4 --weight-decay 1e-3
run cfg6_aug_ls10_lr3e4_resnet50  --aug --label-smoothing 0.1 --lr 3e-4 --backbone resnet50

# Constant-LR: try to recapture the "undertrained ep0" OOD sweet spot
run cfg7_constant_lr1e4           --constant-lr
run cfg8_constant_lr3e4_warm      --constant-lr --lr 3e-4 --warmup-pct 0.05

echo
echo "==================== sweep summary ===================="
$PY -u <<'EOF'
import json, os
rows = []
for name in sorted(os.listdir("runs/sweep")):
    path = f"runs/sweep/{name}/summary.json"
    if not os.path.exists(path):
        rows.append((name, None)); continue
    with open(path) as f:
        s = json.load(f)
    rows.append((name, s))
print(f"{'config':<34s}  {'test_acc':>9s}  {'TPR':>7s}  {'TNR':>7s}  {'OOD':>7s}")
for name, s in rows:
    if s is None:
        print(f"{name:<34s}  -- failed --"); continue
    t = s["test_id"]; o = s["test_ood"]
    print(f"{name:<34s}  {t['acc']:>9.4f}  {t['tpr']:>7.4f}  {t['tnr']:>7.4f}  {o['tpr']:>7.4f}")
EOF
