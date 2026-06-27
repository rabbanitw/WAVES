#!/bin/bash
# 1-epoch seed sweep with the exact best_ood.pt config (no aug, no LS,
# constant LR=1e-4, deterministic CUDA). Save each run's best-by-OOD-COCO
# ckpt and report the per-seed OOD-COCO / OOD-DiffDB / test_acc table.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local seed=$1
    local outdir=runs/seed_sweep/s${seed}
    echo
    echo "==================== seed=$seed ===================="
    mkdir -p $outdir
    $PY -u train_full.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        --constant-lr --seed $seed \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep -E '^epoch |saved best|in-distribution|OOD-NB|OOD-COCO|^n=|wrote|^opt:|^splits|^single examples'
    echo
}

for SEED in 41 42 43 44 45 46 47 48; do
    run $SEED
done

echo
echo "==================== SEED SWEEP SUMMARY ===================="
$PY -u <<'EOF'
import json, os
print(f"{'seed':<6s}  {'test_acc':>9s}  {'TPR':>7s}  {'TNR':>7s}  {'OOD-DiffDB':>10s}  {'OOD-COCO':>9s}")
for name in sorted(os.listdir("runs/seed_sweep")):
    p = f"runs/seed_sweep/{name}/summary.json"
    if not os.path.exists(p):
        print(f"{name}  -- missing --"); continue
    s = json.load(open(p))
    t, o, c = s["test_id"], s["test_ood"], s["test_ood_coco"]
    print(f"{name:<6s}  {t['acc']:>9.4f}  {t['tpr']:>7.4f}  {t['tnr']:>7.4f}  "
          f"{o['tpr']:>10.4f}  {c['tpr']:>9.4f}")
EOF
