#!/bin/bash
# Second seed-sweep batch: seeds 49..56
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
        2>&1 | tee $outdir/full.log | grep -E '^epoch |saved best|OOD-NB|OOD-COCO|^n='
}

for SEED in 57 58 59 60 61 62 63 64; do
    run $SEED
done

echo
echo "==================== ALL SEEDS SUMMARY ===================="
$PY -u <<'EOF'
import json, os
print(f"{'seed':<6s}  {'test_acc':>9s}  {'TPR':>7s}  {'TNR':>7s}  {'OOD-DiffDB':>10s}  {'OOD-COCO':>9s}")
rows = []
for name in sorted(os.listdir("runs/seed_sweep")):
    p = f"runs/seed_sweep/{name}/summary.json"
    if not os.path.exists(p): continue
    s = json.load(open(p))
    t, o, c = s["test_id"], s["test_ood"], s["test_ood_coco"]
    rows.append((name, t, o, c))
# Sort by OOD-COCO descending
rows.sort(key=lambda r: -r[3]["tpr"])
for name, t, o, c in rows:
    print(f"{name:<6s}  {t['acc']:>9.4f}  {t['tpr']:>7.4f}  {t['tnr']:>7.4f}  "
          f"{o['tpr']:>10.4f}  {c['tpr']:>9.4f}")
EOF
