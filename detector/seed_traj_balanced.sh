#!/bin/bash
# Top seeds (44, 48, 53, 55) x constant LR, eval every 100 steps, save best by
# balanced score = OOD-COCO + COCO-nat-TNR - 1.
set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local seed=$1
    local outdir=runs/seed_traj_bal/s${seed}_constant
    echo
    echo "==================== seed=$seed ===================="
    mkdir -p $outdir
    $PY -u train_full.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        --constant-lr --seed $seed --eval-every-steps 100 \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep --line-buffered -E '^  \[TRAJ\]|^epoch |saved best|^---|n=496|n=496|wrote'
    echo
}

for SEED in 44 48 53 55; do
    run $SEED
done

echo
echo "==================== BALANCED SWEEP SUMMARY ===================="
$PY -u <<'EOF'
import json, os, csv
print(f"{'run':<22s}  {'best_step':>9s}  {'test':>6s}  {'TPR':>6s}  {'TNR':>6s}  {'DiffDB':>7s}  {'COCO-edit':>9s}  {'COCO-nat-TNR':>13s}  {'balanced':>9s}")
for name in sorted(os.listdir("runs/seed_traj_bal")):
    csv_path = f"runs/seed_traj_bal/{name}/trajectory.csv"
    if not os.path.exists(csv_path): continue
    best_row = None
    for r in csv.DictReader(open(csv_path)):
        if best_row is None or float(r["balanced"]) > float(best_row["balanced"]):
            best_row = r
    if best_row:
        print(f"{name:<22s}  {best_row['global_step']:>9s}  "
              f"{float(best_row['test_acc']):>6.3f}  {float(best_row['test_tpr']):>6.3f}  "
              f"{float(best_row['test_tnr']):>6.3f}  {float(best_row['ood_diffdb']):>7.3f}  "
              f"{float(best_row['ood_coco_edit']):>9.3f}  {float(best_row['coco_nat_tnr']):>13.3f}  "
              f"{float(best_row['balanced']):>+9.4f}")
EOF
