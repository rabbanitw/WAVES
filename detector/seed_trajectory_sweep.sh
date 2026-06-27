#!/bin/bash
# Multi-seed mid-epoch trajectory sweep.
# 4 seeds (44, 48, 53, 55) x 2 schedules (constant, cosine) = 8 runs.
# Each: 1 epoch, intra-epoch eval every 100 steps. Best-OOD-COCO ckpt saved.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local seed=$1
    local sched=$2  # "constant" or "cosine"
    local outdir=runs/seed_traj/s${seed}_${sched}
    local sched_flag=""
    [ "$sched" = "constant" ] && sched_flag="--constant-lr"
    echo
    echo "==================== seed=$seed sched=$sched ===================="
    mkdir -p $outdir
    $PY -u train_full.py --epochs 1 --batch 128 --image-size 384 --workers 8 \
        $sched_flag --seed $seed --eval-every-steps 100 \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep -E '^  \[TRAJ\]|^epoch |saved best|OOD-NB|OOD-COCO|^n=' \
        > $outdir/traj.log
    cat $outdir/traj.log
}

for SEED in 44 48 53 55; do
    for SCHED in constant cosine; do
        run $SEED $SCHED
    done
done

echo
echo "==================== SUMMARY ===================="
$PY -u <<'EOF'
import json, os, csv
print(f"{'run':<24s}  {'best_step':>9s}  {'test':>7s}  {'TPR':>6s}  {'TNR':>6s}  {'DiffDB':>6s}  {'COCO':>6s}")
for name in sorted(os.listdir("runs/seed_traj")):
    csv_path = f"runs/seed_traj/{name}/trajectory.csv"
    if not os.path.exists(csv_path): continue
    best_row = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if best_row is None or float(row["ood_coco"]) > float(best_row["ood_coco"]):
                best_row = row
    if best_row:
        print(f"{name:<24s}  {best_row['global_step']:>9s}  "
              f"{float(best_row['test_acc']):>7.4f}  "
              f"{float(best_row['test_tpr']):>6.4f}  "
              f"{float(best_row['test_tnr']):>6.4f}  "
              f"{float(best_row['ood_diffdb']):>6.4f}  "
              f"{float(best_row['ood_coco']):>6.4f}")
EOF
