#!/bin/bash
# Train the detector on PSNR-filtered pairs (near-imperceptible NB edits).
# Two thresholds: 25 dB (~9k pairs) and 30 dB (~2.4k pairs). Each at 3 epochs
# to give the smaller filtered sets enough passes.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local psnr=$1
    local outdir=runs/psnr_filtered/psnr${psnr}
    echo
    echo "==================== min_psnr=$psnr ===================="
    mkdir -p $outdir
    set -o pipefail
    $PY -u psnr_filtered_train.py --epochs 3 --batch 128 --image-size 384 --workers 8 \
        --backbone resnet18 --constant-lr --seed 48 --lr 1e-4 \
        --min-psnr $psnr \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep --line-buffered -E '^  ep|^epoch |DiffDB|saved best|Error|Traceback|OutOfMemory'
    set +o pipefail
}

run 25
run 30

echo
echo "==================== PSNR-FILTERED SUMMARY ===================="
$PY -u <<'EOF'
import json, os
for name in sorted(os.listdir("runs/psnr_filtered")):
    summ = f"runs/psnr_filtered/{name}/summary.json"
    if not os.path.exists(summ): continue
    s = json.load(open(summ))
    print(f"  {name}: min_psnr={s['min_psnr']}  "
          f"n_train_pairs={s['n_train_pairs']}  "
          f"best_bal={s['best_score']:+.4f}")
EOF
