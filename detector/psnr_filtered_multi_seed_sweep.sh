#!/bin/bash
# PSNR>=25 dB filtered training, 4 seeds, 8 epochs, cosine LR.
# Look for a stable operating point in the small-set-substrate regime.

set -e
cd /home/trabbani/WAVES/detector
PY=/home/trabbani/WAVES/venv/bin/python

run() {
    local seed=$1
    local outdir=runs/psnr25_multi/s${seed}
    echo
    echo "==================== seed=$seed (PSNR>=25, 8 epochs, cosine LR) ===================="
    mkdir -p $outdir
    set -o pipefail
    $PY -u psnr_filtered_train.py --epochs 8 --batch 128 --image-size 384 --workers 8 \
        --backbone resnet18 --seed $seed --lr 1e-4 \
        --min-psnr 25 \
        --out-dir $outdir \
        2>&1 | tee $outdir/full.log | grep --line-buffered -E '^  ep|^epoch |DiffDB|saved best|Error|Traceback|OutOfMemory'
    set +o pipefail
}

for SEED in 44 48 53 55; do
    run $SEED
done

echo
echo "==================== PSNR25 MULTI-SEED SUMMARY ===================="
$PY -u <<'EOF'
import json, os, shutil
rows = []
best = None
for name in sorted(os.listdir("runs/psnr25_multi")):
    summ = f"runs/psnr25_multi/{name}/summary.json"
    ckpt = f"runs/psnr25_multi/{name}/best.pt"
    if not os.path.exists(summ): continue
    s = json.load(open(summ))
    rows.append((name, s, ckpt))
    if best is None or s["best_score"] > best[1]["best_score"]:
        best = (name, s, ckpt)

print(f"{'seed':<8s}  {'best_bal':>10s}")
for name, s, _ in rows:
    print(f"{name:<8s}  {s['best_score']:>+10.4f}")

if best:
    name, s, ckpt = best
    shutil.copy(ckpt, "best_psnr25_r18.pt")
    print(f"\n-> copied {ckpt} -> detector/best_psnr25_r18.pt "
          f"(best_bal = {s['best_score']:+.4f})")
EOF
