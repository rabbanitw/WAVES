"""Compose multi-seed training trajectory plots from the per-step CSVs
saved by train_full.py runs with --eval-every-steps.

Reads trajectory.csv files from each run dir under runs/<group>/ and
plots one figure per metric, one line per run. Saves PNG + a single
consolidated long-form CSV for downstream analysis."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("test_acc", "In-dist test accuracy"),
    ("test_tpr", "In-dist TPR (NB-edit catch)"),
    ("test_tnr", "In-dist TNR (natural correct-reject)"),
    ("ood_diffdb", "OOD-DiffDB detection rate"),
    ("ood_coco_edit", "OOD-COCO photoreal-edit detection"),
    ("coco_nat_tnr", "OOD-COCO natural correct-reject"),
    ("balanced", "Balanced score (COCO-edit + COCO-nat-TNR - 1)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-dir", default="runs/seed_traj_bal",
                    help="parent dir containing one subdir per run")
    ap.add_argument("--out", default="runs/seed_traj_bal/trajectory_plots")
    args = ap.parse_args()

    group = Path(args.group_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Collect trajectories
    runs = {}  # name -> list[dict]
    for sub in sorted(group.iterdir()):
        csv_path = sub / "trajectory.csv"
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) if k not in ("global_step", "samples_seen", "elapsed_s")
                             else int(float(v))
                             for k, v in r.items()})
        runs[sub.name] = rows
        print(f"loaded {sub.name}: {len(rows)} trajectory points")

    if not runs:
        print(f"no trajectory.csv files under {group}/")
        return

    # 1. Long-form consolidated CSV
    long_csv = out / "all_trajectories.csv"
    with open(long_csv, "w", newline="") as f:
        w = csv.writer(f)
        sample_keys = list(next(iter(runs.values()))[0].keys())
        w.writerow(["run"] + sample_keys)
        for name, rows in runs.items():
            for r in rows:
                w.writerow([name] + [r[k] for k in sample_keys])
    print(f"\nwrote {long_csv}")

    # 2. One plot per metric
    for metric, ylabel in METRICS:
        if metric not in next(iter(runs.values()))[0]:
            print(f"skipping {metric} (not in trajectories)")
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, rows in runs.items():
            xs = [r["global_step"] for r in rows]
            ys = [r[metric] for r in rows]
            ax.plot(xs, ys, marker=".", label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if metric == "balanced":
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        png = out / f"{metric}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"wrote {png}")

    # 3. Composite 2x4 panel for the most useful metrics
    panel_metrics = ["test_acc", "test_tpr", "test_tnr",
                     "ood_diffdb", "ood_coco_edit", "coco_nat_tnr",
                     "balanced"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for i, metric in enumerate(panel_metrics):
        ax = axes[i]
        for name, rows in runs.items():
            xs = [r["global_step"] for r in rows]
            ys = [r[metric] for r in rows]
            ax.plot(xs, ys, marker=".", markersize=4, label=name, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_title(dict(METRICS)[metric], fontsize=10)
        ax.grid(True, alpha=0.3)
        if metric == "balanced":
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")
    axes[-1].axis("off")  # 7 metrics, 8 slots
    fig.suptitle(f"Training-trajectory comparison across seeds ({group.name})",
                 fontsize=14)
    fig.tight_layout()
    composite = out / "composite_panel.png"
    fig.savefig(composite, dpi=120)
    plt.close(fig)
    print(f"\nwrote {composite}")


if __name__ == "__main__":
    main()
