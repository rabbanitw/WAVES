"""Plot DP-FedAvg memorization sweep results."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RUNS = Path("/home/trabbani/fedavg_exp/runs")
OUT = RUNS / "figures"
OUT.mkdir(exist_ok=True)


def load_runs():
    cfg = json.loads((RUNS / "config.json").read_text())
    runs = {}
    for jf in sorted(RUNS.glob("K*.json")):
        h = json.loads(jf.read_text())
        runs[h["K"]] = h
    return cfg, runs


def fig1_per_class(cfg, runs):
    """Panels: per-class final train and test accuracy, classes sorted by frequency."""
    Ks = sorted(runs.keys())
    counts = np.array(cfg["counts"])
    order = np.argsort(-counts)  # head -> tail
    counts_sorted = counts[order]

    n = len(Ks)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, K in enumerate(Ks):
        ax = axes.flat[i]
        h = runs[K]
        tr = np.array(h["train_acc_by_class"][-1])[order]
        te = np.array(h["test_acc_by_class"][-1])[order]
        x = np.arange(len(tr))
        ax.plot(x, tr, "o-", color="#1f77b4", lw=1.4, ms=3, label="train")
        ax.plot(x, te, "s-", color="#d62728", lw=1.4, ms=3, label="test")
        ax.fill_between(x, te, tr, where=tr >= te, alpha=0.15, color="#1f77b4",
                        label="memorization gap")
        ax.set_title(f"K = {K}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i % cols == 0:
            ax.set_ylabel("accuracy")
        if i // cols == rows - 1:
            ax.set_xlabel("class rank (head $\\rightarrow$ tail)")
    for j in range(n, rows * cols):
        axes.flat[j].axis("off")
    fig.suptitle("Per-class accuracy at final round (long-tailed Tiny-ImageNet, "
                 f"head={counts_sorted[0]}, tail={counts_sorted[-1]})", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_per_class_acc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_gap_vs_size(cfg, runs):
    """Memorization gap (train - test) vs class frequency, lines per K."""
    Ks = sorted(runs.keys())
    counts = np.array(cfg["counts"])
    order = np.argsort(-counts)
    counts_sorted = counts[order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    for i, K in enumerate(Ks):
        h = runs[K]
        tr = np.array(h["train_acc_by_class"][-1])[order]
        te = np.array(h["test_acc_by_class"][-1])[order]
        gap = tr - te
        c = cmap(i / max(1, len(Ks) - 1))
        # left: gap vs class rank
        axes[0].plot(np.arange(len(gap)), gap, "o-", color=c,
                     ms=3, lw=1.3, label=f"K={K}")
        # right: gap vs class size (log x)
        axes[1].plot(counts_sorted, gap, "o-", color=c, ms=3, lw=1.3,
                     label=f"K={K}")
    axes[0].set_xlabel("class rank (head $\\rightarrow$ tail)")
    axes[0].set_ylabel("train acc $-$ test acc")
    axes[0].set_title("Memorization gap by class rank")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("class size in training set")
    axes[1].set_ylabel("train acc $-$ test acc")
    axes[1].set_title("Memorization gap vs class size")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Averaging across $K$ clients reduces per-class memorization", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_summary(cfg, runs):
    """Tail vs head memorization gap summary as a function of K."""
    Ks = sorted(runs.keys())
    counts = np.array(cfg["counts"])
    order = np.argsort(-counts)
    n = len(counts)
    head_idx = order[: n // 5]
    tail_idx = order[-n // 5 :]
    head_gap, tail_gap, mean_gap = [], [], []
    head_train, head_test, tail_train, tail_test = [], [], [], []
    for K in Ks:
        h = runs[K]
        tr = np.array(h["train_acc_by_class"][-1])
        te = np.array(h["test_acc_by_class"][-1])
        head_gap.append((tr[head_idx] - te[head_idx]).mean())
        tail_gap.append((tr[tail_idx] - te[tail_idx]).mean())
        mean_gap.append((tr - te).mean())
        head_train.append(tr[head_idx].mean())
        head_test.append(te[head_idx].mean())
        tail_train.append(tr[tail_idx].mean())
        tail_test.append(te[tail_idx].mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(Ks, head_gap, "o-", label=f"head 20% ({len(head_idx)} classes)",
                 color="#1f77b4")
    axes[0].plot(Ks, tail_gap, "s-", label=f"tail 20% ({len(tail_idx)} classes)",
                 color="#d62728")
    axes[0].plot(Ks, mean_gap, "^--", label="all classes", color="0.3", lw=1)
    axes[0].set_xlabel("number of clients $K$")
    axes[0].set_ylabel("mean (train acc $-$ test acc)")
    axes[0].set_title("Memorization gap vs $K$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(Ks, head_train, "o-", color="#1f77b4", label="head train")
    axes[1].plot(Ks, head_test, "o--", color="#1f77b4", label="head test")
    axes[1].plot(Ks, tail_train, "s-", color="#d62728", label="tail train")
    axes[1].plot(Ks, tail_test, "s--", color="#d62728", label="tail test")
    axes[1].set_xlabel("number of clients $K$")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Train vs test accuracy on head and tail")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_summary_vs_K.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_client_heatmap(cfg, runs):
    """For K=10 (or largest K), heatmap of per-client train acc by class
    (global model eval'd on each client's local train data, by class)."""
    Ks = sorted(runs.keys())
    K = max(Ks)
    h = runs[K]
    counts = np.array(cfg["counts"])
    order = np.argsort(-counts)
    grid = np.array(h["client_train_acc_by_class_final"])[:, order]
    client_class_counts = np.array(h.get("client_class_counts", [[0]]))[
        :, order] if h.get("client_class_counts") else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    im0 = axes[0].imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[0].set_xlabel("class rank (head $\\rightarrow$ tail)")
    axes[0].set_ylabel("client index")
    axes[0].set_title(f"Global model train accuracy on each client's data (K={K})")
    fig.colorbar(im0, ax=axes[0], label="accuracy")

    if client_class_counts is not None:
        im1 = axes[1].imshow(client_class_counts, aspect="auto", cmap="Blues",
                             norm=plt.matplotlib.colors.LogNorm(
                                 vmin=max(1, client_class_counts[client_class_counts > 0].min()
                                          if (client_class_counts > 0).any() else 1),
                                 vmax=max(1, client_class_counts.max())))
        axes[1].set_xlabel("class rank (head $\\rightarrow$ tail)")
        axes[1].set_ylabel("client index")
        axes[1].set_title("# samples per (client, class)")
        fig.colorbar(im1, ax=axes[1], label="count (log)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_client_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_progression(cfg, runs):
    """Train and test acc trajectories per K (mean across classes)."""
    Ks = sorted(runs.keys())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, K in enumerate(Ks):
        h = runs[K]
        rounds = np.array(h["rounds"])
        tr = np.array(h["train_acc_by_class"]).mean(axis=1)
        te = np.array(h["test_acc_by_class"]).mean(axis=1)
        c = cmap(i / max(1, len(Ks) - 1))
        axes[0].plot(rounds, tr, "-", color=c, label=f"K={K}")
        axes[1].plot(rounds, te, "-", color=c, label=f"K={K}")
    for ax, lbl in zip(axes, ("mean train accuracy", "mean test accuracy")):
        ax.set_xlabel("round")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_title("Training trajectory")
    axes[1].set_title("Test trajectory")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg, runs = load_runs()
    print("Ks loaded:", sorted(runs.keys()))
    fig1_per_class(cfg, runs)
    fig2_gap_vs_size(cfg, runs)
    fig3_summary(cfg, runs)
    fig4_client_heatmap(cfg, runs)
    fig5_progression(cfg, runs)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
