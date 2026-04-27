"""
plot_row_k_crash.py

Smoothed log-timescale aligned plot of loss, full_valid_ratio, nan_ratio,
and sample_mem_ratio for all row-K runs — to visualize late-stage crash correlations.

Usage:
  python scripts/plot_row_k_crash.py --outpath /tmp/row_k_crash.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"

EXP_DIRS = [
    "DiT_mini_rowK2_n6_N4096",
    "DiT_mini_rowK3_n6_N4096",
    "DiT_mini_rowVarK15_n6_N4096",
    "DiT_mini_rowVarK34_n6_N4096",
    "DiT_mini_rowVarK0246_n6_N4096",
    "DiT_mini_rowVarK3456_n6_N4096",
    "DiT_mini_globalK15_n6_N4096",
    "DiT_mini_globalK24_n6_N4096",
]
LABELS = [
    "row_k K=2",
    "row_k K=3",
    "row_var K∈{1,5}",
    "row_var K∈{3,4}",
    "row_var K∈{0,2,4,6}",
    "row_var K∈{3,4,5,6}",
    "global_K∈{1,5}",
    "global_K∈{2,4}",
]

TAGS = [
    "train/loss",
    "eval/full_valid_ratio",
    "eval/nan_ratio",
    "eval/sample_mem_ratio",
]
TAG_LABELS = ["Loss", "Full valid ratio", "NaN ratio", "Mem ratio"]


def load_tb_scalars(tb_dir, tags):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    data = {}
    for tag in tags:
        if tag in avail:
            events = ea.Scalars(tag)
            data[tag] = {
                "steps": np.array([e.step for e in events]),
                "vals":  np.array([e.value for e in events]),
            }
    return data


def ewm_smooth(vals, alpha=0.95):
    """Exponential weighted moving average (pandas-style, adjust=False)."""
    out = np.empty_like(vals, dtype=float)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * out[i-1] + (1 - alpha) * vals[i]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", default="/tmp/row_k_crash.png")
    parser.add_argument("--alpha", type=float, default=0.95, help="EWM smoothing alpha")
    parser.add_argument("--saveroot", default=SAVEROOT)
    args = parser.parse_args()

    n_runs = len(EXP_DIRS)
    n_metrics = len(TAGS)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, n_runs))

    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3.2 * n_metrics), sharex=True)
    fig.suptitle("Row-K Rules: Loss / Validity / NaN / Memorization (log steps, smoothed α=0.95)",
                 fontsize=13, fontweight="bold")

    all_data = {}
    for exp in EXP_DIRS:
        tb_dir = os.path.join(args.saveroot, exp, "tensorboard")
        if not os.path.isdir(tb_dir):
            print(f"  [warn] missing: {tb_dir}")
            all_data[exp] = {}
            continue
        all_data[exp] = load_tb_scalars(tb_dir, TAGS)

    for m_idx, (tag, tag_label) in enumerate(zip(TAGS, TAG_LABELS)):
        ax = axes[m_idx]
        for i, (exp, label) in enumerate(zip(EXP_DIRS, LABELS)):
            d = all_data[exp].get(tag)
            if d is None:
                continue
            steps = d["steps"]
            vals  = d["vals"]
            mask  = steps > 0
            steps, vals = steps[mask], vals[mask]
            smoothed = ewm_smooth(vals, alpha=args.alpha)
            # raw trace
            ax.plot(steps, vals, color=colors[i], alpha=0.18, linewidth=0.8)
            # smoothed trace
            ax.plot(steps, smoothed, color=colors[i], alpha=0.92, linewidth=1.6,
                    label=label)
        ax.set_xscale("log")
        ax.set_ylabel(tag_label, fontsize=10)
        ax.grid(alpha=0.2)
        if m_idx == 0:
            ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    axes[-1].set_xlabel("Step", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.outpath, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.outpath}")


if __name__ == "__main__":
    main()
