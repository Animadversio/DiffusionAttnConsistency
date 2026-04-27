"""
plot_sigma_loss_evolution.py

Plot DSM loss as a function of training step for different σ bins.

Each panel shows loss vs log(step) for one σ range bin, with three lines:
  train  (blue)  — in-distribution training data
  test   (red)   — valid unseen samples
  random (gray)  — uniform ±1 (no rule)

Also plots the train/test gap (test - train) per σ bin.

Usage:
  python scripts/plot_sigma_loss_evolution.py \\
      --exp_name DiT_mini_rowK2_n6_N4096 \\
      --outpath /tmp/sigma_evolution_rowK2.png

  # Multi-experiment overlay (one figure per bin, one line set per exp):
  python scripts/plot_sigma_loss_evolution.py \\
      --exp_names DiT_mini_rowK2_n6_N4096 DiT_mini_globalK15_n6_N4096 \\
      --outpath /tmp/sigma_evolution_compare.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"

# σ bins: (label, sigma_min, sigma_max)
SIGMA_BINS = [
    ("σ ∈ [0.002, 0.02]",   0.002,  0.02),
    ("σ ∈ [0.02, 0.2]",     0.02,   0.2),
    ("σ ∈ [0.2, 2]",        0.2,    2.0),
    ("σ ∈ [2, 20]",         2.0,    20.0),
    ("σ ∈ [20, 80]",        20.0,   80.0),
]

SPLIT_STYLES = {
    "train":  dict(color="#2166ac", lw=1.8, label="Train (in-dist)"),
    "test":   dict(color="#d73027", lw=1.8, label="Test (valid, unseen)"),
    "random": dict(color="#555555", lw=1.4, ls="--", label="Random ±1"),
}


def load_sigma_data(exp_name):
    """Load all sigma_loss npz files for an experiment, sorted by epoch."""
    sigma_dir = os.path.join(SAVEROOT, exp_name, "sigma_loss")
    if not os.path.isdir(sigma_dir):
        raise FileNotFoundError(f"No sigma_loss dir: {sigma_dir}")

    records = []
    for fname in sorted(os.listdir(sigma_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(sigma_dir, fname)
        d = np.load(path, allow_pickle=True)
        if len(d["sigma_grid"]) < 10:
            continue   # skip incomplete/test runs
        records.append({
            "epoch":        int(d["meta_ckpt_epoch"]),
            "sigma_grid":   d["sigma_grid"],
            "loss_train":   d["loss_train"],
            "loss_test":    d["loss_test"],
            "loss_random":  d["loss_random"],
            "std_train":    d["std_train"],
            "std_test":     d["std_test"],
            "std_random":   d["std_random"],
        })

    records.sort(key=lambda r: r["epoch"])
    return records


def bin_mean(loss, sigma_grid, smin, smax):
    """Average loss over σ values in [smin, smax]."""
    mask = (sigma_grid >= smin) & (sigma_grid <= smax)
    if not mask.any():
        return np.nan
    return float(loss[mask].mean())


def plot_evolution(exp_names, labels, outpath, log_loss=False):
    """Main plotting function."""
    n_bins = len(SIGMA_BINS)
    n_rows = n_bins + 1   # bins + gap panel
    colors_exp = plt.cm.tab10(np.linspace(0, 0.9, len(exp_names)))

    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.8 * n_rows), sharex=True)
    fig.suptitle("DSM Loss vs Training Step — by σ bin", fontsize=13, fontweight="bold")

    all_data = {}
    for exp_name in exp_names:
        try:
            all_data[exp_name] = load_sigma_data(exp_name)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            all_data[exp_name] = []

    for b_idx, (bin_label, smin, smax) in enumerate(SIGMA_BINS):
        ax = axes[b_idx]

        for ei, exp_name in enumerate(exp_names):
            records = all_data[exp_name]
            if not records:
                continue

            steps = np.array([r["epoch"] for r in records])
            sigma_grid = records[0]["sigma_grid"]

            for split in ["train", "test", "random"]:
                vals = np.array([bin_mean(r[f"loss_{split}"], sigma_grid, smin, smax)
                                 for r in records])
                style = SPLIT_STYLES[split].copy()

                if len(exp_names) > 1:
                    # Multi-exp: use exp color, distinguish splits by linestyle
                    style["color"] = colors_exp[ei]
                    if split == "test":
                        style["ls"] = "--"
                    elif split == "random":
                        style["ls"] = ":"
                    lbl = f"{labels[ei]} {split}" if b_idx == 0 else None
                else:
                    lbl = style.pop("label") if b_idx == 0 else style.pop("label", None)

                mask = ~np.isnan(vals)
                steps_plot = steps[mask] + 1  # +1 to avoid log(0) for step=0
                if log_loss:
                    ax.semilogy(steps_plot, vals[mask], **style, label=lbl)
                else:
                    ax.plot(steps_plot, vals[mask], **style, label=lbl)

        ax.set_xscale("log")
        ax.set_ylabel("MSE loss", fontsize=9)
        ax.set_title(bin_label, fontsize=10, pad=2)
        ax.grid(alpha=0.25)
        if b_idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3 if len(exp_names) == 1 else 2)

    # --- Gap panel: test - train, per σ bin ---
    ax_gap = axes[-1]
    bin_colors = plt.cm.plasma(np.linspace(0.05, 0.9, n_bins))

    for b_idx, (bin_label, smin, smax) in enumerate(SIGMA_BINS):
        for ei, exp_name in enumerate(exp_names):
            records = all_data[exp_name]
            if not records:
                continue
            steps = np.array([r["epoch"] for r in records])
            sigma_grid = records[0]["sigma_grid"]
            train_vals = np.array([bin_mean(r["loss_train"], sigma_grid, smin, smax) for r in records])
            test_vals  = np.array([bin_mean(r["loss_test"],  sigma_grid, smin, smax) for r in records])
            gap = test_vals - train_vals

            lw = 1.8 if len(exp_names) == 1 else 1.4
            ls = "-" if ei == 0 else "--"
            lbl = bin_label if (len(exp_names) == 1 or ei == 0) else None
            mask = ~np.isnan(gap)
            ax_gap.plot((steps + 1)[mask], gap[mask],
                        color=bin_colors[b_idx], lw=lw, ls=ls, label=lbl)

    ax_gap.set_xscale("log")
    ax_gap.set_ylabel("Test − Train gap", fontsize=9)
    ax_gap.set_title("Train/Test generalization gap per σ bin", fontsize=10, pad=2)
    ax_gap.set_xlabel("Training step", fontsize=10)
    ax_gap.legend(loc="upper left", fontsize=8, ncol=2)
    ax_gap.grid(alpha=0.25)
    ax_gap.axhline(0, color="k", lw=0.8, ls=":")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved → {outpath}")

    # Also save log-y version
    if not log_loss:
        log_path = outpath.replace(".png", "_logy.png")
        plot_evolution(exp_names, labels, log_path, log_loss=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",  type=str, default=None,
                        help="Single experiment name")
    parser.add_argument("--exp_names", type=str, nargs="+", default=None,
                        help="Multiple experiment names for overlay")
    parser.add_argument("--labels",    type=str, nargs="+", default=None,
                        help="Labels for each experiment (default: exp_name)")
    parser.add_argument("--outpath",   type=str, default="/tmp/sigma_evolution.png")
    args = parser.parse_args()

    if args.exp_names:
        exp_names = args.exp_names
    elif args.exp_name:
        exp_names = [args.exp_name]
    else:
        exp_names = ["DiT_mini_rowK2_n6_N4096"]

    labels = args.labels or exp_names
    plot_evolution(exp_names, labels, args.outpath)


if __name__ == "__main__":
    main()
