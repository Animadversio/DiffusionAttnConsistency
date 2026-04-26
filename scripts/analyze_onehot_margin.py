"""
analyze_onehot_margin.py

For a finished one-hot Latin square run, load the last N sample checkpoints
and evaluate accuracy across a range of confidence thresholds (eps values).

The confidence for each cell is:
    confidence = (max_channel - inactive) / (active - inactive)
A cell is flagged ambiguous (nan) if confidence < (1 - eps).

Three metrics are plotted vs eps:
  - nan_ratio        : fraction of samples with ANY ambiguous cell
  - cond_valid_ratio : full validity AMONG non-nan samples (conditioned)
  - net_valid_ratio  : (1 - nan_ratio) * cond_valid_ratio = fraction of ALL samples valid

The net_valid_ratio is the most informative: it penalises both low sharpness
and low rule-correctness, and is independent of the threshold choice.

Usage:
  python scripts/analyze_onehot_margin.py \\
      --exp_name DiT_mini_latinSq_n6_N4096_onehot \\
      --n_size 6 --n_ckpts 10 \\
      --outpath /tmp/onehot_margin.png

  # encoding ablation runs (non-pm1):
  python scripts/analyze_onehot_margin.py \\
      --exp_name DiT_mini_latinSq_n6_N4096_onehot_zeromean_autoSD \\
      --n_size 6 --n_ckpts 10 \\
      --outpath /tmp/onehot_margin_zeromean.png
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.latin_square_lib import check_latin_square_batch

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"


# ---------------------------------------------------------------------------
# Evaluation at a single eps threshold
# ---------------------------------------------------------------------------

def eval_at_eps(samples_oh, eps, active, inactive):
    """
    samples_oh : np.ndarray (N, n, n*n) float
    Returns: nan_ratio, cond_valid_ratio, net_valid_ratio
    """
    N, n, D = samples_oh.shape
    max_act = samples_oh.max(axis=1)                            # (N, D)
    confidence = (max_act - inactive) / (active - inactive)    # (N, D)
    int_vals = samples_oh.argmax(axis=1).astype(float)         # (N, D)
    int_vals[confidence < (1.0 - eps)] = np.nan

    nan_mask = np.isnan(int_vals).any(axis=1)                  # (N,)
    nan_ratio = float(nan_mask.mean())

    valid_int = int_vals[~nan_mask].astype(int)
    M = len(valid_int)
    if M == 0:
        return nan_ratio, 0.0, 0.0

    row_valid, col_valid = check_latin_square_batch(valid_int, n)
    cond_valid_ratio = float((row_valid & col_valid).mean())
    net_valid_ratio = (1.0 - nan_ratio) * cond_valid_ratio
    return nan_ratio, cond_valid_ratio, net_valid_ratio


# ---------------------------------------------------------------------------
# Margin distribution analysis
# ---------------------------------------------------------------------------

def compute_margin_distribution(samples_oh, active, inactive):
    """Returns per-cell confidence values, shape (N*D,)."""
    max_act = samples_oh.max(axis=1)                            # (N, D)
    confidence = (max_act - inactive) / (active - inactive)
    return confidence.ravel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,
                        default="DiT_mini_latinSq_n6_N4096_onehot")
    parser.add_argument("--saveroot", type=str, default=SAVEROOT)
    parser.add_argument("--n_size", type=int, default=6)
    parser.add_argument("--n_ckpts", type=int, default=10,
                        help="Number of latest sample checkpoints to average over")
    parser.add_argument("--eps_values", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9],
                        help="Confidence threshold values to sweep")
    parser.add_argument("--outpath", type=str, default="/tmp/onehot_margin.png")
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.n_size
    savedir = os.path.join(args.saveroot, args.exp_name)
    sample_dir = os.path.join(savedir, "samples")

    # Load encoding params from args.json
    args_json_path = os.path.join(savedir, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path) as f:
            run_args = json.load(f)
        onehot_type = run_args.get("onehot_type", "pm1")
    else:
        onehot_type = "pm1"

    if onehot_type == "pm1":
        active, inactive = 1.0, -1.0
    elif onehot_type == "zero_one":
        active, inactive = 1.0, 0.0
    elif onehot_type == "zero_mean":
        active  = (n - 1) / n
        inactive = -1.0 / n
    else:
        active, inactive = 1.0, -1.0

    print(f"Experiment: {args.exp_name}")
    print(f"Encoding: {onehot_type}, active={active:.4f}, inactive={inactive:.4f}")

    # Find last N sample files
    sample_files = sorted(glob.glob(os.path.join(sample_dir, "samples_epoch_*.pt")))
    if not sample_files:
        print(f"No sample files found in {sample_dir}")
        sys.exit(1)
    sample_files = sample_files[-args.n_ckpts:]
    steps = [int(os.path.basename(f).split("_epoch_")[1].split(".pt")[0])
             for f in sample_files]
    print(f"Using {len(sample_files)} checkpoints: steps {steps[0]}..{steps[-1]}")

    eps_vals = np.array(args.eps_values)

    # Per-checkpoint results: shape (n_ckpts, n_eps, 3)
    all_results = np.zeros((len(sample_files), len(eps_vals), 3))

    for ci, (fpath, step) in enumerate(zip(sample_files, steps)):
        x = torch.load(fpath, map_location="cpu")            # (N, n, n, n)
        N = x.shape[0]
        samples_oh = x.numpy().reshape(N, n, n * n)          # (N, n, n²)
        for ei, eps in enumerate(eps_vals):
            all_results[ci, ei] = eval_at_eps(samples_oh, eps, active, inactive)
        print(f"  step {step:07d}: net_valid@eps0.3={all_results[ci, eps_vals.tolist().index(0.3) if 0.3 in eps_vals.tolist() else -1, 2]:.3f}")

    # Mean over checkpoints
    mean_results = all_results.mean(axis=0)  # (n_eps, 3)

    # Also compute margin distribution for last checkpoint
    x_last = torch.load(sample_files[-1], map_location="cpu")
    samples_last = x_last.numpy().reshape(x_last.shape[0], n, n * n)
    margins = compute_margin_distribution(samples_last, active, inactive)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"{args.exp_name}\n(last {len(sample_files)} ckpts, steps {steps[0]}–{steps[-1]})",
                 fontsize=10)

    labels = ["nan_ratio", "cond_valid_ratio (|non-nan)", "net_valid_ratio (all samples)"]
    ylabels = ["Fraction nan", "Valid ratio (conditioned)", "Valid ratio (net, all samples)"]
    colors = ["tab:red", "tab:blue", "tab:green"]

    for col, (label, ylabel, clr) in enumerate(zip(labels, ylabels, colors)):
        ax = axes[col]
        # Individual checkpoints (faint)
        for ci in range(len(sample_files)):
            ax.plot(eps_vals, all_results[ci, :, col], color=clr, alpha=0.2, lw=1)
        # Mean (bold)
        ax.plot(eps_vals, mean_results[:, col], color=clr, lw=2.5, label="mean")
        ax.set_xlabel("eps (confidence threshold)")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.set_xlim(eps_vals[0], eps_vals[-1])
        ax.set_ylim(0, 1.05)
        ax.axvline(0.3, color="gray", ls="--", lw=1, label="default eps=0.3")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # Margin distribution histogram
    ax = axes[3]
    ax.hist(margins, bins=80, range=(0, 1.1), color="tab:purple", alpha=0.7, density=True)
    ax.axvline(1.0 - 0.3, color="gray", ls="--", lw=1.5, label="eps=0.3 threshold")
    ax.axvline(1.0 - 0.1, color="orange", ls="--", lw=1.5, label="eps=0.1 threshold")
    ax.set_xlabel("Cell confidence = (max_ch − inactive) / (active − inactive)")
    ax.set_ylabel("Density")
    ax.set_title(f"Margin distribution (step {steps[-1]})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(args.outpath, dpi=140, bbox_inches="tight")
    print(f"Saved → {args.outpath}")


if __name__ == "__main__":
    main()
