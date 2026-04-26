"""
visualize_error_samples.py

Show example output boards with errors alongside their spatial uncertainty maps.
For each example: decoded integer grid (with error cells highlighted) + uncertainty heatmap.

Usage:
  python scripts/visualize_error_samples.py \\
      --exp_name DiT_mini_latinSq_n6_N4096_onehot \\
      --n_size 6 --n_examples 3 \\
      --outpath /tmp/error_examples.png

  python scripts/visualize_error_samples.py \\
      --exp_name DiT_mini_latinSq_n6_N4096_scalar \\
      --encoding scalar --n_size 6 --n_examples 3 \\
      --outpath /tmp/error_examples_scalar.png
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.analyze_error_cell_confidence import (
    find_error_cells, onehot_cell_confidence, scalar_cell_confidence
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["font.family"]  = "sans-serif"

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"


def get_encoding_params(savedir, n):
    args_json = os.path.join(savedir, "args.json")
    onehot_type = "pm1"
    encoding = "onehot"
    if os.path.exists(args_json):
        with open(args_json) as f:
            run_args = json.load(f)
        onehot_type = run_args.get("onehot_type", "pm1")
        encoding    = run_args.get("encoding", "onehot")
    if onehot_type == "pm1":
        return encoding, 1.0, -1.0
    elif onehot_type == "zero_one":
        return encoding, 1.0, 0.0
    elif onehot_type == "zero_mean":
        return encoding, (n - 1) / n, -1.0 / n
    return encoding, 1.0, -1.0


def draw_example(axes_row, decoded_grid, uncertainty_grid, error_mask_grid, n,
                 title="", step=0):
    """
    Draw one example: left = decoded board, right = uncertainty heatmap.

    Parameters
    ----------
    axes_row   : list of 2 Axes
    decoded_grid    : (n, n) int
    uncertainty_grid: (n, n) float, 1−confidence per cell
    error_mask_grid : (n, n) bool
    """
    ax_board, ax_unc = axes_row

    # --- Decoded board ---
    im = ax_board.imshow(decoded_grid, cmap="tab10", vmin=-0.5, vmax=n - 0.5,
                         interpolation="nearest")
    for r in range(n):
        for c in range(n):
            val = decoded_grid[r, c]
            color = "white" if error_mask_grid[r, c] else "black"
            weight = "bold" if error_mask_grid[r, c] else "normal"
            ax_board.text(c, r, str(val), ha="center", va="center",
                          fontsize=12, color=color, fontweight=weight)

    # Highlight error cells with red border
    for r in range(n):
        for c in range(n):
            if error_mask_grid[r, c]:
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                         linewidth=2.5, edgecolor="red",
                                         facecolor="none")
                ax_board.add_patch(rect)

    ax_board.set_xticks([])
    ax_board.set_yticks([])
    ax_board.set_title(f"{title}\nstep {step}", fontsize=9)
    ax_board.spines[:].set_visible(False)

    # --- Uncertainty map ---
    unc_max = max(uncertainty_grid.max(), 0.02)   # avoid all-zero colormap
    im2 = ax_unc.imshow(uncertainty_grid, cmap="Reds", vmin=0, vmax=unc_max,
                        interpolation="nearest")
    plt.colorbar(im2, ax=ax_unc, fraction=0.046, pad=0.04)
    for r in range(n):
        for c in range(n):
            val = uncertainty_grid[r, c]
            color = "white" if val > unc_max * 0.6 else "black"
            ax_unc.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color)
    # Mark error cells with red border
    for r in range(n):
        for c in range(n):
            if error_mask_grid[r, c]:
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                         linewidth=2.5, edgecolor="red",
                                         facecolor="none")
                ax_unc.add_patch(rect)
    ax_unc.set_xticks([])
    ax_unc.set_yticks([])
    ax_unc.set_title("Uncertainty map\n(red = error cells)", fontsize=9)
    ax_unc.spines[:].set_visible(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,
                        default="DiT_mini_latinSq_n6_N4096_onehot")
    parser.add_argument("--saveroot", type=str, default=SAVEROOT)
    parser.add_argument("--n_size", type=int, default=6)
    parser.add_argument("--n_examples", type=int, default=3,
                        help="Number of invalid examples to show")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outpath", type=str, default="/tmp/error_examples.png")
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.n_size
    savedir = os.path.join(args.saveroot, args.exp_name)
    sample_dir = os.path.join(savedir, "samples")

    encoding, active, inactive = get_encoding_params(savedir, n)

    # Load last checkpoint
    files = sorted(glob.glob(os.path.join(sample_dir, "samples_epoch_*.pt")))
    if not files:
        print(f"No sample files in {sample_dir}")
        sys.exit(1)
    fpath = files[-1]
    step  = int(os.path.basename(fpath).split("_epoch_")[1].split(".pt")[0])
    print(f"Loading {fpath}  (step {step})")

    x = torch.load(fpath, map_location="cpu", weights_only=False)
    N = x.shape[0]

    if encoding == "onehot":
        samples = x.numpy().reshape(N, n, n * n)
        conf, decoded = onehot_cell_confidence(samples, active, inactive)
    else:
        samples = x.numpy().reshape(N, n * n)
        conf, decoded, _ = scalar_cell_confidence(samples, n)

    uncertainty = 1.0 - conf                          # (N, n²)
    err_mask    = find_error_cells(decoded, n)         # (N, n²)

    # Find samples with errors (invalid)
    has_error = err_mask.any(axis=1)                  # (N,)
    error_idx = np.where(has_error)[0]
    print(f"Found {len(error_idx)} invalid samples out of {N}")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(error_idx, size=min(args.n_examples, len(error_idx)),
                        replace=False)

    # ---------------------------------------------------------------------------
    # Plot: n_examples rows × 2 cols
    # ---------------------------------------------------------------------------
    n_ex = len(chosen)
    fig, axes = plt.subplots(n_ex, 2, figsize=(8, 4.2 * n_ex))
    if n_ex == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.45, wspace=0.25)

    for row_idx, sample_idx in enumerate(chosen):
        dec_grid = decoded[sample_idx].reshape(n, n)
        unc_grid = uncertainty[sample_idx].reshape(n, n)
        err_grid = err_mask[sample_idx].reshape(n, n)
        n_err    = err_grid.sum()

        draw_example(
            axes[row_idx], dec_grid, unc_grid, err_grid, n,
            title=f"Sample #{sample_idx}  ({n_err} error cells)",
            step=step,
        )

    enc_label = f"{encoding}" + (f" ({active:.2f}/{inactive:.2f})" if encoding == "onehot" else "")
    fig.suptitle(
        f"{args.exp_name}\n"
        f"Encoding: {enc_label}  |  Red border = row/col violation",
        fontsize=10,
    )

    base, _ = os.path.splitext(args.outpath)
    plt.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    print(f"Saved → {base}.png  +  {base}.pdf")


if __name__ == "__main__":
    main()
