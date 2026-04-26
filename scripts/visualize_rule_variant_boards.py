"""
visualize_rule_variant_boards.py

Show example boards for the three constraint rule variants:
  - Latin square (row + column constraint)
  - Row-only (row permutation, no column constraint)
  - 6×6 Sudoku (row + column + 2×3 block constraint)

Usage:
  python scripts/visualize_rule_variant_boards.py \
      --n_size 6 --n_examples 4 --seed 42 \
      --outpath /tmp/rule_variants_boards.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.latin_square_lib import (
    sample_latin_square_dataset,
    sample_row_permutation_matrix,
    sample_sudoku_dataset,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["font.family"]  = "sans-serif"

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"


def draw_board(ax, grid, n, is_sudoku=False, block_h=2, block_w=3, colors=None):
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n))
    for r in range(n):
        for c in range(n):
            v = grid[r, c]
            ax.add_patch(patches.Rectangle(
                (c, n - 1 - r), 1, 1,
                facecolor=colors[v], edgecolor="white", lw=0.5,
            ))
            ax.text(c + 0.5, n - 1 - r + 0.5, str(v),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white")
    if is_sudoku:
        n_brow = n // block_h
        n_bcol = n // block_w
        for br in range(n_brow):
            for bc in range(n_bcol):
                ax.add_patch(patches.Rectangle(
                    (bc * block_w, br * block_h), block_w, block_h,
                    fill=False, edgecolor="black", lw=2.8,
                ))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_size", type=int, default=6)
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_h", type=int, default=2)
    parser.add_argument("--block_w", type=int, default=3)
    parser.add_argument("--outpath", type=str,
                        default="/tmp/rule_variants_boards.png")
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.n_size
    k = args.n_examples
    np.random.seed(args.seed)

    print("Sampling Latin square examples...")
    ls_data  = sample_latin_square_dataset(k, n)
    print("Sampling row-only examples...")
    row_data = sample_row_permutation_matrix(k, n)
    print("Sampling 6×6 Sudoku examples...")
    sud_data = sample_sudoku_dataset(k, n, args.block_h, args.block_w)

    datasets = [
        ("Latin square\n(row + col)", ls_data, False),
        ("Row-only\n(row perm, no col)", row_data, False),
        (f"{n}×{n} Sudoku\n(row + col + {args.block_h}×{args.block_w} block)", sud_data, True),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    fig, axes = plt.subplots(3, k, figsize=(3.5 * k, 11))
    fig.suptitle(
        "Rule complexity comparison: Latin square / Row-only / Sudoku\n"
        f"n={n}, {k} random examples each",
        fontsize=13, fontweight="bold",
    )

    for row_idx, (label, data, is_sudoku) in enumerate(datasets):
        for col_idx in range(k):
            ax = axes[row_idx, col_idx]
            grid = data[col_idx].reshape(n, n)
            draw_board(ax, grid, n, is_sudoku=is_sudoku,
                       block_h=args.block_h, block_w=args.block_w,
                       colors=colors)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                               rotation=90, labelpad=10)

    fig.subplots_adjust(hspace=0.12, wspace=0.12)

    base, _ = os.path.splitext(args.outpath)
    plt.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    print(f"Saved → {base}.png  +  {base}.pdf")


if __name__ == "__main__":
    main()
