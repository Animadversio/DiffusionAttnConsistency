"""
plot_tb_curves.py

Reusable script to plot TensorBoard training curves for any set of experiments.

Usage examples:
  # Exact-K sweep
  python scripts/plot_tb_curves.py \
    --exp_dirs DiT_mini_exactK_N4096_D36_K3 DiT_mini_exactK_N4096_D36_K6 ... \
    --labels "K=3" "K=6" ... \
    --title "DiT-mini Exact-K Learning" \
    --outpath /tmp/exactK_curves.png

  # Latin square sweep
  python scripts/plot_tb_curves.py \
    --exp_dirs DiT_mini_latinSq_n5_N4096_scalar DiT_mini_latinSq_n6_N4096_scalar ... \
    --labels "n=5 scalar" "n=6 scalar" ... \
    --title "DiT-mini Latin Square Learning" \
    --outpath /tmp/latinsq_curves.png

  # Auto-discover by prefix
  python scripts/plot_tb_curves.py \
    --prefix DiT_mini_exactK_N4096_D36_K \
    --title "DiT-mini Exact-K" \
    --outpath /tmp/exactK_curves.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"


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
                "steps": [e.step for e in events],
                "vals":  [e.value for e in events],
            }
    return data


def discover_experiments(saveroot, prefix):
    """Find all exp dirs matching prefix, sorted by name."""
    dirs = sorted([
        d for d in os.listdir(saveroot)
        if d.startswith(prefix) and os.path.isdir(os.path.join(saveroot, d))
    ])
    return dirs


def make_label(exp_name, prefix):
    """Strip common prefix to make a short label."""
    return exp_name[len(prefix):] if exp_name.startswith(prefix) else exp_name


def plot_curves(exp_dirs, labels, tags, title, outpath, saveroot=SAVEROOT,
                log_x=True, ncols=3, figsize=None):
    """
    Load TensorBoard data and plot one panel per tag.

    Parameters
    ----------
    exp_dirs : list of str, experiment folder names (relative to saveroot)
    labels   : list of str, legend labels (same length as exp_dirs)
    tags     : list of str, TensorBoard scalar tags to plot
    title    : str
    outpath  : str, output image path
    saveroot : str
    log_x    : bool, use log scale on x-axis
    ncols    : int
    figsize  : tuple or None
    """
    n_panels = len(tags)
    nrows = (n_panels + ncols - 1) // ncols
    if figsize is None:
        figsize = (5.5 * ncols, 4.2 * nrows)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(exp_dirs), 2)))

    # Load all data
    all_data = {}
    for exp in exp_dirs:
        tb_dir = os.path.join(saveroot, exp, "tensorboard")
        if not os.path.isdir(tb_dir):
            print(f"  [warn] no tensorboard dir: {tb_dir}")
            all_data[exp] = {}
            continue
        all_data[exp] = load_tb_scalars(tb_dir, tags)

    # Print latest values
    for exp, label in zip(exp_dirs, labels):
        d = all_data[exp]
        parts = []
        for tag in tags:
            if tag in d and d[tag]["vals"]:
                parts.append(f"{tag.split('/')[-1]}={d[tag]['vals'][-1]:.4f}@{d[tag]['steps'][-1]}")
        print(f"  {label}: {' | '.join(parts) if parts else 'no data'}")

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.42, wspace=0.32)

    for p_idx, tag in enumerate(tags):
        row, col = divmod(p_idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        for i, (exp, label) in enumerate(zip(exp_dirs, labels)):
            d = all_data[exp].get(tag)
            if d and d["vals"]:
                ax.plot(d["steps"], d["vals"], color=colors[i], label=label,
                        linewidth=1.8, marker="o", markersize=2, alpha=0.9)
        tag_short = tag.replace("eval/", "").replace("train/", "").replace("_", " ")
        ax.set_title(tag_short, fontsize=11)
        ax.set_xlabel("Step")
        if log_x and any(
            all_data[exp].get(tag, {}).get("steps", [0])[-1] > 10
            for exp in exp_dirs
        ):
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    # Hide unused panels
    for p_idx in range(n_panels, nrows * ncols):
        row, col = divmod(p_idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Legend on first panel
    fig.get_axes()[0].legend(fontsize=8, ncol=2, loc="best")

    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    print(f"Saved → {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Pre-baked experiment groups
# ---------------------------------------------------------------------------

GROUPS = {
    "exactK": {
        "prefix": "DiT_mini_exactK_N4096_D36_K",
        "tags": [
            "train/loss",
            "eval/k_correct_ratio",
            "eval/sample_mem_ratio",
            "eval/nan_ratio_eps_1e-1",
            "eval/mean_ones",
            "eval/nan_ratio_eps_1e-2",
        ],
        "title": "DiT-mini Exact-K Learning (N=4096, D=36)",
    },
    "latinsq": {
        "prefix": "DiT_mini_latinSq_",
        "tags": [
            "train/loss",
            "eval/full_valid_ratio",
            "eval/row_valid_ratio",
            "eval/col_valid_ratio",
            "eval/sample_mem_ratio",
            "eval/nan_ratio_permissive",
        ],
        "title": "DiT-mini Latin Square Learning (N=4096)",
    },
    "latinsq_B": {
        "exp_dirs": [
            "DiT_B_latinSq_n5_N4096_scalar",
            "DiT_B_latinSq_n5_N4096_onehot",
            "DiT_B_latinSq_n6_N4096_scalar",
            "DiT_B_latinSq_n6_N4096_onehot",
        ],
        "tags": [
            "train/loss",
            "eval/full_valid_ratio",
            "eval/row_valid_ratio",
            "eval/col_valid_ratio",
            "eval/sample_mem_ratio",
            "eval/nan_ratio_permissive",
        ],
        "title": "DiT-B Latin Square Learning (N=4096, 12L 12H 768D)",
    },
    "latinsq_rules": {
        "exp_dirs": [
            "DiT_mini_latinSq_n6_N4096_onehot",
            "DiT_mini_rowOnly_n6_N4096_onehot",
            "DiT_mini_sudoku6x6_N4096_onehot",
        ],
        "tags": [
            "train/loss",
            "eval/full_valid_ratio",
            "eval/row_valid_ratio",
            "eval/col_valid_ratio",
            "eval/block_valid_ratio",
            "eval/nan_ratio_permissive",
        ],
        "title": "DiT-mini n=6 Rule Complexity Comparison (onehot)",
        "labels": ["Latin square (row+col)", "Row-only", "Sudoku 6×6 (row+col+block)"],
    },
    "latinsq_encoding": {
        "exp_dirs": [
            "DiT_mini_latinSq_n6_N4096_onehot",
            "DiT_mini_latinSq_n6_N4096_onehot_zeromean_autoSD",
            "DiT_mini_latinSq_n6_N4096_onehot_zeroone_autoSD",
        ],
        "tags": [
            "train/loss",
            "eval/full_valid_ratio",
            "eval/row_valid_ratio",
            "eval/col_valid_ratio",
            "eval/sample_mem_ratio",
            "eval/nan_ratio_permissive",
        ],
        "title": "DiT-mini Latin Square n=6 One-Hot Encoding Ablation",
        "labels": ["{-1,+1} σ=1.0 (baseline)", "zero-mean σ=auto", "{0,1} σ=auto"],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard training curves")
    parser.add_argument("--group", type=str, choices=list(GROUPS), default=None,
                        help="Use a pre-baked experiment group (exactK / latinsq)")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Auto-discover exp dirs with this prefix")
    parser.add_argument("--exp_dirs", type=str, nargs="+", default=None,
                        help="Explicit list of experiment folder names")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Legend labels (default: derived from exp dir names)")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                        help="TensorBoard scalar tags to plot")
    parser.add_argument("--title", type=str, default="Training Curves")
    parser.add_argument("--outpath", type=str, default="/tmp/tb_curves.png")
    parser.add_argument("--saveroot", type=str, default=SAVEROOT)
    parser.add_argument("--no_log_x", action="store_true")
    args = parser.parse_args()

    # Resolve group shortcut
    if args.group:
        g = GROUPS[args.group]
        tags  = args.tags or g["tags"]
        title = args.title if args.title != "Training Curves" else g["title"]
        if "exp_dirs" in g:
            # explicit list — no prefix discovery
            exp_dirs = args.exp_dirs or g["exp_dirs"]
            prefix   = ""
            labels   = args.labels or g.get("labels") or [e.replace("DiT_B_latinSq_", "").replace("DiT_mini_latinSq_", "") for e in exp_dirs]
        else:
            prefix   = g["prefix"]
            exp_dirs = args.exp_dirs or discover_experiments(args.saveroot, prefix)
            labels   = args.labels or [make_label(e, prefix) for e in exp_dirs]
    elif args.prefix:
        prefix   = args.prefix
        exp_dirs = args.exp_dirs or discover_experiments(args.saveroot, prefix)
        labels   = args.labels or [make_label(e, prefix) for e in exp_dirs]
        tags     = args.tags or ["train/loss"]
        title    = args.title
    else:
        if not args.exp_dirs or not args.tags:
            parser.error("Provide --group, --prefix, or both --exp_dirs and --tags")
        exp_dirs = args.exp_dirs
        labels   = args.labels or exp_dirs
        tags     = args.tags
        title    = args.title

    if not exp_dirs:
        print(f"No experiments found in {args.saveroot}")
        sys.exit(1)

    print(f"Plotting {len(exp_dirs)} experiments: {exp_dirs}")
    plot_curves(
        exp_dirs=exp_dirs,
        labels=labels,
        tags=tags,
        title=title,
        outpath=args.outpath,
        saveroot=args.saveroot,
        log_x=not args.no_log_x,
    )


if __name__ == "__main__":
    main()
