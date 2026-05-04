#!/usr/bin/env python3
"""
Plot GPT parity-learning CE analysis figures.

Layout (per run):
  Top row  : CE loss vs training step (3 splits on one axes)
  Bottom row: per-position CE heatmaps (train | valid_novel | boolean_cube)

Color convention matches DiT sigma-loss plots:
  Train        #2166ac  blue  solid
  Valid-novel  #d73027  red   solid
  Boolean cube #555555  gray  dashed

Usage
-----
  python scripts/plot_GPT_CE_analysis.py \
      --exp_names GPT_mini_parity_N4096_D36_G6_even_lr1e4 \
                  GPT_mini_parity_N4096_D36_G6_even_wd1e2 \
      --saveroot /n/.../DiffusionParityLearning \
      --figdir figures/GPT_parity_learn_dissection \
      --group_size 6
"""

import argparse
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ── colour / style constants (match DiT plots) ──────────────────────────────
SPLIT_STYLES = {
    "train":        dict(color="#2166ac", lw=2.0, ls="-",  label="Train"),
    "valid_novel":  dict(color="#d73027", lw=2.0, ls="-",  label="Valid (novel)"),
    "boolean_cube": dict(color="#555555", lw=1.6, ls="--", label="Boolean cube"),
}
HMAP_CMAP = "magma"   # shared colormap; reversed so low CE = bright, high CE = dark
HMAP_TITLE_COLORS = {
    "train":        "#2166ac",   # blue  — matches CE curve
    "valid_novel":  "#d73027",   # red
    "boolean_cube": "#888888",   # gray (slightly lighter than #555 for visibility on white)
}
HMAP_TITLES = {
    "train":        "Train",
    "valid_novel":  "Valid (novel)",
    "boolean_cube": "Boolean cube",
}

SAVEROOT_DEFAULT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)
FIGDIR_DEFAULT = (
    "/n/home12/binxuwang/Github/DiffusionAttnConsistency/figures/"
    "GPT_parity_learn_dissection"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_ce_data(exp_dir, n_eval=4096, suffix=""):
    tag = f"_n{n_eval}{suffix}" if suffix else f"_n{n_eval}"
    path = os.path.join(exp_dir, "ce_analysis", f"ce_vs_step{tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CE data not found: {path}")
    d = np.load(path, allow_pickle=True)
    return d


def _decade_ticks(epochs):
    """Return (step_values, labels) at exact powers of 10 within the epoch range."""
    emin, emax = epochs[0], epochs[-1]
    lo = int(np.floor(np.log10(max(emin, 1))))
    hi = int(np.ceil(np.log10(emax)))
    powers = [10**p for p in range(lo, hi + 1) if emin <= 10**p <= emax]
    labels = [f"$10^{{{p}}}$" for p in range(lo, hi + 1) if emin <= 10**p <= emax]
    return powers, labels


def make_step_axis(ax, epochs):
    """Configure a log-scale x-axis with standard 10^n ticks."""
    ax.set_xscale("log")
    ax.set_xlabel("Training step", fontsize=11)
    powers, labels = _decade_ticks(epochs)
    ax.set_xticks(powers)
    ax.set_xticklabels(labels, fontsize=9)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def heatmap_step_ticks(ax, epochs):
    """Set x-ticks on a heatmap axes (imshow, x = pixel index) at 10^n steps.
    Maps each decade value to its nearest checkpoint index."""
    epochs = np.asarray(epochs)
    log_e  = np.log10(epochs.astype(float) + 1)
    powers, labels = _decade_ticks(epochs)
    # Map each power-of-10 to the nearest checkpoint pixel index
    idxs = np.array([np.argmin(np.abs(epochs - p)) for p in powers])
    ax.set_xticks(idxs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Step", fontsize=10)


def group_boundary_lines(ax, group_size, n_pos, orientation="horizontal", lw=0.8, color="cyan"):
    """Draw group-boundary lines every group_size positions."""
    for k in range(group_size, n_pos, group_size):
        if orientation == "horizontal":
            ax.axhline(k - 0.5, color=color, lw=lw)
        else:
            ax.axvline(k - 0.5, color=color, lw=lw)


# ── main figure ──────────────────────────────────────────────────────────────

def plot_ce_figure(exp_dir, group_size=6, n_eval=4096, suffix="",
                   figsize=(13, 7), dpi=150, vmax=None):
    """
    Build the 4-panel figure:
      [top-left]  CE loss vs step (3 splits)
      [bot-left]  per-position heatmap: Train
      [bot-mid]   per-position heatmap: Valid (novel)
      [bot-right] per-position heatmap: Boolean cube
    """
    d = load_ce_data(exp_dir, n_eval=n_eval, suffix=suffix)
    epochs = d["epochs"]

    try:
        args_dict = json.loads(str(d["args_json"]))
        exp_tag = args_dict.get("exp_name", os.path.basename(exp_dir))
        lr  = args_dict.get("lr", "?")
        wd  = args_dict.get("weight_decay", "?")
    except Exception:
        exp_tag = os.path.basename(exp_dir)
        lr, wd = "?", "?"

    splits = ["train", "valid_novel", "boolean_cube"]
    loss   = {s: d[f"loss_{s}"] for s in splits}
    pos_loss = {s: d[f"pos_loss_{s}"] for s in splits}   # (C, 36)
    n_pos = pos_loss["train"].shape[1]

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.4], hspace=0.45, wspace=0.30)

    # ── top: CE loss curves (span all 3 columns) ────────────────────────────
    ax_loss = fig.add_subplot(gs[0, :])
    for s in splits:
        sty = SPLIT_STYLES[s]
        ax_loss.plot(epochs, loss[s], **sty)

    ax_loss.axhline(np.log(2), color="gray", lw=0.9, ls=":", label=f"log(2)={np.log(2):.3f}")
    ax_loss.set_ylabel("CE loss", fontsize=11)
    ax_loss.set_title(f"{exp_tag}  (lr={lr}, wd={wd})", fontsize=11)
    ax_loss.legend(loc="upper left", fontsize=10)
    make_step_axis(ax_loss, epochs)

    # ── bottom: per-position heatmaps ────────────────────────────────────────
    if vmax is None:
        vmax = max(np.nanpercentile(pos_loss[s], 97) for s in splits)

    hax = [fig.add_subplot(gs[1, c]) for c in range(3)]
    for col, s in enumerate(splits):
        ax = hax[col]
        mat = pos_loss[s].T          # (36, C) → rows=position, cols=checkpoint
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       cmap=HMAP_CMAP,
                       norm=Normalize(vmin=0, vmax=vmax),
                       interpolation="nearest")
        group_boundary_lines(ax, group_size, n_pos, orientation="horizontal")
        heatmap_step_ticks(ax, epochs)
        ax.set_ylabel("Bit position" if col == 0 else "", fontsize=10)
        ax.set_title(HMAP_TITLES[s], fontsize=11, color=HMAP_TITLE_COLORS[s], fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85, label="CE")

    fig.suptitle("GPT CE analysis", fontsize=12, y=1.01)
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_names", nargs="+", required=True)
    p.add_argument("--saveroot",   default=SAVEROOT_DEFAULT)
    p.add_argument("--figdir",     default=FIGDIR_DEFAULT)
    p.add_argument("--group_size", type=int, default=6)
    p.add_argument("--n_eval",     type=int, default=4096)
    p.add_argument("--suffix",     default="",
                   help="NPZ file suffix (e.g. '_v2')")
    p.add_argument("--dpi",        type=int,   default=150)
    p.add_argument("--vmax",       type=float, default=None,
                   help="Color limit for heatmaps (default: 97th percentile)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    for exp_name in args.exp_names:
        exp_dir = os.path.join(args.saveroot, exp_name)
        if not os.path.isdir(exp_dir):
            print(f"WARNING: {exp_dir} not found — skipping.")
            continue

        # derive a short tag from exp_name for the filename
        short = exp_name.replace("GPT_mini_parity_N4096_D36_", "").replace("_even", "")
        out_base = os.path.join(args.figdir, f"GPT_G6_CE_analysis_{short}")

        fig = plot_ce_figure(
            exp_dir,
            group_size=args.group_size,
            n_eval=args.n_eval,
            suffix=args.suffix,
            dpi=args.dpi,
            vmax=args.vmax,
        )
        fig.savefig(out_base + ".pdf", bbox_inches="tight")
        fig.savefig(out_base + ".png", bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)
        print(f"Saved → {out_base}.{{pdf,png}}")


if __name__ == "__main__":
    main()
