"""
analyze_error_cell_confidence.py

Test whether cells that cause row/col violations (error cells) have lower
model confidence than correct cells.

Two confidence measures depending on encoding:

  onehot:  confidence = (max_channel − inactive) / (active − inactive) ∈ [0, 1]
           1.0 = perfectly sharp one-hot, 0 = completely flat

  scalar:  dist_to_nearest = min_v |x − v_valid|  (v_valid = valid float levels)
           normalized: confidence = 1 − dist / (spacing/2)
           0 = exactly on a valid level, negative = farther than half-spacing

Key reusable components (importable):
  find_error_cells()             — identify row/col violating positions
  onehot_cell_confidence()       — per-cell confidence for one-hot encoding
  scalar_cell_confidence()       — per-cell confidence for scalar encoding

Usage:
  # onehot baseline
  python scripts/analyze_error_cell_confidence.py \\
      --exp_name DiT_mini_latinSq_n6_N4096_onehot \\
      --encoding onehot --n_size 6 --n_ckpts 10 \\
      --outpath /tmp/errcell_pm1.png

  # scalar run
  python scripts/analyze_error_cell_confidence.py \\
      --exp_name DiT_mini_latinSq_n6_N4096_scalar \\
      --encoding scalar --n_size 6 --n_ckpts 10 \\
      --outpath /tmp/errcell_scalar.png
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.latin_square_lib import valid_float_values, snap_to_integer

# Publication style
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["font.family"]  = "sans-serif"

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"


# ===========================================================================
# Reusable components
# ===========================================================================

def find_error_cells(decoded_int, n):
    """
    Identify cells that cause row or column violations.

    A cell (r, c) is an error cell if its decoded symbol appears more than
    once in row r OR more than once in column c.

    Parameters
    ----------
    decoded_int : np.ndarray of shape (N, n²), int (no NaNs)
    n           : int

    Returns
    -------
    error_mask : np.ndarray of shape (N, n²), bool
                 True where the cell participates in a row or col violation
    """
    N = len(decoded_int)
    grids = decoded_int.reshape(N, n, n)    # (N, n, n)
    error = np.zeros((N, n, n), dtype=bool)

    for r in range(n):
        row_vals = grids[:, r, :]           # (N, n)
        for v in range(n):
            dup_samples = np.where((row_vals == v).sum(axis=1) > 1)[0]
            for s in dup_samples:
                error[s, r, row_vals[s] == v] = True

    for c in range(n):
        col_vals = grids[:, :, c]           # (N, n)
        for v in range(n):
            dup_samples = np.where((col_vals == v).sum(axis=1) > 1)[0]
            for s in dup_samples:
                error[s, col_vals[s] == v, c] = True

    return error.reshape(N, n * n)


def onehot_cell_confidence(samples_oh, active, inactive):
    """
    Per-cell confidence for one-hot encoded samples.

    confidence = (max_channel − inactive) / (active − inactive) ∈ [0, 1]

    Parameters
    ----------
    samples_oh : np.ndarray (N, n, n²) float
    active, inactive : float, encoding values

    Returns
    -------
    conf : np.ndarray (N, n²), float in [0, 1]
    decoded_int : np.ndarray (N, n²), int (argmax, no nan filtering)
    """
    max_act    = samples_oh.max(axis=1)                          # (N, n²)
    conf       = (max_act - inactive) / (active - inactive)
    decoded    = samples_oh.argmax(axis=1)                       # (N, n²) int
    return conf, decoded


def scalar_cell_confidence(samples_sc, n, eps=0.15):
    """
    Per-cell confidence for scalar encoded samples.

    Confidence = 1 − dist_to_nearest / (half_spacing)
    where half_spacing = 1 / (n − 1) (half the gap between adjacent valid levels).

    Values > 1: cell is very close to a valid level (more confident than midpoint).
    Values < 0: cell is farther than half-spacing from any valid level (unsnappable).
    For a cell exactly on a valid level: confidence = 1.0.

    Parameters
    ----------
    samples_sc : np.ndarray (N, n²) float
    n          : int
    eps        : float, snap tolerance (for decoding; does not affect confidence)

    Returns
    -------
    conf : np.ndarray (N, n²), float (can be negative for unsnappable cells)
    decoded_int : np.ndarray (N, n²), int (NaN where unsnappable, cast to -1 here)
    snappable_mask : np.ndarray (N,) bool — samples where ALL cells snapped
    """
    vf = valid_float_values(n)           # (n,) valid float levels
    half_spacing = 1.0 / (n - 1)        # spacing / 2

    x = samples_sc                       # (N, n²)
    dists = np.abs(x[:, :, None] - vf[None, None, :])   # (N, n², n)
    nearest_dist = dists.min(axis=2)     # (N, n²)
    nearest_idx  = dists.argmin(axis=2)  # (N, n²) int

    conf = 1.0 - nearest_dist / half_spacing   # 1=on grid, 0=midpoint, <0=off

    # Mask samples where any cell is unsnappable
    snappable_mask = (nearest_dist <= eps).all(axis=1)    # (N,)

    # Decode: use nearest_idx for all cells (NaN-free for analysis)
    decoded = nearest_idx.astype(int)
    return conf, decoded, snappable_mask


# ===========================================================================
# Per-experiment analysis
# ===========================================================================

def analyse_exp(savedir, encoding, n, n_ckpts, eps_onehot=0.3, eps_scalar=0.15):
    """
    Returns dicts with per-cell confidence arrays split by error/correct.
    """
    sample_dir = os.path.join(savedir, "samples")
    files = sorted(glob.glob(os.path.join(sample_dir, "samples_epoch_*.pt")))
    if not files:
        return None
    files = files[-n_ckpts:]
    steps = [int(os.path.basename(f).split("_epoch_")[1].split(".pt")[0])
             for f in files]

    # Get encoding params
    args_json = os.path.join(savedir, "args.json")
    onehot_type = "pm1"
    if os.path.exists(args_json):
        with open(args_json) as f:
            run_args = json.load(f)
        onehot_type = run_args.get("onehot_type", "pm1")

    if onehot_type == "pm1":
        active, inactive = 1.0, -1.0
    elif onehot_type == "zero_one":
        active, inactive = 1.0, 0.0
    elif onehot_type == "zero_mean":
        active  = (n - 1) / n
        inactive = -1.0 / n
    else:
        active, inactive = 1.0, -1.0

    conf_correct = []
    conf_error   = []

    for fpath in files:
        x = torch.load(fpath, map_location="cpu", weights_only=False)
        N = x.shape[0]

        if encoding == "onehot":
            samples = x.numpy().reshape(N, n, n * n)    # (N, n, n²)
            conf, decoded = onehot_cell_confidence(samples, active, inactive)
            use_mask = np.ones(N, dtype=bool)            # use all samples
        else:  # scalar
            samples = x.numpy().reshape(N, n * n)        # (N, n²)
            conf, decoded, use_mask = scalar_cell_confidence(samples, n, eps=eps_scalar)

        # Only analyse samples that decoded cleanly
        decoded_clean = decoded[use_mask]
        conf_clean    = conf[use_mask]

        err_mask = find_error_cells(decoded_clean, n)     # (M, n²)

        conf_correct.append(conf_clean[~err_mask])
        conf_error.append(conf_clean[err_mask])

    conf_correct = np.concatenate(conf_correct)
    conf_error   = np.concatenate(conf_error)

    return dict(
        conf_correct=conf_correct,
        conf_error=conf_error,
        steps=steps,
        encoding=encoding,
        onehot_type=onehot_type if encoding == "onehot" else None,
        active=active if encoding == "onehot" else None,
        inactive=inactive if encoding == "onehot" else None,
        n=n,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def print_summary(res, label):
    cc = res["conf_correct"]
    ce = res["conf_error"]
    print(f"\n{'='*60}")
    print(f"  {label}  |  encoding={res['encoding']}  steps={res['steps'][0]}–{res['steps'][-1]}")
    print(f"{'='*60}")
    print(f"  Cells: {len(cc):,} correct, {len(ce):,} error "
          f"({100*len(ce)/(len(cc)+len(ce)):.1f}% error rate)")
    print(f"\n  {'Metric':<28} {'Correct':>10} {'Error':>10}  {'Diff':>9}")
    print(f"  {'-'*57}")
    for lbl, fn in [
        ("mean confidence",           np.mean),
        ("median confidence",         np.median),
        ("std confidence",            np.std),
        ("% conf > 0.99",  lambda x: (x > 0.99).mean()),
        ("% conf > 0.95",  lambda x: (x > 0.95).mean()),
        ("% conf > 0.90",  lambda x: (x > 0.90).mean()),
        ("% conf < 0.70",  lambda x: (x < 0.70).mean()),
        ("% conf < 0.50",  lambda x: (x < 0.50).mean()),
    ]:
        vc = fn(cc)
        ec = fn(ce)
        print(f"  {lbl:<28} {vc:>10.4f} {ec:>10.4f}  {ec-vc:>+9.4f}")


def plot_results(results, labels, outpath):
    n_exps = len(results)
    fig, axes = plt.subplots(2, n_exps, figsize=(5.5 * n_exps, 8))
    if n_exps == 1:
        axes = axes[:, None]
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    COLORS_ERR  = "tab:red"
    COLORS_CORR = "tab:blue"

    for col, (res, label) in enumerate(zip(results, labels)):
        cc = res["conf_correct"]
        ce = res["conf_error"]
        enc = res["encoding"]

        if enc == "onehot":
            bins = np.linspace(0.6, 1.02, 60)
            xlabel = "Cell confidence (normalised)"
            pct_label = "% conf > 0.99"
            pct_fn = lambda x: (x > 0.99).mean()
        else:
            bins = np.linspace(-0.2, 1.05, 60)
            xlabel = "Confidence (1 − dist/half-spacing)"
            pct_label = "% conf > 0.95"
            pct_fn = lambda x: (x > 0.95).mean()

        # Panel 1: histogram
        ax = axes[0, col]
        ax.hist(cc, bins=bins, density=True, color=COLORS_CORR,
                alpha=0.55, label=f"correct ({len(cc):,})")
        ax.hist(ce, bins=bins, density=True, color=COLORS_ERR,
                alpha=0.55, label=f"error ({len(ce):,})")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{label}\n({enc})", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

        # Panel 2: summary bars
        ax2 = axes[1, col]
        metrics = [
            ("mean conf",  np.mean(cc),        np.mean(ce)),
            ("median",     np.median(cc),       np.median(ce)),
            (pct_label,    pct_fn(cc),          pct_fn(ce)),
            ("% conf>0.90", (cc>0.90).mean(),   (ce>0.90).mean()),
            ("% conf<0.70", (cc<0.70).mean(),   (ce<0.70).mean()),
        ]
        x_pos = np.arange(len(metrics))
        w = 0.35
        ax2.bar(x_pos - w/2, [m[1] for m in metrics], w, color=COLORS_CORR,
                label="correct", alpha=0.8)
        ax2.bar(x_pos + w/2, [m[2] for m in metrics], w, color=COLORS_ERR,
                label="error", alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m[0] for m in metrics], rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Value", fontsize=10)
        ax2.set_title("Summary comparison", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(alpha=0.2, axis="y")

    steps_str = f"steps {results[0]['steps'][0]}–{results[0]['steps'][-1]}"
    fig.suptitle(f"Error cell confidence analysis  |  last {len(results[0]['steps'])} ckpts, {steps_str}",
                 fontsize=11)

    base, _ = os.path.splitext(outpath)
    plt.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    print(f"\nSaved → {base}.png  +  {base}.pdf")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_names", type=str, nargs="+",
                        default=["DiT_mini_latinSq_n6_N4096_onehot"])
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--encodings", type=str, nargs="+", default=None,
                        help="'onehot' or 'scalar' for each exp (default: read from args.json)")
    parser.add_argument("--n_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--saveroot", type=str, default=SAVEROOT)
    parser.add_argument("--n_ckpts", type=int, default=10)
    parser.add_argument("--outpath", type=str, default="/tmp/errcell_confidence.png")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_names = args.exp_names
    labels    = args.labels or exp_names
    n_sizes   = args.n_sizes or [6] * len(exp_names)

    # Infer encoding from args.json if not provided
    if args.encodings is None:
        encodings = []
        for exp in exp_names:
            aj = os.path.join(args.saveroot, exp, "args.json")
            enc = "onehot"
            if os.path.exists(aj):
                with open(aj) as f:
                    enc = json.load(f).get("encoding", "onehot")
            encodings.append(enc)
    else:
        encodings = args.encodings

    results = []
    for exp, enc, n, label in zip(exp_names, encodings, n_sizes, labels):
        print(f"\nAnalysing {exp}  [{enc}, n={n}] ...")
        savedir = os.path.join(args.saveroot, exp)
        res = analyse_exp(savedir, enc, n, args.n_ckpts)
        if res is None:
            print("  [skip] no sample files found")
            continue
        results.append(res)
        print_summary(res, label)

    if results:
        plot_results(results, labels[:len(results)], args.outpath)


if __name__ == "__main__":
    main()
