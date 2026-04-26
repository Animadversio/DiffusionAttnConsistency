"""
plot_onehot_margin_pubfig.py

Publication-quality figure showing how validity depends on confidence threshold (eps)
for one or more one-hot Latin square training runs.

Key insight: at strict thresholds (small eps) only the most confident samples are
decoded — these tend to be more valid. The gap shows that "uncertain" model outputs
are systematically less likely to be valid Latin squares.

Three panels:
  1. Conditioned valid ratio vs eps  — validity among decoded (threshold-passing) samples
  2. Nan ratio vs eps                — fraction rejected by threshold
  3. Margin histogram                — distribution of per-cell confidence values

Usage:
  python scripts/plot_onehot_margin_pubfig.py \\
      --exp_names DiT_mini_latinSq_n6_N4096_onehot \\
                  DiT_mini_latinSq_n6_N4096_onehot_zeromean_autoSD \\
                  DiT_mini_latinSq_n6_N4096_onehot_zeroone_autoSD \\
      --labels "{-1,+1} baseline" "zero-mean" "{0,1}" \\
      --n_sizes 6 6 6 \\
      --n_ckpts 10 \\
      --outpath /tmp/margin_pubfig.png
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

# Publication style
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["font.family"]  = "sans-serif"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.latin_square_lib import check_latin_square_batch

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def get_encoding_params(savedir, n):
    args_json = os.path.join(savedir, "args.json")
    onehot_type = "pm1"
    if os.path.exists(args_json):
        with open(args_json) as f:
            run_args = json.load(f)
        onehot_type = run_args.get("onehot_type", "pm1")
    if onehot_type == "pm1":
        return 1.0, -1.0
    elif onehot_type == "zero_one":
        return 1.0, 0.0
    elif onehot_type == "zero_mean":
        return (n - 1) / n, -1.0 / n
    return 1.0, -1.0


def sweep_eps_for_exp(savedir, n, eps_vals, n_ckpts):
    """Returns arrays of shape (n_ckpts, n_eps) for nan_ratio and cond_valid_ratio."""
    sample_dir = os.path.join(savedir, "samples")
    files = sorted(glob.glob(os.path.join(sample_dir, "samples_epoch_*.pt")))
    if not files:
        return None, None, None, None
    files = files[-n_ckpts:]
    steps = [int(os.path.basename(f).split("_epoch_")[1].split(".pt")[0]) for f in files]

    active, inactive = get_encoding_params(savedir, n)
    all_nan  = np.zeros((len(files), len(eps_vals)))
    all_cond = np.zeros((len(files), len(eps_vals)))

    # Load last ckpt for margin histogram
    x_last = torch.load(files[-1], map_location="cpu", weights_only=False)
    samples_last = x_last.numpy().reshape(x_last.shape[0], n, n * n)
    max_act_last = samples_last.max(axis=1)
    margins_last = ((max_act_last - inactive) / (active - inactive)).ravel()

    for ci, fpath in enumerate(files):
        x = torch.load(fpath, map_location="cpu", weights_only=False)
        samples = x.numpy().reshape(x.shape[0], n, n * n)
        max_act = samples.max(axis=1)
        conf = (max_act - inactive) / (active - inactive)
        int_base = samples.argmax(axis=1).astype(float)
        for ei, eps in enumerate(eps_vals):
            iv = int_base.copy()
            iv[conf < (1.0 - eps)] = np.nan
            nan_mask = np.isnan(iv).any(axis=1)
            all_nan[ci, ei] = nan_mask.mean()
            valid_int = iv[~nan_mask].astype(int)
            if len(valid_int) > 0:
                r, c = check_latin_square_batch(valid_int, n)
                all_cond[ci, ei] = (r & c).mean()

    return all_nan, all_cond, margins_last, steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_names", type=str, nargs="+",
                        default=["DiT_mini_latinSq_n6_N4096_onehot"])
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--n_sizes", type=int, nargs="+", default=None,
                        help="n for each exp (default: all 6)")
    parser.add_argument("--saveroot", type=str, default=SAVEROOT)
    parser.add_argument("--n_ckpts", type=int, default=10)
    parser.add_argument("--eps_values", type=float, nargs="+",
                        default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                                 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--outpath", type=str, default="/tmp/margin_pubfig.png")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_names = args.exp_names
    labels = args.labels or exp_names
    n_sizes = args.n_sizes or [6] * len(exp_names)
    eps_vals = np.array(args.eps_values)

    # Gather data for all experiments
    results = []
    for exp, n in zip(exp_names, n_sizes):
        savedir = os.path.join(args.saveroot, exp)
        print(f"Loading {exp} ...")
        all_nan, all_cond, margins, steps = sweep_eps_for_exp(
            savedir, n, eps_vals, args.n_ckpts)
        if all_nan is None:
            print(f"  [skip] no sample files found")
            results.append(None)
        else:
            print(f"  steps {steps[0]}..{steps[-1]}, n_ckpts={len(steps)}")
            results.append({"nan": all_nan, "cond": all_cond,
                             "margins": margins, "steps": steps})

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.32)

    ax_cond, ax_nan, ax_hist = axes

    # Compute y-range for cond_valid panel (zoom in)
    all_cond_vals = []
    for r in results:
        if r is not None:
            all_cond_vals.append(r["cond"])
    if all_cond_vals:
        cond_min = np.concatenate(all_cond_vals).min()
        cond_max = np.concatenate(all_cond_vals).max()
        pad = (cond_max - cond_min) * 0.15
        y_lo = max(0, cond_min - pad)
        y_hi = min(1, cond_max + pad)
    else:
        y_lo, y_hi = 0, 1

    for i, (exp, label, r) in enumerate(zip(exp_names, labels, results)):
        if r is None:
            continue
        clr = COLORS[i % len(COLORS)]
        mean_nan  = r["nan"].mean(axis=0)
        mean_cond = r["cond"].mean(axis=0)
        std_cond  = r["cond"].std(axis=0)
        std_nan   = r["nan"].std(axis=0)

        # Panel 1: conditioned valid ratio
        ax_cond.plot(eps_vals, mean_cond, color=clr, lw=2, label=label, marker="o", ms=4)
        ax_cond.fill_between(eps_vals, mean_cond - std_cond, mean_cond + std_cond,
                              color=clr, alpha=0.15)

        # Panel 2: nan ratio
        ax_nan.plot(eps_vals, mean_nan, color=clr, lw=2, label=label, marker="o", ms=4)
        ax_nan.fill_between(eps_vals, mean_nan - std_nan, mean_nan + std_nan,
                             color=clr, alpha=0.15)

        # Panel 3: margin histogram (last ckpt)
        ax_hist.hist(r["margins"], bins=80, range=(0, 1.05), color=clr,
                     alpha=0.5, density=True, label=f"{label} (step {r['steps'][-1]})")

    # Remove top/right spines from all panels
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Style panel 1
    ax_cond.set_xscale("log")
    ax_cond.set_xlabel("Confidence threshold eps", fontsize=11)
    ax_cond.set_ylabel("Valid ratio (conditioned on non-nan)", fontsize=10)
    ax_cond.set_title("Validity vs threshold", fontsize=12)
    ax_cond.set_xlim(eps_vals[0], eps_vals[-1])
    ax_cond.set_ylim(y_lo, y_hi)
    ax_cond.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_cond.axvline(0.3, color="gray", ls="--", lw=1, label="default eps=0.3")
    ax_cond.legend(fontsize=8, loc="lower right")
    ax_cond.grid(alpha=0.25, which="both")

    # Style panel 2
    ax_nan.set_xscale("log")
    ax_nan.set_xlabel("Confidence threshold eps", fontsize=11)
    ax_nan.set_ylabel("Nan ratio (fraction rejected)", fontsize=10)
    ax_nan.set_title("Rejection rate vs threshold", fontsize=12)
    ax_nan.set_xlim(eps_vals[0], eps_vals[-1])
    ax_nan.set_ylim(0, 1.0)
    ax_nan.axvline(0.3, color="gray", ls="--", lw=1, label="default eps=0.3")
    ax_nan.legend(fontsize=8)
    ax_nan.grid(alpha=0.25, which="both")

    # Style panel 3
    ax_hist.axvline(0.7, color="gray", ls="--", lw=1.2, label="eps=0.3 threshold")
    ax_hist.axvline(0.9, color="orange", ls="--", lw=1.2, label="eps=0.1 threshold")
    ax_hist.set_xlabel("Cell confidence", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.set_title("Margin distribution (last ckpt)", fontsize=12)
    ax_hist.set_xlim(0, 1.05)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.25)

    n_ckpts_str = f"mean ± std over last {args.n_ckpts} ckpts"
    fig.suptitle(f"One-hot confidence threshold analysis  |  {n_ckpts_str}", fontsize=11)

    base, _ = os.path.splitext(args.outpath)
    plt.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    print(f"Saved → {base}.png  +  {base}.pdf")


if __name__ == "__main__":
    main()
