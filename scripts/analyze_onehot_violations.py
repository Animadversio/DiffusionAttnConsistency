"""
analyze_onehot_violations.py

Analyze one-hot encoding quality over training for Latin square experiments.

Loads pre-computed CSV (onehot_violation_metrics.csv) from each experiment dir,
or recomputes from saved .pt sample files if --recompute is set or CSV is missing.

Generates a 2×3 figure per experiment pair showing:
  - Stackplot of cell violation types over training
  - Binarization vs rule-learning curves + phase markers
  - Channel activation margin over training

Usage:
  python analyze_onehot_violations.py \\
      --exp_names DiT_mini_latinSq_n5_N4096_onehot DiT_mini_latinSq_n6_N4096_onehot \\
      --saveroot /path/to/experiments \\
      --outpath /tmp/onehot_analysis.png \\
      [--recompute] [--n_steps 40]
"""
import argparse, os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
from latin_square_lib import onehot_to_int, check_latin_square_batch

HOT_THRESH  = 0.7
COLD_THRESH = -0.7


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_onehot_metrics(x, n):
    """Compute violation metrics for a batch of one-hot encoded samples."""
    N = len(x)
    xnp = x.numpy().reshape(N, n, n * n)   # (N, n, n²)

    max_ch  = xnp.max(axis=1)              # (N, n²)
    sec_max = np.sort(xnp, axis=1)[:, -2, :]  # (N, n²) 2nd highest
    margin  = max_ch - sec_max             # (N, n²) → 2.0 when perfect
    n_hot   = (xnp > HOT_THRESH).sum(axis=1)   # (N, n²)
    n_cold  = (xnp < COLD_THRESH).sum(axis=1)  # (N, n²)

    frac_perfect = ((n_hot == 1) & (n_cold == n - 1)).mean()
    frac_twohot  = (n_hot >= 2).mean()
    frac_missing = (n_hot == 0).mean()
    frac_ambig   = ((n_hot == 1) & (n_cold < n - 1)).mean()
    sample_perfect = ((n_hot == 1) & (n_cold == n - 1)).all(axis=1).mean()

    xi = onehot_to_int(xnp, n, eps=0.3)
    nm = np.isnan(xi).any(axis=1)
    vi = xi[~nm].astype(int)
    if len(vi):
        rv, cv = check_latin_square_batch(vi, n)
        full_valid = (rv & cv).mean()
        row_valid  = rv.mean()
        col_valid  = cv.mean()
    else:
        full_valid = row_valid = col_valid = 0.0

    return dict(
        frac_perfect_cells   =float(frac_perfect),
        frac_twohot_cells    =float(frac_twohot),
        frac_missing_cells   =float(frac_missing),
        frac_ambig_cells     =float(frac_ambig),
        mean_max_channel     =float(max_ch.mean()),
        mean_2nd_max         =float(sec_max.mean()),
        mean_margin          =float(margin.mean()),
        mean_n_hot           =float(n_hot.mean()),
        mean_n_cold          =float(n_cold.mean()),
        sample_perfect_onehot=float(sample_perfect),
        full_valid_ratio     =float(full_valid),
        row_valid_ratio      =float(row_valid),
        col_valid_ratio      =float(col_valid),
        nan_ratio            =float(nm.mean()),
    )


def compute_or_load(exp_dir, n, recompute=False, n_steps=40):
    """Return DataFrame of metrics over training, loading from CSV or recomputing."""
    csv_path = os.path.join(exp_dir, "onehot_violation_metrics.csv")
    if not recompute and os.path.exists(csv_path):
        print(f"  Loading CSV: {csv_path}")
        return pd.read_csv(csv_path)

    print(f"  Recomputing from .pt files in {exp_dir}/samples/")
    sdir     = os.path.join(exp_dir, "samples")
    pt_files = sorted(glob.glob(os.path.join(sdir, "samples_epoch_*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {sdir}")
    total = len(pt_files)
    idxs  = np.unique(np.concatenate(
        [[0], np.round(np.geomspace(1, total - 1, n_steps)).astype(int)]
    ))
    rows = []
    for i in idxs:
        fpath = pt_files[i]
        step  = int(os.path.basename(fpath).replace("samples_epoch_", "").replace(".pt", ""))
        x     = torch.load(fpath, map_location="cpu", weights_only=False)
        m     = compute_onehot_metrics(x, n)
        m["step"] = step
        rows.append(m)
        print(f"    step={step:7d}  perfect={m['frac_perfect_cells']:.3f}  "
              f"twohot={m['frac_twohot_cells']:.3f}  full_valid={m['full_valid_ratio']:.3f}")
    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_onehot_analysis(dfs, labels, ns, outpath):
    """
    dfs    : list of DataFrames
    labels : list of str
    ns     : list of int (n values)
    """
    n_exps = len(dfs)
    fig, axes = plt.subplots(n_exps, 3, figsize=(15, 4.5 * n_exps))
    if n_exps == 1:
        axes = axes[None, :]  # ensure 2-D
    fig.suptitle("One-Hot Quality vs Rule Learning — Latin Square", fontsize=13, fontweight='bold')

    for row, (df, label, n, clr) in enumerate(zip(dfs, labels, ns, PALETTE)):
        steps = df["step"].values

        # ---- Panel 0: stackplot ----
        ax = axes[row, 0]
        ax.stackplot(
            steps,
            df["frac_perfect_cells"] * 100,
            df["frac_ambig_cells"]   * 100,
            df["frac_missing_cells"] * 100,
            df["frac_twohot_cells"]  * 100,
            labels=["Perfect one-hot", "Ambiguous", "Missing-hot", "Two-hot"],
            colors=["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"],
            alpha=0.85,
        )
        ax.set_xscale("log"); ax.set_xlim(steps[0], steps[-1])
        ax.set_ylim(0, 100)
        ax.set_xlabel("Training Step"); ax.set_ylabel("% of all cells")
        ax.set_title(f"{label}: Cell One-Hot Quality", fontsize=10)
        ax.legend(fontsize=7, loc="center right"); ax.grid(alpha=0.2)

        # ---- Panel 1: binarization vs rule learning ----
        ax  = axes[row, 1]
        ax2 = ax.twinx()
        l1, = ax.plot(steps, df["frac_perfect_cells"]   * 100, 'o-', color=clr,      lw=2, ms=3, label="Perfect cells (%)")
        l2, = ax.plot(steps, df["sample_perfect_onehot"]* 100, 's--', color=clr,     lw=2, ms=3, alpha=0.55, label="Samples fully perfect (%)")
        l3, = ax2.plot(steps, df["full_valid_ratio"]    * 100, '^-', color="#e377c2", lw=2.5, ms=4, label="Full valid LS (%)")
        ax.set_xscale("log"); ax.set_xlim(steps[0], steps[-1])
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Binarization quality (%)", color=clr)
        ax2.set_ylabel("Full valid ratio (%)", color="#e377c2")
        ax.set_title(f"{label}: Binarization vs Rule Learning", fontsize=10)
        ax.set_ylim(-2, 105); ax2.set_ylim(-2, 105)
        ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], fontsize=7, loc="center left")
        ax.grid(alpha=0.2)
        # phase markers
        t_bin  = steps[(df["frac_perfect_cells"] >= 0.95).values]
        t_rule = steps[(df["full_valid_ratio"]    >= 0.05).values]
        if len(t_bin):
            ax.axvline(t_bin[0],  color=clr,       linestyle=':', alpha=0.7)
            ax.text(t_bin[0]*1.15, 55, f"Binarized\n@{t_bin[0]}", fontsize=7, color='darkorange')
        if len(t_rule):
            ax.axvline(t_rule[0], color="#e377c2", linestyle=':', alpha=0.7)
            ax.text(t_rule[0]*1.15, 20, f"Rule starts\n@{t_rule[0]}", fontsize=7, color='deeppink')

        # ---- Panel 2: channel margin ----
        ax = axes[row, 2]
        ax.plot(steps, df["mean_max_channel"], 'o-', color=clr,    lw=2, ms=3, label="Mean max channel")
        ax.plot(steps, df["mean_2nd_max"],     's--', color=clr,   lw=2, ms=3, alpha=0.5, label="Mean 2nd max")
        ax.plot(steps, df["mean_margin"],      '^-', color="gray", lw=2, ms=3, label="Mean margin")
        for val, lbl, col in [(1.0, "Target max (+1)", "green"), (-1.0, "Target 2nd (−1)", "red"), (2.0, "Target margin", "gray")]:
            ax.axhline(val, color=col, linestyle=':', alpha=0.5)
        ax.set_xscale("log"); ax.set_xlim(steps[0], steps[-1])
        ax.set_xlabel("Training Step"); ax.set_ylabel("Channel value")
        ax.set_title(f"{label}: Channel Activation Quality", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.set_ylim(-1.3, 2.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"Figure saved → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze one-hot violation quality over training.")
    parser.add_argument("--exp_names", nargs="+", required=True,
                        help="Experiment folder names under --saveroot")
    parser.add_argument("--ns", nargs="+", type=int, required=True,
                        help="n value for each experiment (e.g. 5 6)")
    parser.add_argument("--saveroot", type=str,
                        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                        help="Root directory containing experiment folders")
    parser.add_argument("--outpath", type=str, default="/tmp/onehot_violations.png")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics even if CSV exists")
    parser.add_argument("--n_steps", type=int, default=40,
                        help="Number of log-spaced steps to sample when recomputing")
    args = parser.parse_args()

    if len(args.exp_names) != len(args.ns):
        parser.error("--exp_names and --ns must have the same length")

    dfs, labels = [], []
    for exp_name, n in zip(args.exp_names, args.ns):
        exp_dir = os.path.join(args.saveroot, exp_name)
        print(f"\n{exp_name}  (n={n})")
        df = compute_or_load(exp_dir, n, recompute=args.recompute, n_steps=args.n_steps)
        dfs.append(df)
        labels.append(exp_name.split("DiT_mini_")[-1])

    plot_onehot_analysis(dfs, labels, args.ns, args.outpath)


if __name__ == "__main__":
    main()
