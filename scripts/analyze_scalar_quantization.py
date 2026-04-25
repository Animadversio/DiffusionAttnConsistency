"""
analyze_scalar_quantization.py

Analyze scalar encoding quantization quality over training for Latin square experiments.

Loads pre-computed CSV (scalar_quantization_metrics.csv) from each experiment dir,
or recomputes from saved .pt sample files if --recompute is set or CSV is missing.

Generates a figure showing:
  - Snappability % and mean distance over training (vs rule accuracy)
  - Cell value histograms at selected training steps

Usage:
  python analyze_scalar_quantization.py \\
      --exp_names DiT_mini_latinSq_n5_N4096_scalar DiT_mini_latinSq_n6_N4096_scalar \\
      --ns 5 6 \\
      --saveroot /path/to/experiments \\
      --outpath /tmp/scalar_quant.png \\
      [--recompute] [--n_steps 40]
"""
import argparse, os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
from latin_square_lib import snap_to_integer, check_latin_square_batch, valid_float_values


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_scalar_metrics(x, n):
    """Compute quantization metrics for a batch of scalar-encoded samples."""
    xnp = x.flatten(1).numpy()     # (N, n²)
    vf  = valid_float_values(n)    # (n,) valid float levels
    gap = 2.0 / (n - 1)

    dists        = np.abs(xnp[:, :, None] - vf[None, None, :])  # (N, n², n)
    dist_nearest = dists.min(axis=2)                              # (N, n²)
    nearest_idx  = dists.argmin(axis=2)                           # (N, n²)

    eps_perm   = gap * 0.375
    eps_strict = gap * 0.125

    frac_snappable_perm   = (dist_nearest < eps_perm).mean()
    frac_snappable_strict = (dist_nearest < eps_strict).mean()

    snap_mask = dist_nearest < eps_perm
    if snap_mask.sum() > 0:
        sym_counts  = np.bincount(nearest_idx[snap_mask], minlength=n)
        sym_entropy = -np.sum((sym_counts / sym_counts.sum()) *
                              np.log(sym_counts / sym_counts.sum() + 1e-9))
    else:
        sym_entropy = 0.0

    xi = snap_to_integer(xnp, n, eps=eps_perm)
    nm = np.isnan(xi).any(axis=1)
    vi = xi[~nm].astype(int)
    if len(vi):
        rv, cv   = check_latin_square_batch(vi, n)
        full_valid = (rv & cv).mean(); row_valid = rv.mean(); col_valid = cv.mean()
    else:
        full_valid = row_valid = col_valid = 0.0

    return dict(
        frac_snappable_permissive=float(frac_snappable_perm),
        frac_snappable_strict    =float(frac_snappable_strict),
        mean_dist_to_nearest     =float(dist_nearest.mean()),
        median_dist_to_nearest   =float(np.median(dist_nearest)),
        p90_dist_to_nearest      =float(np.percentile(dist_nearest, 90)),
        p99_dist_to_nearest      =float(np.percentile(dist_nearest, 99)),
        sym_entropy              =float(sym_entropy),
        nan_ratio                =float(nm.mean()),
        full_valid_ratio         =float(full_valid),
        row_valid_ratio          =float(row_valid),
        col_valid_ratio          =float(col_valid),
        gap=float(gap), eps_perm=float(eps_perm), eps_strict=float(eps_strict),
    )


def compute_or_load(exp_dir, n, recompute=False, n_steps=40):
    csv_path = os.path.join(exp_dir, "scalar_quantization_metrics.csv")
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
        m     = compute_scalar_metrics(x, n)
        m["step"] = step
        rows.append(m)
        print(f"    step={step:7d}  snap%={m['frac_snappable_permissive']:.3f}  "
              f"mean_dist={m['mean_dist_to_nearest']:.4f}  full_valid={m['full_valid_ratio']:.3f}")
    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

HIST_STEPS_BY_N = {
    5: [5, 250, 1500, 25000],
    6: [5, 250, 1500, 23000],
}


def plot_scalar_analysis(dfs, labels, ns, exp_dirs, outpath):
    n_exps = len(dfs)
    # rows: n_exps for dynamics, n_exps for histograms
    fig = plt.figure(figsize=(18, 5 * n_exps + 4 * n_exps))
    fig.suptitle("Scalar Encoding: Quantization Quality vs Rule Learning", fontsize=13, fontweight='bold')
    gs = plt.GridSpec(n_exps * 2, 4, figure=fig, hspace=0.55, wspace=0.35)

    for row, (df, label, n, exp_dir, clr) in enumerate(zip(dfs, labels, ns, exp_dirs, PALETTE)):
        steps = df["step"].values
        gap   = 2.0 / (n - 1)

        # ---- Dynamics left: snap% vs full_valid ----
        ax  = fig.add_subplot(gs[row * 2, :2])
        ax2 = ax.twinx()
        l1, = ax.plot(steps, df["frac_snappable_permissive"] * 100, 'o-', color=clr, lw=2, ms=3, label="Snap % (permissive)")
        l2, = ax.plot(steps, df["frac_snappable_strict"]     * 100, 's--', color=clr, lw=2, ms=3, alpha=0.5, label="Snap % (strict)")
        l3, = ax2.plot(steps, df["full_valid_ratio"]         * 100, '^-', color="#e377c2", lw=2.5, ms=4, label="Full valid (%)")
        ax.set_xscale("log"); ax.set_xlabel("Training Step")
        ax.set_ylabel("Snap % (quantization)", color=clr)
        ax2.set_ylabel("Full valid ratio (%)", color="#e377c2")
        ax.set_title(f"{label}: Snappability vs Rule Learning", fontsize=10)
        ax.set_ylim(40, 105); ax2.set_ylim(-2, 105)
        ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], fontsize=7)
        ax.grid(alpha=0.2)
        t99   = steps[(df["frac_snappable_permissive"] >= 0.99).values]
        t_rule = steps[(df["full_valid_ratio"]         >= 0.05).values]
        if len(t99):   ax.axvline(t99[0],   color=clr,       linestyle=':', alpha=0.7)
        if len(t_rule): ax.axvline(t_rule[0], color="#e377c2", linestyle=':', alpha=0.7)

        # ---- Dynamics right: distance metrics ----
        ax  = fig.add_subplot(gs[row * 2, 2:])
        ax2 = ax.twinx()
        ax.plot(steps, df["mean_dist_to_nearest"], 'o-', color=clr, lw=2, ms=3, label="Mean dist")
        ax.plot(steps, df["p90_dist_to_nearest"],  's--', color=clr, lw=1.5, ms=2, alpha=0.5, label="P90 dist")
        ax.plot(steps, df["p99_dist_to_nearest"],  '^:', color=clr, lw=1.5, ms=2, alpha=0.3, label="P99 dist")
        ax.axhline(gap / 2, color='gray', linestyle='--', alpha=0.4)
        ax.text(steps[1], gap / 2 * 1.05, f"half-gap={gap/2:.2f}", fontsize=7, color='gray')
        ax2.plot(steps, df["full_valid_ratio"] * 100, color="#e377c2", lw=2.5, ms=4, label="Full valid (%)")
        ax.set_xscale("log"); ax.set_xlabel("Training Step")
        ax.set_ylabel("Dist to nearest valid level", color=clr)
        ax2.set_ylabel("Full valid ratio (%)", color="#e377c2")
        ax.set_title(f"{label}: Quantization Error Over Training", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        # ---- Histograms ----
        sdir     = os.path.join(exp_dir, "samples")
        pt_files = sorted(glob.glob(os.path.join(sdir, "samples_epoch_*.pt")))
        avail    = np.array([int(os.path.basename(f).replace("samples_epoch_","").replace(".pt","")) for f in pt_files])
        vf       = valid_float_values(n)
        targets  = HIST_STEPS_BY_N.get(n, [5, 500, 5000, 50000])

        for col_idx, target in enumerate(targets):
            ax   = fig.add_subplot(gs[row * 2 + 1, col_idx])
            near = int(np.argmin(np.abs(avail - target)))
            fpath = pt_files[near]
            step_actual = avail[near]
            x = torch.load(fpath, map_location="cpu", weights_only=False)
            vals = x.flatten().numpy()
            ax.hist(vals, bins=100, range=(-1.25, 1.25), color=clr, alpha=0.75, density=True)
            for v in vf:
                ax.axvline(v, color='black', linestyle='--', alpha=0.55, linewidth=1)
            # snap & valid from CSV
            row_match = df[df["step"] == step_actual]
            snap_pct = row_match["frac_snappable_permissive"].values[0] * 100 if len(row_match) else 0
            fv_pct   = row_match["full_valid_ratio"].values[0]           * 100 if len(row_match) else 0
            ax.set_title(f"step={step_actual}\nsnap={snap_pct:.0f}%  valid={fv_pct:.0f}%", fontsize=8)
            ax.set_xlim(-1.25, 1.25)
            ax.set_xlabel("Cell value" if row == n_exps - 1 else "")
            if col_idx == 0: ax.set_ylabel(f"{label} density")
            ax.tick_params(labelsize=7)

    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"Figure saved → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze scalar quantization quality over training.")
    parser.add_argument("--exp_names", nargs="+", required=True)
    parser.add_argument("--ns",        nargs="+", type=int, required=True)
    parser.add_argument("--saveroot",  type=str,
                        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning")
    parser.add_argument("--outpath",   type=str, default="/tmp/scalar_quantization.png")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--n_steps",   type=int, default=40)
    args = parser.parse_args()

    if len(args.exp_names) != len(args.ns):
        parser.error("--exp_names and --ns must have the same length")

    dfs, labels, exp_dirs = [], [], []
    for exp_name, n in zip(args.exp_names, args.ns):
        exp_dir = os.path.join(args.saveroot, exp_name)
        print(f"\n{exp_name}  (n={n})")
        df = compute_or_load(exp_dir, n, recompute=args.recompute, n_steps=args.n_steps)
        dfs.append(df)
        labels.append(exp_name.split("DiT_mini_")[-1])
        exp_dirs.append(exp_dir)

    plot_scalar_analysis(dfs, labels, args.ns, exp_dirs, args.outpath)


if __name__ == "__main__":
    main()
