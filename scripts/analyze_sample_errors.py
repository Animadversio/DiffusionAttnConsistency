"""
analyze_sample_errors.py

Analyze error patterns in generated samples from the final (or specified) checkpoint.
Works for both exact-K and Latin square experiments.

Usage — Latin square:
  python analyze_sample_errors.py \\
      --group latinsq \\
      --saveroot /path/to/experiments \\
      --outpath /tmp/latinsq_errors.png

Usage — Exact-K:
  python analyze_sample_errors.py \\
      --group exactK \\
      --saveroot /path/to/experiments \\
      --outpath /tmp/exactK_errors.png

Usage — custom experiments:
  python analyze_sample_errors.py \\
      --exp_names DiT_mini_latinSq_n5_N4096_scalar DiT_mini_latinSq_n6_N4096_onehot \\
      --enc_types scalar onehot \\
      --ns 5 6 \\
      --saveroot /path/to/experiments \\
      --outpath /tmp/custom_errors.png
"""
import argparse, os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
from latin_square_lib import (snap_to_integer, onehot_to_int,
                               check_latin_square_batch, valid_float_values)

# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

EXACTK_EXPS = [
    {"name": "DiT_mini_exactK_N4096_D36_K3",  "type": "exactK", "k": 3},
    {"name": "DiT_mini_exactK_N4096_D36_K4",  "type": "exactK", "k": 4},
    {"name": "DiT_mini_exactK_N4096_D36_K6",  "type": "exactK", "k": 6},
    {"name": "DiT_mini_exactK_N4096_D36_K8",  "type": "exactK", "k": 8},
    {"name": "DiT_mini_exactK_N4096_D36_K9",  "type": "exactK", "k": 9},
    {"name": "DiT_mini_exactK_N4096_D36_K12", "type": "exactK", "k": 12},
    {"name": "DiT_mini_exactK_N4096_D36_K18", "type": "exactK", "k": 18},
]

LATINSQ_EXPS = [
    {"name": "DiT_mini_latinSq_n5_N4096_scalar", "type": "scalar", "n": 5},
    {"name": "DiT_mini_latinSq_n5_N4096_onehot", "type": "onehot", "n": 5},
    {"name": "DiT_mini_latinSq_n6_N4096_scalar", "type": "scalar", "n": 6},
    {"name": "DiT_mini_latinSq_n6_N4096_onehot", "type": "onehot", "n": 6},
]


# ---------------------------------------------------------------------------
# Per-experiment analysis
# ---------------------------------------------------------------------------

def load_last_samples(exp_dir, epoch=None):
    """Load sample tensor from exp_dir/samples/. If epoch is None, load last file."""
    sdir = os.path.join(exp_dir, "samples")
    pt_files = sorted(glob.glob(os.path.join(sdir, "samples_epoch_*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {sdir}")
    if epoch is None:
        fpath = pt_files[-1]
    else:
        fpath = os.path.join(sdir, f"samples_epoch_{epoch:06d}.pt")
    step = int(os.path.basename(fpath).replace("samples_epoch_","").replace(".pt",""))
    x = torch.load(fpath, map_location="cpu", weights_only=False)
    return x, step


def analyze_exactK(x, k):
    x_flat = x.flatten(1).numpy()          # (N, 36)
    ones   = (x_flat > 0).sum(axis=1)      # hard threshold at 0
    ambig  = (np.abs(x_flat) < 0.5).any(axis=1)
    return dict(
        ones_counts=ones,
        n_ambig=ambig.sum(),
        exact_ratio=(ones == k).mean(),
        N=len(x_flat),
    )


def analyze_latinsq(x, n, enc):
    N = len(x)
    if enc == "scalar":
        xi = snap_to_integer(x.flatten(1).numpy(), n, eps=0.15)
    else:
        xi = onehot_to_int(x.numpy().reshape(N, n, n * n), n, eps=0.3)

    nm = np.isnan(xi).any(axis=1)
    vi = xi[~nm].astype(int)
    M  = len(vi)

    if M == 0:
        return dict(nan_ratio=1.0, full=0., row=0., col=0.,
                    row_only=0., col_only=0., both_bad=0.,
                    row_fail_pos=np.zeros(n), col_fail_pos=np.zeros(n),
                    n_rows_valid=np.array([]), n_cols_valid=np.array([]))

    grids   = vi.reshape(M, n, n)
    sym     = np.arange(n)
    row_ok  = (np.sort(grids, axis=2) == sym[None, None, :]).all(axis=2)  # (M, n)
    col_ok  = (np.sort(grids, axis=1) == sym[None, :, None]).all(axis=1)  # (M, n)
    row_v   = row_ok.all(axis=1); col_v = col_ok.all(axis=1)
    full_v  = row_v & col_v

    return dict(
        nan_ratio     =float(nm.mean()),
        full          =float(full_v.mean()),
        row           =float(row_v.mean()),
        col           =float(col_v.mean()),
        row_only      =float((~row_v & col_v).mean()),
        col_only      =float((row_v & ~col_v).mean()),
        both_bad      =float((~row_v & ~col_v).mean()),
        row_fail_pos  =(~row_ok).mean(axis=0),
        col_fail_pos  =(~col_ok).mean(axis=0),
        n_rows_valid  =row_ok.sum(axis=1),
        n_cols_valid  =col_ok.sum(axis=1),
    )


# ---------------------------------------------------------------------------
# Plotting — exact-K
# ---------------------------------------------------------------------------

def plot_exactK_errors(results, ks, outpath):
    colors_k = plt.cm.plasma(np.linspace(0.1, 0.9, len(ks)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exact-K Error Analysis — Final Checkpoint", fontsize=13, fontweight='bold')

    # Panel 0: ones distribution
    ax = axes[0]
    for i, (k, r) in enumerate(zip(ks, results)):
        u, c = np.unique(r["ones_counts"], return_counts=True)
        ax.bar(u + i * 0.1 - len(ks) * 0.05, c / r["N"] * 100, 0.09,
               color=colors_k[i], label=f"K={k}", alpha=0.85)
    ax.set_xlabel("Number of Ones"); ax.set_ylabel("% samples")
    ax.set_title("Ones Count Distribution\n(errors always off-by-one)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis='y')

    # Panel 1: accuracy bar
    ax = axes[1]
    accs = [r["exact_ratio"] * 100 for r in results]
    bars = ax.bar([str(k) for k in ks], accs, color=colors_k, alpha=0.85)
    ax.set_ylim(75, 101); ax.set_xlabel("K"); ax.set_ylabel("Exact-K accuracy (%)")
    ax.set_title("Final Accuracy by K"); ax.grid(alpha=0.3, axis='y')
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{a:.1f}", ha='center', va='bottom', fontsize=8)

    # Panel 2: error magnitude (Δ ones)
    ax = axes[2]
    for i, (k, r) in enumerate(zip(ks, results)):
        u, c = np.unique(r["ones_counts"] - k, return_counts=True)
        ax.bar(u + i * 0.1 - len(ks) * 0.05, c / r["N"] * 100, 0.09,
               color=colors_k[i], label=f"K={k}", alpha=0.85)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Δ ones (generated − target K)")
    ax.set_ylabel("% samples"); ax.set_xlim(-3.5, 3.5)
    ax.set_title("Error Magnitude\n(never off by 2+)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"Figure saved → {outpath}")


# ---------------------------------------------------------------------------
# Plotting — Latin square
# ---------------------------------------------------------------------------

LS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_latinsq_errors(results, labels, ns, encs, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Latin Square Error Analysis — Final Checkpoint", fontsize=13, fontweight='bold')

    clrs = LS_COLORS[:len(results)]

    # Panel 0: accuracy bars (full, row, col)
    ax = axes[0, 0]
    xp = np.arange(len(results)); w = 0.25
    ax.bar(xp - w, [r["full"] * 100 for r in results], w, color=clrs, alpha=0.9, label="Full")
    ax.bar(xp,     [r["row"]  * 100 for r in results], w, color=clrs, alpha=0.55, label="Row")
    ax.bar(xp + w, [r["col"]  * 100 for r in results], w, color=clrs, alpha=0.25, label="Col")
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Final Accuracy\n(dark→full  mid→row  light→col)", fontsize=10)
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)
    for xi, r in enumerate(results):
        ax.text(xi - w, r["full"] * 100 + 0.3, f"{r['full']*100:.0f}", ha='center', va='bottom', fontsize=7)

    # Panel 1: row-col gap
    ax = axes[0, 1]
    gaps = [(r["row"] - r["col"]) * 100 for r in results]
    bars = ax.bar(labels, gaps, color=clrs, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax.set_ylabel("Row − Col acc (pp)")
    ax.set_title("Row vs Col Gap\n(+: rows easier; −: cols easier)", fontsize=10)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.grid(alpha=0.3, axis='y')
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, g + (0.3 if g >= 0 else -0.8),
                f"{g:+.1f}", ha='center', va='bottom', fontsize=9)

    # Panel 2: failure mode stacked bar
    ax = axes[0, 2]
    ro = [r["row_only"] * 100 for r in results]
    co = [r["col_only"] * 100 for r in results]
    bb = [r["both_bad"] * 100 for r in results]
    ax.bar(range(len(results)), ro, label="Row-only fail", color="#e377c2", alpha=0.9)
    ax.bar(range(len(results)), co, bottom=ro, label="Col-only fail", color="#17becf", alpha=0.9)
    ax.bar(range(len(results)), bb, bottom=[r + c for r, c in zip(ro, co)],
           label="Both fail", color="#bcbd22", alpha=0.9)
    ax.set_xticks(range(len(results))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("% samples"); ax.set_title("Failure Mode Breakdown", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Panel 3: valid row/col count distribution
    ax = axes[1, 0]
    from matplotlib.lines import Line2D
    for ri, (r, clr, label) in enumerate(zip(results, clrs, labels)):
        nr = r["n_rows_valid"]; nc = r["n_cols_valid"]
        n  = ns[ri]
        ls = "-" if encs[ri] == "scalar" else "--"
        for arr, mrkr, alpha in [(nr, 'o', 0.9), (nc, 's', 0.5)]:
            u, c = np.unique(arr, return_counts=True)
            ax.plot(u, c / len(arr) * 100, mrkr + ls, color=clr, lw=1.6, ms=5, alpha=alpha)
    handles = [Line2D([0],[0], color=c, ls=("-" if e=="scalar" else "--"), label=l)
               for c, e, l in zip(clrs, encs, labels)]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel("# valid rows (o) / cols (s) per sample")
    ax.set_ylabel("% samples"); ax.set_title("Valid Row/Col Count per Sample", fontsize=10)
    ax.grid(alpha=0.25)

    # Panels 4-5: per-position failure rates for n=5 and n=6
    for panel_i, ntgt in enumerate([5, 6]):
        ax = axes[1, 1 + panel_i]
        plotted = False
        for ri, (r, clr, label, n, enc) in enumerate(zip(results, clrs, labels, ns, encs)):
            if n != ntgt: continue
            ls = "-" if enc == "scalar" else "--"
            ax.plot(range(n), r["row_fail_pos"] * 100, 'o' + ls, color=clr, lw=2, ms=6, label=f"{label} row")
            ax.plot(range(n), r["col_fail_pos"] * 100, 's' + ls, color=clr, lw=2, ms=6, alpha=0.6, label=f"{label} col")
            plotted = True
        ax.set_xlabel(f"Position (0..{ntgt-1})")
        ax.set_ylabel("Failure rate (%)")
        ax.set_title(f"n={ntgt}: Failure by Position\n(flat → no spatial bias)", fontsize=10)
        if plotted: ax.legend(fontsize=7)
        ax.grid(alpha=0.3); ax.set_xticks(range(ntgt))

    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"Figure saved → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze error patterns in generated samples.")
    parser.add_argument("--group", choices=["exactK", "latinsq"],
                        help="Use a pre-defined experiment group")
    parser.add_argument("--exp_names", nargs="+", help="Custom experiment names")
    parser.add_argument("--enc_types", nargs="+", choices=["scalar", "onehot"],
                        help="Encoding type per experiment (for latinsq custom)")
    parser.add_argument("--ns",        nargs="+", type=int,
                        help="n value per experiment (for latinsq custom)")
    parser.add_argument("--ks",        nargs="+", type=int,
                        help="k value per experiment (for exactK custom)")
    parser.add_argument("--saveroot",  type=str,
                        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning")
    parser.add_argument("--outpath",   type=str, default="/tmp/sample_errors.png")
    parser.add_argument("--epoch",     type=int, default=None,
                        help="Epoch to load; if None, uses last checkpoint")
    args = parser.parse_args()

    if args.group == "exactK":
        exps = EXACTK_EXPS
        ks   = [e["k"] for e in exps]
        results = []
        for e in exps:
            exp_dir = os.path.join(args.saveroot, e["name"])
            x, step = load_last_samples(exp_dir, args.epoch)
            print(f"{e['name']}: step={step}")
            results.append(analyze_exactK(x, e["k"]))
        plot_exactK_errors(results, ks, args.outpath)

    elif args.group == "latinsq":
        exps = LATINSQ_EXPS
        labels, ns, encs, results = [], [], [], []
        for e in exps:
            exp_dir = os.path.join(args.saveroot, e["name"])
            try:
                x, step = load_last_samples(exp_dir, args.epoch)
            except FileNotFoundError:
                print(f"Skipping {e['name']} — no samples found")
                continue
            print(f"{e['name']}: step={step}")
            r = analyze_latinsq(x, e["n"], e["type"])
            results.append(r); labels.append(e["name"].split("DiT_mini_")[-1])
            ns.append(e["n"]); encs.append(e["type"])
        plot_latinsq_errors(results, labels, ns, encs, args.outpath)

    else:
        # custom
        if not args.exp_names:
            parser.error("Provide --group or --exp_names")
        if args.enc_types:
            # latin square mode
            labels, ns, encs, results = [], [], [], []
            for exp_name, enc, n in zip(args.exp_names, args.enc_types, args.ns):
                exp_dir = os.path.join(args.saveroot, exp_name)
                x, step = load_last_samples(exp_dir, args.epoch)
                print(f"{exp_name}: step={step}")
                results.append(analyze_latinsq(x, n, enc))
                labels.append(exp_name); ns.append(n); encs.append(enc)
            plot_latinsq_errors(results, labels, ns, encs, args.outpath)
        elif args.ks:
            results = []
            for exp_name, k in zip(args.exp_names, args.ks):
                exp_dir = os.path.join(args.saveroot, exp_name)
                x, step = load_last_samples(exp_dir, args.epoch)
                print(f"{exp_name}: step={step}")
                results.append(analyze_exactK(x, k))
            plot_exactK_errors(results, args.ks, args.outpath)
        else:
            parser.error("For custom mode provide --enc_types (latinsq) or --ks (exactK)")


if __name__ == "__main__":
    main()
