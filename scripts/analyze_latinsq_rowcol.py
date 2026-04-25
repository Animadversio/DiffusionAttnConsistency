"""
analyze_latinsq_rowcol.py

Focused analysis of row vs column accuracy asymmetry in Latin square experiments.
Compares scalar and one-hot encodings for n=5 and n=6.

Generates a 3×4 figure:
  Row 0: Training dynamics — full valid, row valid, col valid, row-col gap over training
  Row 1: Final accuracy bars | row-col gap bar | failure mode stacked bar | valid row/col distribution
  Row 2: Per-position failure rates for n=5 (left pair) and n=6 (right pair)

Usage:
  python analyze_latinsq_rowcol.py \\
      --saveroot /path/to/experiments \\
      --outpath /tmp/latinsq_rowcol.png

  # Custom subset:
  python analyze_latinsq_rowcol.py \\
      --exp_names DiT_mini_latinSq_n5_N4096_scalar DiT_mini_latinSq_n5_N4096_onehot \\
      --ns 5 5 --enc_types scalar onehot \\
      --outpath /tmp/latinsq_n5_rowcol.png
"""
import argparse, os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
from latin_square_lib import snap_to_integer, onehot_to_int, check_latin_square_batch

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False

# ---------------------------------------------------------------------------
# Default experiment set
# ---------------------------------------------------------------------------

DEFAULT_EXPS = [
    ("DiT_mini_latinSq_n5_N4096_scalar", 5, "scalar"),
    ("DiT_mini_latinSq_n5_N4096_onehot", 5, "onehot"),
    ("DiT_mini_latinSq_n6_N4096_scalar", 6, "scalar"),
    ("DiT_mini_latinSq_n6_N4096_onehot", 6, "onehot"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tb(exp_dir, tags):
    """Load scalar TensorBoard tags. Returns dict tag -> (steps, vals)."""
    if not HAS_TB:
        return {}
    tb_dir = os.path.join(exp_dir, "tensorboard")
    if not os.path.isdir(tb_dir):
        return {}
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    avail = ea.Tags().get("scalars", [])
    out = {}
    for tag in tags:
        if tag in avail:
            evs = ea.Scalars(tag)
            out[tag] = ([e.step for e in evs], [e.value for e in evs])
    return out


def load_final_samples(exp_dir, n, enc, epoch=None):
    """Decode final (or specified) samples and return per-sample row/col stats."""
    sdir = os.path.join(exp_dir, "samples")
    pt_files = sorted(glob.glob(os.path.join(sdir, "samples_epoch_*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt sample files in {sdir}")
    if epoch is None:
        fpath = pt_files[-1]
    else:
        fpath = os.path.join(sdir, f"samples_epoch_{epoch:06d}.pt")
    step = int(os.path.basename(fpath).replace("samples_epoch_", "").replace(".pt", ""))
    x = torch.load(fpath, map_location="cpu", weights_only=False)
    N = len(x)

    if enc == "scalar":
        xi = snap_to_integer(x.flatten(1).numpy(), n, eps=0.15)
    else:
        xi = onehot_to_int(x.numpy().reshape(N, n, n * n), n, eps=0.3)

    nan_mask = np.isnan(xi).any(axis=1)
    vi = xi[~nan_mask].astype(int)
    M  = len(vi)

    if M == 0:
        return dict(step=step, M=0, nan_ratio=1.0,
                    full=0., row=0., col=0.,
                    row_only=0., col_only=0., both_bad=0.,
                    row_fail_pos=np.zeros(n), col_fail_pos=np.zeros(n),
                    n_rows_valid=np.array([0]), n_cols_valid=np.array([0]))

    grids   = vi.reshape(M, n, n)
    sym     = np.arange(n)
    row_ok  = (np.sort(grids, axis=2) == sym[None, None, :]).all(axis=2)  # (M, n)
    col_ok  = (np.sort(grids, axis=1) == sym[None, :, None]).all(axis=1)  # (M, n)
    row_v   = row_ok.all(axis=1)
    col_v   = col_ok.all(axis=1)
    full_v  = row_v & col_v

    return dict(
        step         =step,
        M            =M,
        nan_ratio    =float(nan_mask.mean()),
        full         =float(full_v.mean()),
        row          =float(row_v.mean()),
        col          =float(col_v.mean()),
        row_only     =float((~row_v & col_v).mean()),
        col_only     =float((row_v & ~col_v).mean()),
        both_bad     =float((~row_v & ~col_v).mean()),
        row_fail_pos =(~row_ok).mean(axis=0),   # (n,) per-row failure rate
        col_fail_pos =(~col_ok).mean(axis=0),   # (n,)
        n_rows_valid =row_ok.sum(axis=1),        # (M,)
        n_cols_valid =col_ok.sum(axis=1),        # (M,)
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

TB_TAGS = ["eval/full_valid_ratio", "eval/row_valid_ratio", "eval/col_valid_ratio"]


def plot_rowcol_analysis(exp_list, saveroot, outpath, epoch=None):
    """
    exp_list : list of (exp_name, n, enc)
    """
    n_exps = len(exp_list)
    labels = []
    clrs   = COLORS[:n_exps]
    tb_data, final_data = [], []
    ns, encs = [], []

    for (exp_name, n, enc), clr in zip(exp_list, clrs):
        exp_dir = os.path.join(saveroot, exp_name)
        short   = exp_name.replace("DiT_mini_", "").replace("_N4096", "")
        labels.append(short)
        ns.append(n); encs.append(enc)
        print(f"Loading {short}…")
        tb_data.append(load_tb(exp_dir, TB_TAGS))
        final_data.append(load_final_samples(exp_dir, n, enc, epoch))
        step = final_data[-1]["step"]
        r = final_data[-1]
        print(f"  step={step}  full={r['full']:.3f}  row={r['row']:.3f}  col={r['col']:.3f}")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Latin Square: Row vs Column Accuracy Analysis — n=5 & n=6, Scalar vs One-hot",
                 fontsize=14, fontweight='bold')
    gs = plt.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)

    # ------------------------------------------------------------------
    # Row 0: Training dynamics
    # ------------------------------------------------------------------
    tag_info = [
        ("eval/full_valid_ratio", "Full Valid Ratio (training)"),
        ("eval/row_valid_ratio",  "Row Valid Ratio (training)"),
        ("eval/col_valid_ratio",  "Col Valid Ratio (training)"),
    ]
    for pi, (tag, title) in enumerate(tag_info):
        ax = fig.add_subplot(gs[0, pi])
        for i, (tb, clr, label, n, enc) in enumerate(zip(tb_data, clrs, labels, ns, encs)):
            ls   = "-" if enc == "scalar" else "--"
            mrkr = "o" if n == 5 else "s"
            if tag in tb:
                steps, vals = tb[tag]
                ax.plot(steps, vals, color=clr, linestyle=ls, linewidth=1.6,
                        marker=mrkr, markersize=1.5, alpha=0.9, label=label)
        ax.set_xscale("log"); ax.set_xlabel("Step"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25); ax.set_ylim(-0.02, 1.05)
        if pi == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Row-col gap over training
    ax = fig.add_subplot(gs[0, 3])
    for tb, clr, label, n, enc in zip(tb_data, clrs, labels, ns, encs):
        ls = "-" if enc == "scalar" else "--"; mrkr = "o" if n == 5 else "s"
        if "eval/row_valid_ratio" in tb and "eval/col_valid_ratio" in tb:
            sr, rv = tb["eval/row_valid_ratio"]
            sc, cv = tb["eval/col_valid_ratio"]
            ml = min(len(rv), len(cv))
            gap = np.array(rv[:ml]) - np.array(cv[:ml])
            ax.plot(sr[:ml], gap, color=clr, linestyle=ls, linewidth=1.6,
                    marker=mrkr, markersize=1.5, alpha=0.9, label=label)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xscale("log"); ax.set_xlabel("Step")
    ax.set_title("Row − Col Gap (training)\n(+: rows easier, −: cols easier)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Row 1: Final accuracy | gap bar | failure mode | distribution
    # ------------------------------------------------------------------

    # Accuracy bars (full, row, col)
    ax = fig.add_subplot(gs[1, 0])
    xp = np.arange(n_exps); w = 0.25
    ax.bar(xp - w, [r["full"] * 100 for r in final_data], w, color=clrs, alpha=0.9,  label="Full")
    ax.bar(xp,     [r["row"]  * 100 for r in final_data], w, color=clrs, alpha=0.55, label="Row")
    ax.bar(xp + w, [r["col"]  * 100 for r in final_data], w, color=clrs, alpha=0.25, label="Col")
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Final Accuracy\n(dark→full  mid→row  light→col)", fontsize=10)
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)
    for xi, r in enumerate(final_data):
        ax.text(xi - w, r["full"] * 100 + 0.5, f"{r['full']*100:.0f}",
                ha='center', va='bottom', fontsize=7)

    # Row-col gap bar
    ax = fig.add_subplot(gs[1, 1])
    gaps = [(r["row"] - r["col"]) * 100 for r in final_data]
    bars = ax.bar(range(n_exps), gaps, color=clrs, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax.set_ylabel("Row − Col acc (pp)")
    ax.set_title("Row vs Col Accuracy Gap (final)\n(+: rows easier; −: cols easier)", fontsize=10)
    ax.set_xticks(range(n_exps)); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.grid(alpha=0.3, axis='y')
    for bar, g in zip(bars, gaps):
        yoff = 0.3 if g >= 0 else -0.9
        ax.text(bar.get_x() + bar.get_width() / 2, g + yoff,
                f"{g:+.1f}pp", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Failure mode stacked bar
    ax = fig.add_subplot(gs[1, 2])
    ro = [r["row_only"] * 100 for r in final_data]
    co = [r["col_only"] * 100 for r in final_data]
    bb = [r["both_bad"] * 100 for r in final_data]
    ax.bar(range(n_exps), ro, label="Row-only fail",  color="#e377c2", alpha=0.9)
    ax.bar(range(n_exps), co, bottom=ro, label="Col-only fail", color="#17becf", alpha=0.9)
    ax.bar(range(n_exps), bb, bottom=[r + c for r, c in zip(ro, co)],
           label="Both fail", color="#bcbd22", alpha=0.9)
    ax.set_xticks(range(n_exps)); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("% of samples"); ax.set_title("Failure Mode Breakdown", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Valid row/col count distribution
    ax = fig.add_subplot(gs[1, 3])
    for r, clr, label, n, enc in zip(final_data, clrs, labels, ns, encs):
        ls = "-" if enc == "scalar" else "--"
        for arr, mrkr, alpha in [(r["n_rows_valid"], 'o', 0.9),
                                  (r["n_cols_valid"], 's', 0.5)]:
            if len(arr):
                u, c = np.unique(arr, return_counts=True)
                ax.plot(u, c / len(arr) * 100, mrkr + ls, color=clr,
                        lw=1.6, ms=5, alpha=alpha)
    handles = [mlines.Line2D([0],[0], color=c, ls=("-" if e=="scalar" else "--"), label=l)
               for c, e, l in zip(clrs, encs, labels)]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel("# valid rows (circles) / cols (squares) per sample")
    ax.set_ylabel("% samples")
    ax.set_title("Valid Row/Col Count Distribution\nper Sample", fontsize=10)
    ax.grid(alpha=0.25)

    # ------------------------------------------------------------------
    # Row 2: Per-position failure rates — n=5 (left) and n=6 (right)
    # ------------------------------------------------------------------
    for ni, ntgt in enumerate([5, 6]):
        ax_r = fig.add_subplot(gs[2, ni * 2])
        ax_c = fig.add_subplot(gs[2, ni * 2 + 1])
        any_plotted = False
        for r, clr, label, n, enc in zip(final_data, clrs, labels, ns, encs):
            if n != ntgt:
                continue
            ls = "-" if enc == "scalar" else "--"
            ax_r.plot(range(n), r["row_fail_pos"] * 100, 'o' + ls,
                      color=clr, lw=2, ms=7, label=label)
            ax_c.plot(range(n), r["col_fail_pos"] * 100, 's' + ls,
                      color=clr, lw=2, ms=7, label=label)
            any_plotted = True
        for ax, axis_name in [(ax_r, "Row"), (ax_c, "Col")]:
            ax.set_xlabel(f"{axis_name} index (0..{ntgt-1})")
            ax.set_ylabel("Failure rate (%)")
            ax.set_title(f"n={ntgt}: {axis_name} failure by position\n(flat → no spatial bias)",
                         fontsize=10)
            ax.grid(alpha=0.3); ax.set_xticks(range(ntgt))
            if any_plotted:
                ax.legend(fontsize=8)

    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\nFigure saved → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze row vs col accuracy asymmetry for Latin square experiments.")
    parser.add_argument("--saveroot", type=str,
                        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning")
    parser.add_argument("--outpath",  type=str, default="/tmp/latinsq_rowcol.png")
    parser.add_argument("--epoch",    type=int, default=None,
                        help="Epoch to analyze; default = last checkpoint")
    parser.add_argument("--exp_names", nargs="+", default=None,
                        help="Override default experiment list")
    parser.add_argument("--ns",       nargs="+", type=int, default=None)
    parser.add_argument("--enc_types", nargs="+", choices=["scalar","onehot"], default=None)
    args = parser.parse_args()

    if args.exp_names:
        if not (args.ns and args.enc_types):
            parser.error("With --exp_names, also provide --ns and --enc_types")
        exp_list = list(zip(args.exp_names, args.ns, args.enc_types))
    else:
        exp_list = DEFAULT_EXPS

    plot_rowcol_analysis(exp_list, args.saveroot, args.outpath, epoch=args.epoch)


if __name__ == "__main__":
    main()
