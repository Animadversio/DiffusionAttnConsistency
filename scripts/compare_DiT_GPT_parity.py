"""
compare_DiT_GPT_parity.py

Compare DiT vs GPT on even parity learning at matched N values.
Produces two figure types:
  1. Trajectory grid  — loss, valid_acc, mem_ratio, innovation vs log-step for all G values at a given N
  2. Speed summary    — step to reach threshold valid_acc, DiT vs GPT, for each (N, G)

Model pairs (--pair):
  nano  : DiT-nano (3L 6H 384D) vs GPT-nano (3L 6H 384D), N=4096 only
  mini  : DiT-mini (6L 6H 384D) vs GPT-mini (6L 6H 384D), N=4096/8192/16384

Usage:
  python scripts/compare_DiT_GPT_parity.py --outdir /tmp/dit_gpt_compare --pair mini
  python scripts/compare_DiT_GPT_parity.py --outdir /tmp/dit_gpt_compare --pair nano
  python scripts/compare_DiT_GPT_parity.py --outdir /tmp/dit_gpt_compare --pair mini --N 4096
  python scripts/compare_DiT_GPT_parity.py --outdir /tmp/dit_gpt_compare --pair mini --mode summary
"""

import os
import csv
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"

G_VALUES  = [2, 3, 4, 6, 9, 12, 18, 36]
THRESHOLD = 0.80   # valid_acc threshold for "learned"

# Model pair definitions: (dit_prefix, gpt_prefix, matched_N_list, dit_color, gpt_color, dit_label, gpt_label)
MODEL_PAIRS = {
    "nano": ("DiT_nano", "GPT_nano", [4096],              "steelblue", "tomato", "DiT-nano", "GPT-nano"),
    "mini": ("DiT_mini", "GPT_mini", [4096, 8192, 16384], "steelblue", "tomato", "DiT-mini", "GPT-mini"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dit_parity(N, G, prefix="DiT_mini"):
    """Load DiT parity run. Returns dict of arrays."""
    exp = f"{prefix}_parity_N{N}_D36_G{G}_even"
    d   = os.path.join(SAVEROOT, exp)
    if not os.path.isdir(d):
        return None

    # Loss trajectory (full 1M-step list — downsample)
    loss_path = os.path.join(d, "loss_traj.pkl")
    if os.path.exists(loss_path):
        with open(loss_path, "rb") as f:
            lt = pickle.load(f)
        lt = np.array(lt, dtype=np.float32)
        stride = max(1, len(lt) // 2000)
        loss_steps = np.arange(1, len(lt)+1)[::stride]
        loss_vals  = lt[::stride]
    else:
        loss_steps = loss_vals = np.array([])

    # Eval CSV (validity + mem)
    eval_path = os.path.join(d, "mem_eval_stats.csv")
    if not os.path.exists(eval_path):
        eval_path = os.path.join(d, "pergroup_eval_stats.csv")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            rows = list(csv.DictReader(f))
        eval_steps = np.array([int(r["step"]) for r in rows])
        valid_acc  = np.array([float(r["sample_corr_acc"]) for r in rows])
        mem_ratio  = np.array([float(r.get("sample_mem_ratio", "nan")) for r in rows])
    else:
        eval_steps = valid_acc = mem_ratio = np.array([])

    return dict(loss_steps=loss_steps, loss_vals=loss_vals,
                eval_steps=eval_steps, valid_acc=valid_acc, mem_ratio=mem_ratio,
                max_steps=1_000_000)


def load_gpt_parity(N, G, prefix="GPT_mini"):
    """Load GPT parity run from TensorBoard. Returns dict of arrays."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    exp    = f"{prefix}_parity_N{N}_D36_G{G}_even"
    tb_dir = os.path.join(SAVEROOT, exp, "tensorboard")
    if not os.path.isdir(tb_dir):
        return None

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()

    def get(tag):
        try:
            evs = ea.Scalars(tag)
            return np.array([e.step for e in evs]), np.array([e.value for e in evs])
        except:
            return np.array([]), np.array([])

    ls, lv  = get("Training/Loss_Step")
    vs, vv  = get("Eval/Sample_Accuracy")
    ms, mv  = get("Eval/Sample_Mem_Ratio")

    # Align valid and mem to same steps (use shorter of the two)
    n = min(len(vs), len(ms))
    vs, vv, ms, mv = vs[:n], vv[:n], ms[:n], mv[:n]

    return dict(loss_steps=ls, loss_vals=lv,
                eval_steps=vs, valid_acc=vv, mem_ratio=mv,
                max_steps=100_000)


def ewm_smooth(vals, alpha=0.9):
    if len(vals) == 0:
        return vals
    out = np.empty_like(vals, dtype=float)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * out[i-1] + (1 - alpha) * vals[i]
    return out


def step_to_threshold(steps, vals, thr=THRESHOLD):
    """Return first step where vals >= thr, or None."""
    if len(steps) == 0:
        return None
    idx = np.where(vals >= thr)[0]
    return int(steps[idx[0]]) if len(idx) > 0 else None


# ---------------------------------------------------------------------------
# Figure 1: trajectory grid for one N value
# ---------------------------------------------------------------------------

def plot_trajectory_grid(N, outdir, dit_prefix="DiT_mini", gpt_prefix="GPT_mini",
                         dit_color="steelblue", gpt_color="tomato",
                         dit_label="DiT-mini", gpt_label="GPT-mini"):
    n_g = len(G_VALUES)
    fig, axes = plt.subplots(n_g, 4, figsize=(19, 2.8 * n_g), sharex=False)
    fig.suptitle(f"{dit_label} vs {gpt_label}  |  N={N:,}  |  Even Parity D=36",
                 fontsize=13, fontweight="bold")

    col_titles = ["Loss", "Valid acc (sample)", "Mem ratio", "Innovation (valid − mem)"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11)

    for ri, G in enumerate(G_VALUES):
        dit = load_dit_parity(N, G, prefix=dit_prefix)
        gpt = load_gpt_parity(N, G, prefix=gpt_prefix)

        for ci, (metric, ylabel) in enumerate([
            ("loss",  "Loss"),
            ("valid", "Valid acc"),
            ("mem",   "Mem ratio"),
            ("innov", "Innovation"),
        ]):
            ax = axes[ri, ci]

            if dit is not None:
                if metric == "loss" and len(dit["loss_steps"]) > 0:
                    sm = ewm_smooth(dit["loss_vals"], alpha=0.95)
                    ax.plot(dit["loss_steps"], dit["loss_vals"], color=dit_color, alpha=0.15, lw=0.7)
                    ax.plot(dit["loss_steps"], sm, color=dit_color, lw=1.6, label=dit_label)
                elif metric == "valid" and len(dit["eval_steps"]) > 0:
                    ax.plot(dit["eval_steps"], dit["valid_acc"], color=dit_color, lw=1.8,
                            marker=".", markersize=2, label=dit_label)
                elif metric == "mem" and len(dit["eval_steps"]) > 0:
                    ax.plot(dit["eval_steps"], dit["mem_ratio"], color=dit_color, lw=1.8,
                            marker=".", markersize=2, label=dit_label)
                elif metric == "innov" and len(dit["eval_steps"]) > 0:
                    innov = np.clip(dit["valid_acc"] - dit["mem_ratio"], 0, 1)
                    ax.plot(dit["eval_steps"], innov, color=dit_color, lw=1.8,
                            marker=".", markersize=2, label=dit_label)

            if gpt is not None:
                if metric == "loss" and len(gpt["loss_steps"]) > 0:
                    sm = ewm_smooth(gpt["loss_vals"], alpha=0.9)
                    ax.plot(gpt["loss_steps"], gpt["loss_vals"], color=gpt_color, alpha=0.15, lw=0.7)
                    ax.plot(gpt["loss_steps"], sm, color=gpt_color, lw=1.6, label=gpt_label)
                elif metric == "valid" and len(gpt["eval_steps"]) > 0:
                    ax.plot(gpt["eval_steps"], gpt["valid_acc"], color=gpt_color, lw=1.8,
                            marker=".", markersize=2, label=gpt_label)
                elif metric == "mem" and len(gpt["eval_steps"]) > 0:
                    ax.plot(gpt["eval_steps"], gpt["mem_ratio"], color=gpt_color, lw=1.8,
                            marker=".", markersize=2, label=gpt_label)
                elif metric == "innov" and len(gpt["eval_steps"]) > 0:
                    innov = np.clip(gpt["valid_acc"] - gpt["mem_ratio"], 0, 1)
                    ax.plot(gpt["eval_steps"], innov, color=gpt_color, lw=1.8,
                            marker=".", markersize=2, label=gpt_label)

            # Innovation panel: fixed ylim + random baseline
            if metric == "innov":
                ax.set_ylim(-0.02, 1.05)
                # P(all groups satisfy parity) for uniform random ±1 sequence
                random_baseline = 0.5 ** (36 / G)
                ax.axhline(random_baseline, color="gray", lw=1.2, ls="--", alpha=0.8,
                           label=f"random ({random_baseline:.3f})")
                if ri == 0:
                    ax.legend(fontsize=7, loc="upper left")

            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            if ci == 0:
                ax.set_ylabel(f"G={G}", fontsize=9)
            if ri == n_g - 1:
                ax.set_xlabel("Step", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    pair_tag = f"{dit_label.replace('-','').replace(' ','_')}_vs_{gpt_label.replace('-','').replace(' ','_')}"
    out = os.path.join(outdir, f"trajectory_{pair_tag}_N{N}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: speed summary — step to threshold, DiT vs GPT
# ---------------------------------------------------------------------------

def plot_speed_summary(outdir, dit_prefix="DiT_mini", gpt_prefix="GPT_mini",
                       n_list=None, dit_label="DiT-mini", gpt_label="GPT-mini"):
    if n_list is None:
        n_list = [4096, 8192, 16384]
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5), sharey=False)
    if len(n_list) == 1:
        axes = [axes]
    fig.suptitle(f"Steps to reach {THRESHOLD*100:.0f}% valid accuracy — {dit_label} vs {gpt_label}",
                 fontsize=13, fontweight="bold")

    for ai, N in enumerate(n_list):
        ax = axes[ai]
        dit_steps, gpt_steps = [], []
        valid_g = []

        for G in G_VALUES:
            dit = load_dit_parity(N, G, prefix=dit_prefix)
            gpt = load_gpt_parity(N, G, prefix=gpt_prefix)
            ds = step_to_threshold(dit["eval_steps"], dit["valid_acc"]) if dit else None
            gs = step_to_threshold(gpt["eval_steps"], gpt["valid_acc"]) if gpt else None
            dit_steps.append(ds)
            gpt_steps.append(gs)
            valid_g.append(G)

        x = np.arange(len(valid_g))
        w = 0.35
        dit_y = [s if s is not None else np.nan for s in dit_steps]
        gpt_y = [s if s is not None else np.nan for s in gpt_steps]

        bars1 = ax.bar(x - w/2, dit_y, w, label=dit_label, color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + w/2, gpt_y, w, label=gpt_label, color="tomato", alpha=0.85)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"G={g}" for g in valid_g], rotation=45, ha="right")
        ax.set_title(f"N={N:,}", fontsize=11)
        ax.set_ylabel("Steps to 80% valid acc (log scale)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        # Annotate "never reached" as text
        for xi, (ds, gs) in enumerate(zip(dit_steps, gpt_steps)):
            if ds is None:
                ax.text(xi - w/2, ax.get_ylim()[0] * 2, "✗", ha="center", color="steelblue", fontsize=9)
            if gs is None:
                ax.text(xi + w/2, ax.get_ylim()[0] * 2, "✗", ha="center", color="tomato", fontsize=9)

    plt.tight_layout()
    out = os.path.join(outdir, "speed_summary.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: final valid_acc and mem_ratio heatmaps
# ---------------------------------------------------------------------------

def plot_final_heatmap(outdir, dit_prefix="DiT_mini", gpt_prefix="GPT_mini",
                       n_list=None, dit_label="DiT-mini", gpt_label="GPT-mini"):
    if n_list is None:
        n_list = [4096, 8192, 16384]
    fig, axes = plt.subplots(2, len(n_list), figsize=(6 * len(n_list), 8))
    if len(n_list) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f"Final valid acc (top) and mem ratio (bottom) — {dit_label} vs {gpt_label}",
                 fontsize=12, fontweight="bold")

    for ai, N in enumerate(n_list):
        dit_valid, gpt_valid = [], []
        dit_mem,   gpt_mem   = [], []

        for G in G_VALUES:
            dit = load_dit_parity(N, G, prefix=dit_prefix)
            gpt = load_gpt_parity(N, G, prefix=gpt_prefix)
            dit_valid.append(dit["valid_acc"][-1] if dit and len(dit["valid_acc"]) else np.nan)
            gpt_valid.append(gpt["valid_acc"][-1] if gpt and len(gpt["valid_acc"]) else np.nan)
            dit_mem.append(dit["mem_ratio"][-1]   if dit and len(dit["mem_ratio"])  else np.nan)
            gpt_mem.append(gpt["mem_ratio"][-1]   if gpt and len(gpt["mem_ratio"])  else np.nan)

        data_valid = np.array([dit_valid, gpt_valid])   # (2, 8)
        data_mem   = np.array([dit_mem,   gpt_mem])

        for row, (data, label) in enumerate([(data_valid, "Valid acc"), (data_mem, "Mem ratio")]):
            ax = axes[row, ai]
            im = ax.imshow(data, aspect="auto", vmin=0, vmax=1,
                           cmap="RdYlGn" if row == 0 else "RdYlGn_r")
            ax.set_xticks(range(len(G_VALUES)))
            ax.set_xticklabels([f"G={g}" for g in G_VALUES], rotation=45, ha="right", fontsize=8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels([dit_label, gpt_label], fontsize=9)
            ax.set_title(f"{label}  N={N:,}", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            # Annotate values
            for yi in range(2):
                for xi in range(len(G_VALUES)):
                    v = data[yi, xi]
                    if not np.isnan(v):
                        ax.text(xi, yi, f"{v:.2f}", ha="center", va="center",
                                fontsize=7.5, color="black")

    plt.tight_layout()
    out = os.path.join(outdir, "final_heatmap.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: compact multi-row overview
#   Each row = one (pair, N) combination
#   4 panels: loss / valid / mem / innovation
#   G values color-coded; DiT = solid, GPT = dashed
# ---------------------------------------------------------------------------

# Color progression for G: small G = cool (blue/purple), large G = warm (yellow/red)
# Uses plasma colormap for a smooth perceptual gradient
_G_CMAP = plt.cm.plasma
_G_POSITIONS = np.linspace(0.05, 0.88, len(G_VALUES))   # avoid too-light yellow end
G_COLORS = {G: _G_CMAP(p) for G, p in zip(G_VALUES, _G_POSITIONS)}

# DiT: solid, full saturation; GPT: dashed, same hue but desaturated+lighter
def _gpt_color(rgba, mix=0.45):
    """Blend RGBA color toward white to give a lighter GPT shade."""
    r, g, b, a = rgba
    return (r + (1-r)*mix, g + (1-g)*mix, b + (1-b)*mix, a)


def plot_compact(rows, outdir, outname="compact_overview.png"):
    """
    rows: list of dicts with keys:
        dit_prefix, gpt_prefix, N, dit_label, gpt_label, row_label
    Each row produces one row of 4 panels.
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 3.0 * n_rows), squeeze=False)
    col_titles = ["Loss", "Valid acc", "Mem ratio", "Innovation (valid − mem)"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11)

    random_baseline = {G: 0.5 ** (36 / G) for G in G_VALUES}

    for ri, row in enumerate(rows):
        dit_prefix = row["dit_prefix"]
        gpt_prefix = row["gpt_prefix"]
        N          = row["N"]
        row_label  = row.get("row_label", f"{row['dit_label']} vs {row['gpt_label']}  N={N:,}")

        for ci, metric in enumerate(["loss", "valid", "mem", "innov"]):
            ax = axes[ri, ci]

            for G in G_VALUES:
                col = G_COLORS[G]
                dit = load_dit_parity(N, G, prefix=dit_prefix)
                gpt = load_gpt_parity(N, G, prefix=gpt_prefix)

                lbl_dit = f"G={G}" if ci == 0 and ri == 0 else "_"
                lbl_gpt = "_"
                col_dit = col
                col_gpt = _gpt_color(col)

                def _get(d, m, is_dit):
                    if d is None: return None, None
                    if m == "loss":  return d["loss_steps"], ewm_smooth(d["loss_vals"], alpha=0.95 if is_dit else 0.9)
                    if m == "valid": return d["eval_steps"], d["valid_acc"]
                    if m == "mem":   return d["eval_steps"], d["mem_ratio"]
                    if m == "innov": return d["eval_steps"], np.clip(d["valid_acc"] - d["mem_ratio"], 0, 1)
                    return None, None

                xs, ys = _get(dit, metric, True)
                if xs is not None and len(xs) > 0:
                    ax.plot(xs, ys, color=col_dit, lw=1.6, ls="-",  alpha=0.90, label=lbl_dit)

                xs, ys = _get(gpt, metric, False)
                if xs is not None and len(xs) > 0:
                    ax.plot(xs, ys, color=col_gpt, lw=1.6, ls="--", alpha=0.85, label=lbl_gpt)

            ax.set_xscale("log")
            ax.grid(alpha=0.2)

            if metric == "innov":
                ax.set_ylim(-0.02, 1.05)
                # draw random baselines per G as tiny colored ticks on right
                for G in G_VALUES:
                    ax.axhline(random_baseline[G], color=G_COLORS[G], lw=0.7, ls=":", alpha=0.6)

            if ci == 0:
                ax.set_ylabel(row_label, fontsize=8.5, labelpad=4)
            if ri == n_rows - 1:
                ax.set_xlabel("Step", fontsize=9)

    # Legend: G color progression + DiT/GPT style
    handles = [plt.Line2D([0], [0], color=G_COLORS[G], lw=2.0, ls="-", label=f"G={G}") for G in G_VALUES]
    handles += [
        plt.Line2D([0], [0], color="0.3", lw=1.8, ls="-",  label="DiT (saturated)"),
        plt.Line2D([0], [0], color="0.6", lw=1.8, ls="--", label="GPT (light+dashed)"),
    ]
    axes[0, 3].legend(handles=handles, fontsize=7.5, ncol=1, loc="upper left",
                      framealpha=0.85, title="G value", title_fontsize=8)

    fig.suptitle("DiT vs GPT — Even Parity D=36 (compact, color=G, solid=DiT, dashed=GPT)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(outdir, outname)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Time-heatmap: x=log-step, y=G, color=metric
# ---------------------------------------------------------------------------

def plot_timeheatmap(compact_rows, outdir, outname="timeheatmap.png"):
    """
    For each row spec in compact_rows, produce a 2×3 heatmap block:
      rows    = [DiT, GPT]
      columns = [valid_acc, mem_ratio, innovation]
      x-axis  = log step (shared grid 1 → 1e6)
      y-axis  = G value (G_VALUES rows)
      color   = metric value in [0, 1]

    All compact_rows are stacked vertically (one 2×3 block per row spec),
    separated by a thin gap.
    """
    STEP_GRID  = np.geomspace(1, 1_000_000, 400)
    METRICS    = ["valid", "mem", "innov"]
    MET_LABELS = ["Valid fraction", "Mem ratio", "Innovation (valid − mem)"]
    CMAPS      = ["viridis", "Reds", "Blues"]
    VRANGES    = [(0, 1), (0, 1), (0, 1)]

    n_rows   = len(compact_rows)
    n_blocks = n_rows          # one 2-row block per compact_row
    fig_rows = n_blocks * 2    # DiT row + GPT row per block
    fig_cols = 3

    fig_h = 1.6 * fig_rows + 0.4 * n_blocks
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * 5.5, fig_h),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.28})

    fig.suptitle("DiT vs GPT — Even Parity D=36\n(heatmap: x=log-step, y=rule complexity G)",
                 fontsize=13, fontweight="bold")

    def interp_metric(data, metric):
        if data is None:
            return np.full(len(STEP_GRID), np.nan)
        xs = np.array(data["eval_steps"], dtype=float)
        if metric == "valid":   ys = np.array(data["valid_acc"])
        elif metric == "mem":   ys = np.array(data["mem_ratio"])
        else:                   ys = np.clip(np.array(data["valid_acc"]) - np.array(data["mem_ratio"]), 0, 1)
        if len(xs) == 0:
            return np.full(len(STEP_GRID), np.nan)
        # Only interpolate within data range; fill outside with nan
        out = np.interp(STEP_GRID, xs, ys, left=np.nan, right=ys[-1])
        out[STEP_GRID < xs[0]] = np.nan
        return out

    for bi, row_spec in enumerate(compact_rows):
        dit_prefix = row_spec["dit_prefix"]
        gpt_prefix = row_spec["gpt_prefix"]
        N          = row_spec["N"]
        dit_label  = row_spec.get("dit_label", dit_prefix)
        gpt_label  = row_spec.get("gpt_label", gpt_prefix)
        row_label  = row_spec.get("row_label", f"N={N:,}")

        for mi, (metric, met_label, cmap, (vmin, vmax)) in enumerate(
                zip(METRICS, MET_LABELS, CMAPS, VRANGES)):

            # Build heatmap matrices (n_G × n_steps) for DiT and GPT
            dit_mat = np.zeros((len(G_VALUES), len(STEP_GRID)))
            gpt_mat = np.zeros((len(G_VALUES), len(STEP_GRID)))
            for gi, G in enumerate(G_VALUES):
                dit = load_dit_parity(N, G, prefix=dit_prefix)
                gpt = load_gpt_parity(N, G, prefix=gpt_prefix)
                dit_mat[gi] = interp_metric(dit, metric)
                gpt_mat[gi] = interp_metric(gpt, metric)

            for model_idx, (mat, model_label) in enumerate(
                    [(dit_mat, dit_label), (gpt_mat, gpt_label)]):
                ax_row = bi * 2 + model_idx
                ax = axes[ax_row, mi]

                im = ax.pcolormesh(
                    STEP_GRID, np.arange(len(G_VALUES)),
                    mat, cmap=cmap, vmin=vmin, vmax=vmax,
                    shading="nearest")
                ax.set_xscale("log")
                ax.set_xlim(STEP_GRID[0], STEP_GRID[-1])
                ax.set_yticks(np.arange(len(G_VALUES)))
                ax.set_yticklabels([f"G={G}" for G in G_VALUES], fontsize=7.5)
                ax.set_xlabel("Step", fontsize=8)

                # Add innovation baseline dashed lines (only for innovation panel)
                if metric == "innov":
                    n_groups = 36  # D=36
                    for gi, G in enumerate(G_VALUES):
                        baseline = 0.5 ** (n_groups // G)
                        # Draw as a small tick-mark on the right edge
                        ax.axhline(gi, color="white", lw=0.3, alpha=0.3)

                # Colorbar on rightmost metric column
                if mi == fig_cols - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Row + metric title
                if mi == 0:
                    ax.set_ylabel(f"{row_label}\n{model_label}", fontsize=8, labelpad=4)
                if ax_row == 0:
                    ax.set_title(met_label, fontsize=10, fontweight="bold")

    outpath = os.path.join(outdir, outname)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved → {outpath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/tmp/dit_gpt_compare")
    parser.add_argument("--N",    type=int, default=None, help="Only plot this N value")
    parser.add_argument("--mode", choices=["all", "trajectory", "summary", "heatmap", "compact", "timeheatmap"],
                        default="all")
    parser.add_argument("--pair", choices=list(MODEL_PAIRS.keys()), default="mini",
                        help="Model pair to compare")
    args = parser.parse_args()

    dit_prefix, gpt_prefix, n_matched, dit_color, gpt_color, dit_label, gpt_label = MODEL_PAIRS[args.pair]
    ns = [args.N] if args.N else n_matched

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode in ("all", "trajectory"):
        for N in ns:
            print(f"Trajectory grid {dit_label} vs {gpt_label} N={N}...")
            plot_trajectory_grid(N, args.outdir,
                                 dit_prefix=dit_prefix, gpt_prefix=gpt_prefix,
                                 dit_color=dit_color, gpt_color=gpt_color,
                                 dit_label=dit_label, gpt_label=gpt_label)

    if args.mode in ("all", "summary"):
        print("Speed summary...")
        plot_speed_summary(args.outdir, dit_prefix=dit_prefix, gpt_prefix=gpt_prefix,
                           n_list=ns, dit_label=dit_label, gpt_label=gpt_label)

    if args.mode in ("all", "heatmap"):
        print("Final heatmap...")
        plot_final_heatmap(args.outdir, dit_prefix=dit_prefix, gpt_prefix=gpt_prefix,
                           n_list=ns, dit_label=dit_label, gpt_label=gpt_label)

    if args.mode in ("all", "compact"):
        print("Compact overview...")
        # nano N=4096 + mini N=4096/8192/16384 all in one figure
        compact_rows = [
            dict(dit_prefix="DiT_nano", gpt_prefix="GPT_nano", N=4096,
                 dit_label="DiT-nano", gpt_label="GPT-nano",
                 row_label="nano  N=4,096"),
            dict(dit_prefix="DiT_mini", gpt_prefix="GPT_mini", N=4096,
                 dit_label="DiT-mini", gpt_label="GPT-mini",
                 row_label="mini  N=4,096"),
            dict(dit_prefix="DiT_mini", gpt_prefix="GPT_mini", N=8192,
                 dit_label="DiT-mini", gpt_label="GPT-mini",
                 row_label="mini  N=8,192"),
            dict(dit_prefix="DiT_mini", gpt_prefix="GPT_mini", N=16384,
                 dit_label="DiT-mini", gpt_label="GPT-mini",
                 row_label="mini  N=16,384"),
        ]
        plot_compact(compact_rows, args.outdir, outname="compact_overview.png")

    if args.mode in ("all", "timeheatmap"):
        print("Time heatmap...")
        heatmap_rows = [
            dict(dit_prefix="DiT_nano", gpt_prefix="GPT_nano", N=4096,
                 dit_label="DiT-nano", gpt_label="GPT-nano",
                 row_label="nano  N=4,096"),
            dict(dit_prefix="DiT_mini", gpt_prefix="GPT_mini", N=4096,
                 dit_label="DiT-mini", gpt_label="GPT-mini",
                 row_label="mini  N=4,096"),
            dict(dit_prefix="DiT_mini", gpt_prefix="GPT_mini", N=8192,
                 dit_label="DiT-mini", gpt_label="GPT-mini",
                 row_label="mini  N=8,192"),
            dict(dit_prefix="DiT_mini", gpt_prefix="GPT_mini", N=16384,
                 dit_label="DiT-mini", gpt_label="GPT-mini",
                 row_label="mini  N=16,384"),
        ]
        plot_timeheatmap(heatmap_rows, args.outdir, outname="timeheatmap.png")


if __name__ == "__main__":
    main()
