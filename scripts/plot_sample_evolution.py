"""
Generate sample evolution plots for one experiment.

Produces three figures:
  1. evolution_overview.png       — change rate, bits flipped, valid/mem, confidence, arrival CDF
  2. state_transitions.png        — 4-state stacked area + 6 transition curves (EMA) + late zoom
  3. state_raster_4state.png      — per-sample state raster (4-state, sorted, log-x)

Usage:
  python scripts/plot_sample_evolution.py --exp_name DiT_mini_rowK2_n6_N4096
  python scripts/plot_sample_evolution.py --exp_name DiT_mini_globalK15_n6_N4096 --figdir /tmp/figs
"""

import os, sys, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top']   = False
plt.rcParams['figure.dpi'] = 120

DEFAULT_SAVEROOT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)
DEFAULT_FIGDIR = (
    "/n/home12/binxuwang/Github/DiffusionAttnConsistency/figures/rowK_analysis"
)

# ── State definitions ─────────────────────────────────────────────────────────
# 0 = Invalid + quant ambiguous  (|x|<0.1 for ≥1 bit)  → orange
# 1 = Invalid + clean quant (rule error only)           → red
# 2 = Valid novel                                       → green
# 3 = Valid & Memorized                                 → purple
STATE_INFO = [
    (0, 'Invalid (quant ambiguous)', '#ff7f0e'),
    (1, 'Invalid (rule error)',      '#d62728'),
    (2, 'Valid (novel)',             '#2ca02c'),
    (3, 'Valid & Memorized',        '#9467bd'),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def ema(arr, alpha=0.9):
    s = np.empty_like(arr, dtype=np.float32)
    s[0] = arr[0]
    for i in range(1, len(arr)):
        s[i] = (1 - alpha) * arr[i] + alpha * s[i-1]
    return s


def load_data(exp_name, saveroot):
    path = os.path.join(saveroot, exp_name, 'evolution_analysis', 'evolution_metrics.npz')
    return np.load(path)


def build_state(is_valid, is_mem, has_ambig):
    T, N = is_valid.shape
    state = np.zeros((T, N), dtype=np.int8)
    state[~is_valid &  has_ambig] = 0
    state[~is_valid & ~has_ambig] = 1
    state[ is_valid & ~is_mem]    = 2
    state[ is_valid &  is_mem]    = 3
    return state


def sort_state(state, is_valid, is_mem):
    T, N = state.shape
    first_mem   = np.full(N, np.inf)
    first_valid = np.full(N, np.inf)
    epochs_dummy = np.arange(T, dtype=float)
    for t in range(T):
        first_mem[np.isinf(first_mem)   & is_mem[t]]    = t
        first_valid[np.isinf(first_valid) & is_valid[t]] = t
    final_state = state[-1]
    sort_key = np.lexsort((first_valid, first_mem, -final_state))
    return state[:, sort_key], sort_key, final_state


def pcolormesh_edges(epochs):
    log_ep  = np.log10(epochs + 1)
    mid     = (log_ep[:-1] + log_ep[1:]) / 2
    x_edges = np.concatenate([[log_ep[0] - (mid[0] - log_ep[0])],
                               mid,
                               [log_ep[-1] + (log_ep[-1] - mid[-1])]])
    return 10**x_edges


def savefig(fig, figdir, name, exp_name):
    tag = exp_name.replace('DiT_mini_', '')
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(figdir, f'{name}_{tag}.{ext}'),
                    dpi=300, bbox_inches='tight')


# ── Transition matrix ─────────────────────────────────────────────────────────

def compute_transition_matrix(d):
    """
    Compute per-step 4×4 state transition count and probability matrices.

    States (rows/cols):
        0 = Invalid, quant-ambiguous  (|x| < 0.1 for ≥1 bit)
        1 = Invalid, rule-error       (clean quantization, wrong rule)
        2 = Valid, novel              (rule-valid, not in training set)
        3 = Valid, memorized          (exact match to a training sample)

    Convention:
        T_count[t, i, j]  = # samples in state i at checkpoint t
                             that transition to state j at checkpoint t+1
        T_prob[t, i, j]   = T_count[t, i, j] / (# samples in state i at t)
                           = 0.0 when source population is 0

        Row  = source state (FROM)
        Col  = destination state (TO)
        t    indexes the T-1 *transitions* between T checkpoints

    Returns
    -------
    T_count : np.ndarray, shape (T-1, 4, 4), int32
    T_prob  : np.ndarray, shape (T-1, 4, 4), float32
    epochs  : np.ndarray, shape (T,),  int64   — checkpoint epochs
    ep_mid  : np.ndarray, shape (T-1,), float64 — midpoint epoch of each transition
    """
    is_valid  = d['is_valid']    # (T, N) bool
    is_mem    = d['is_mem']      # (T, N) bool
    has_ambig = d['has_ambiguous']  # (T, N) bool
    epochs    = d['epochs']      # (T,) int64

    state = build_state(is_valid, is_mem, has_ambig)  # (T, N) int8, values 0-3
    T, N  = state.shape

    T_count = np.zeros((T - 1, 4, 4), dtype=np.int32)
    T_prob  = np.zeros((T - 1, 4, 4), dtype=np.float32)

    for i in range(4):
        src_mask = (state[:-1] == i)          # (T-1, N) bool
        src_pop  = src_mask.sum(axis=1)       # (T-1,) int
        for j in range(4):
            cnt = (src_mask & (state[1:] == j)).sum(axis=1)   # (T-1,) int
            T_count[:, i, j] = cnt
            T_prob[:, i, j]  = np.where(src_pop > 0, cnt / src_pop, 0.0)

    ep_mid = (epochs[:-1] + epochs[1:]) / 2.0
    return T_count, T_prob, epochs, ep_mid


# ── Transition matrix plot ────────────────────────────────────────────────────

STATE_LABELS = [
    'Invalid\n(quant-ambig)',
    'Invalid\n(rule-error)',
    'Valid\n(novel)',
    'Valid\n(memorized)',
]
STATE_COLORS_4 = ['#ff7f0e', '#d62728', '#2ca02c', '#9467bd']


def plot_transition_heatmap(mat44, ax=None, normalize=True, title=''):
    """
    Plot a single (4, 4) transition matrix as an annotated heatmap.

    Convention:  mat44[i, j] = FROM state i  →  TO state j
                 (row = source, col = destination)

    Parameters
    ----------
    mat44     : array-like (4, 4) — counts or probabilities
    normalize : row-normalize to get transition probabilities (default True)
    ax        : existing Axes to draw into; creates new figure if None
    title     : axes title string

    Typical usage
    -------------
    T_count, T_prob, epochs, ep_mid = compute_transition_matrix(d)
    plot_transition_heatmap(T_count[t0:t1].sum(axis=0), normalize=True)
    plot_transition_heatmap(T_count[t0:t1].sum(axis=0), normalize=False)
    """
    import seaborn as sns

    short = ['Inv-ambig (0)', 'Inv-rule (1)', 'Valid-novel (2)', 'Memorized (3)']
    mat44 = np.array(mat44, dtype=np.float64)

    if normalize:
        row_sums = mat44.sum(axis=1, keepdims=True)
        mat_show = np.where(row_sums > 0, mat44 / row_sums, 0.0)
        fmt, vmin, vmax = '.2f', 0.0, 1.0
    else:
        mat_show = mat44
        fmt, vmin, vmax = '.0f', 0, mat44.max() or 1

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(mat_show, annot=True, fmt=fmt, cmap='Blues',
                vmin=vmin, vmax=vmax,
                xticklabels=short, yticklabels=short,
                linewidths=0.5, ax=ax)

    ax.set_xlabel('Destination state  (TO →)', fontsize=10)
    ax.set_ylabel('Source state  (FROM ↓)', fontsize=10)

    for tick, col in zip(ax.get_xticklabels(), STATE_COLORS_4):
        tick.set_color(col)
    for tick, col in zip(ax.get_yticklabels(), STATE_COLORS_4):
        tick.set_color(col)

    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_transition_heatmap_both(mat44, title='', vertical=False, figsize=(12, 5)):
    """
    Convenience wrapper: show counts and row-normalized probabilities
    side by side (horizontal, default) or stacked (vertical) in a single figure.

    Parameters
    ----------
    mat44    : array-like (4, 4) — counts or probabilities
    title    : string, overall plot title prefix
    vertical : bool, if True use vertical (2 rows), else horizontal (2 columns)

    Returns the figure.
    """
    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_transition_heatmap(mat44, ax=axes[0], normalize=False, title=f'{title} — counts')
    plot_transition_heatmap(mat44, ax=axes[1], normalize=True,  title=f'{title} — prob (row-norm)')
    fig.tight_layout()
    return fig


def plot_transition_matrix(d, exp_name, figdir=None, save=True,
                           smooth_alpha=0.85, which='prob'):
    """
    4×4 grid of transition curves — one panel per (source, destination) state pair.

    Layout:
        Row i = source state i   (FROM)
        Col j = destination state j  (TO)
        Diagonal (i==j) = self-retention probability (staying in same state)
        Off-diagonal    = actual transition probability/count

    State indices:
        0 = Invalid, quant-ambiguous  (orange)
        1 = Invalid, rule-error       (red)
        2 = Valid, novel              (green)
        3 = Valid, memorized          (purple)

    Parameters
    ----------
    which : 'prob' (default) or 'count'
        Whether to plot conditional transition probabilities or raw counts.
    smooth_alpha : float
        EMA smoothing factor (0 = no smoothing, higher = more smoothing).
    save : bool
        If False, skip savefig/plt.close so the figure renders inline in notebooks.
    """
    T_count, T_prob, epochs, ep_mid = compute_transition_matrix(d)
    mat = T_prob if which == 'prob' else T_count.astype(np.float32)
    ylabel = 'Transition probability' if which == 'prob' else 'Transition count'

    fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"State transition {'probabilities' if which == 'prob' else 'counts'} — {exp_name}\n"
        f"Row = source state (FROM),  Col = destination state (TO)",
        fontsize=11, fontweight='bold'
    )

    for i in range(4):   # source (FROM)
        for j in range(4):   # destination (TO)
            ax = axes[i, j]
            y_raw = mat[:, i, j]
            y_ema = ema(y_raw, alpha=smooth_alpha) if smooth_alpha > 0 else y_raw

            col = STATE_COLORS_4[j]
            lw  = 2.0 if i == j else 1.6
            ls  = '-' if i != j else '--'

            ax.plot(ep_mid + 1, y_raw, color=col, lw=0.4, alpha=0.2)
            ax.plot(ep_mid + 1, y_ema, color=col, lw=lw, ls=ls)
            ax.set_xscale('log')
            ax.grid(alpha=0.2)

            # Shade diagonal differently (self-retention)
            if i == j:
                ax.set_facecolor('#f5f5f5')

            # Row label on left column only
            if j == 0:
                ax.set_ylabel(f'FROM\n{STATE_LABELS[i]}', fontsize=7,
                               color=STATE_COLORS_4[i], labelpad=4)
            # Column label on top row only
            if i == 0:
                ax.set_title(f'TO\n{STATE_LABELS[j]}', fontsize=7,
                              color=STATE_COLORS_4[j])
            # x-label on bottom row only
            if i == 3:
                ax.set_xlabel('Step', fontsize=7)

            if which == 'prob':
                ax.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        savefig(fig, figdir, 'transition_matrix', exp_name)
        plt.close(fig)
        print(f"  Saved: transition_matrix")


# ── Figure 1: Overview ────────────────────────────────────────────────────────

def plot_overview(d, exp_name, figdir=None, save=True):
    epochs      = d['epochs']
    change_rate = d['change_rate']
    n_bits      = d['n_bits_changed']
    is_valid    = d['is_valid']
    is_mem      = d['is_mem']
    has_ambig   = d['has_ambiguous']
    confidence  = d['mean_confidence']
    arrival     = d['arrival_epochs']
    ep_mid      = (epochs[:-1] + epochs[1:]) / 2

    fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
    fig.suptitle(f"Sample evolution overview — {exp_name}", fontsize=11, fontweight='bold')

    # 1. Change rate
    ax = axes[0]
    ax.plot(ep_mid + 1, change_rate, color='#1f77b4', lw=0.5, alpha=0.2)
    ax.plot(ep_mid + 1, ema(change_rate), color='#1f77b4', lw=1.8)
    ax.set_ylabel('Fraction changed\n(≥1 bit)', fontsize=9)
    ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.set_title('Sample change rate per transition', fontsize=9)

    # 2. Mean bits changed (median + IQR)
    ax = axes[1]
    mean_nb = n_bits.mean(axis=1).astype(np.float32)
    ax.plot(ep_mid + 1, mean_nb, color='#ff7f0e', lw=0.5, alpha=0.2)
    ax.plot(ep_mid + 1, ema(mean_nb), color='#ff7f0e', lw=1.8)
    ax.fill_between(ep_mid + 1,
                    np.percentile(n_bits, 25, axis=1),
                    np.percentile(n_bits, 75, axis=1),
                    color='#ff7f0e', alpha=0.15)
    ax.set_ylabel('Mean bits changed\n(25-75 pct band)', fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title('Bits flipped per sample per transition', fontsize=9)

    # 3. Valid / Mem / Ambiguous
    ax = axes[2]
    ax.plot(epochs + 1, is_valid.mean(axis=1),  color='#2ca02c', lw=1.8, label='valid')
    ax.plot(epochs + 1, is_mem.mean(axis=1),    color='#9467bd', lw=1.8, label='memorized')
    ax.plot(epochs + 1, has_ambig.mean(axis=1), color='#ff7f0e', lw=1.2, ls='--',
            label='has ambiguous bit (|x|<0.1)')
    ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.set_ylabel('Fraction of samples', fontsize=9)
    ax.legend(fontsize=8, loc='center left')
    ax.set_title('Validity / Memorization / Ambiguity over training', fontsize=9)

    # 4. State counts stacked
    state    = build_state(is_valid, is_mem, has_ambig)
    counts   = np.stack([(state == s).sum(axis=1) for s, *_ in STATE_INFO], axis=1)
    ax = axes[3]
    ax.stackplot(epochs + 1,
                 counts[:,0], counts[:,1], counts[:,2], counts[:,3],
                 labels=[info[1] for info in STATE_INFO],
                 colors=[info[2] for info in STATE_INFO], alpha=0.85)
    ax.set_ylim(0, is_valid.shape[1])
    ax.set_ylabel('# samples', fontsize=9); ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_title('4-state count: quant-ambiguous vs rule-error vs valid-novel vs memorized', fontsize=9)

    # 5. Confidence
    ax = axes[4]
    ax.plot(epochs + 1, np.median(confidence, axis=1), color='#8c564b', lw=1.8, label='median')
    ax.fill_between(epochs + 1,
                    np.percentile(confidence, 10, axis=1),
                    np.percentile(confidence, 90, axis=1),
                    color='#8c564b', alpha=0.2, label='10-90 pct')
    ax.set_ylabel('Mean |x|\n(confidence)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title('Model confidence (mean |output|) per sample', fontsize=9)

    # 6. Memorization arrival CDF
    ax = axes[5]
    seen = arrival[arrival >= 0]
    not_seen_frac = (arrival == -1).mean()
    all_ep_sorted = np.sort(seen)
    cum = np.arange(1, len(all_ep_sorted) + 1) / len(arrival)
    ax.plot(all_ep_sorted + 1, cum, color='#1f77b4', lw=1.8)
    ax.axhline(1 - not_seen_frac, color='gray', ls='--', lw=1.0,
               label=f'max reachable ({(1-not_seen_frac)*100:.0f}%)')
    ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.set_ylabel('Fraction of train patterns\never seen in samples', fontsize=9)
    ax.set_xlabel('Training step', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(f'Memorization arrival CDF  ({len(seen)}/{len(arrival)} ever seen)', fontsize=9)

    for ax in axes:
        ax.set_xscale('log')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        savefig(fig, figdir, 'evolution_overview', exp_name)
        plt.close(fig)
        print(f"  Saved: evolution_overview")


# ── Figure 2: State transitions ───────────────────────────────────────────────

def plot_transitions(d, exp_name, figdir=None, save=True):
    epochs   = d['epochs']
    is_valid = d['is_valid']
    is_mem   = d['is_mem']
    has_ambig = d['has_ambiguous']
    ep_mid   = (epochs[:-1] + epochs[1:]) / 2

    state  = build_state(is_valid, is_mem, has_ambig)
    T, N   = state.shape
    counts = np.stack([(state == s).sum(axis=1) for s, *_ in STATE_INFO], axis=1)

    # 6 transitions between the 3 "meaningful" states (0+1 merged as "invalid" for readability,
    # but we keep all 4-state transitions)
    trans_spec = {
        (0,2): ('Ambig invalid→Valid novel',   '#aec7e8', '-',  1.6),
        (0,3): ('Ambig invalid→Memorized',      '#c5b0d5', '-',  1.4),
        (1,2): ('Rule-err→Valid novel',         '#2ca02c', '-',  2.0),
        (1,3): ('Rule-err→Memorized',           '#9467bd', '-',  1.6),
        (2,1): ('Valid novel→Rule-err',         '#d62728', '-',  2.0),
        (2,3): ('Valid novel→Memorized',        '#ff7f0e', '-',  2.2),
        (3,2): ('Memorized→Valid novel',        '#8c564b', '--', 1.8),
        (3,1): ('Memorized→Rule-err',           '#e377c2', '--', 1.6),
    }
    transitions = {k: ((state[:-1] == k[0]) & (state[1:] == k[1])).sum(axis=1).astype(np.float32)
                   for k in trans_spec}

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(f"State transitions — {exp_name}", fontsize=11, fontweight='bold')

    # Panel 1: stacked area (4-state)
    ax = axes[0]
    ax.stackplot(epochs + 1,
                 counts[:,0], counts[:,1], counts[:,2], counts[:,3],
                 labels=[info[1] for info in STATE_INFO],
                 colors=[info[2] for info in STATE_INFO], alpha=0.85)
    ax.set_ylim(0, N); ax.set_ylabel('# samples', fontsize=9); ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_title('State counts (stacked)', fontsize=10)

    # Panel 2: all transitions, raw faint + EMA bold
    ax = axes[1]
    for k, (lbl, col, ls, lw) in trans_spec.items():
        y = transitions[k]
        ax.plot(ep_mid + 1, y,      color=col, lw=0.5, ls=ls, alpha=0.12)
        ax.plot(ep_mid + 1, ema(y), color=col, lw=lw,  ls=ls, alpha=1.0, label=lbl)
    ax.set_ylabel('# samples / transition', fontsize=9); ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.set_title('All transitions (EMA smoothed, α=0.9)', fontsize=10)

    # Panel 3: zoom late (>100k)
    ax = axes[2]
    mask = ep_mid >= 1e5
    for k, (lbl, col, ls, lw) in trans_spec.items():
        y = transitions[k]
        ax.plot(ep_mid[mask] + 1, y[mask],       color=col, lw=0.5, ls=ls, alpha=0.12)
        ax.plot(ep_mid[mask] + 1, ema(y)[mask],  color=col, lw=lw,  ls=ls, alpha=1.0, label=lbl)
    ax.set_ylabel('# samples / transition', fontsize=9)
    ax.set_xlabel('Training step', fontsize=10); ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_title('Zoom: late-training transitions (step > 100k, EMA smoothed)', fontsize=10)

    for ax in axes:
        ax.set_xscale('log')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        savefig(fig, figdir, 'state_transitions', exp_name)
        plt.close(fig)
        print(f"  Saved: state_transitions")


# ── Figure 3: Per-sample raster ───────────────────────────────────────────────

def plot_raster(d, exp_name, figdir=None, save=True,
                sort_idxs=None, sort_str=None):
    """
    Plot per-sample 4-state raster (stacked area + raster panels).

    Parameters
    ----------
    d         : dict — output of load_data()
    exp_name  : str
    sort_idxs : (N,) array-like of int or None
                Custom sample ordering (bottom→top). If None, uses the
                default sort: final state ↓, first-memorized epoch ↑,
                first-valid epoch ↑.
    sort_str  : str or None — y-axis label describing the sort. Defaults
                to the standard description when sort_idxs is None.
    """
    epochs    = d['epochs']
    is_valid  = d['is_valid']
    is_mem    = d['is_mem']
    has_ambig = d['has_ambiguous']
    T, N      = is_valid.shape

    state = build_state(is_valid, is_mem, has_ambig)

    if sort_idxs is None:
        state_sorted, sort_key, final_state = sort_state(state, is_valid, is_mem)
        ylabel = 'Sample index\n(sorted: final state ↓, first mem ↑, first valid ↑)'
    else:
        sort_idxs    = np.asarray(sort_idxs)
        state_sorted = state[:, sort_idxs]
        sort_key     = sort_idxs
        final_state  = state[-1]
        ylabel       = f'Sample index\n(sorted: {sort_str})' if sort_str else 'Sample index (custom order)'

    x_lin   = pcolormesh_edges(epochs)
    y_edges = np.arange(N + 1)

    cmap = mcolors.ListedColormap([info[2] for info in STATE_INFO])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    counts = np.stack([(state == s).sum(axis=1) for s, *_ in STATE_INFO], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                              gridspec_kw={'height_ratios': [1.5, 3]})
    fig.suptitle(f"Per-sample state raster — {exp_name}", fontsize=11, fontweight='bold')

    # Panel 1: stacked area
    ax = axes[0]
    ax.stackplot(epochs + 1,
                 counts[:,0], counts[:,1], counts[:,2], counts[:,3],
                 labels=[info[1] for info in STATE_INFO],
                 colors=[info[2] for info in STATE_INFO], alpha=0.85)
    ax.set_ylim(0, N); ax.set_ylabel('# samples', fontsize=9)
    ax.set_xscale('log'); ax.set_xlim(x_lin[0], x_lin[-1])
    ax.legend(fontsize=8, loc='upper left', ncol=2); ax.grid(alpha=0.2)
    ax.set_title('State counts (stacked) — orange=quant ambiguous, red=rule error', fontsize=9)

    # Panel 2: raster
    ax = axes[1]
    ax.pcolormesh(x_lin, y_edges, state_sorted.T.astype(float),
                  cmap=cmap, norm=norm, rasterized=True, zorder=0)
    ax.set_rasterization_zorder(1)
    ax.set_xscale('log')
    ax.set_xlim(x_lin[0], x_lin[-1]); ax.set_ylim(0, N)
    ax.set_xlabel('Training step', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title('Per-sample state trajectory — each row = same noise seed across training', fontsize=9)

    # Dividers and annotations (only meaningful for default sort)
    if sort_idxs is None:
        y_cursor = 0
        for s_id, lbl, col in [(3,'Memorized','#9467bd'), (2,'Valid novel','#2ca02c'),
                                 (1,'Invalid (rule)','#d62728'), (0,'Invalid (ambig)','#ff7f0e')]:
            count = int((final_state[sort_key] == s_id).sum())
            ax.axhline(y_cursor + count, color='white', lw=1.0, ls='--', alpha=0.7)
            if count > 0:
                ax.annotate(f'{lbl}\n({count})',
                            xy=(1.005, (y_cursor + count / 2) / N), xycoords='axes fraction',
                            fontsize=8, color=col, va='center')
            y_cursor += count

    legend_els = [Patch(color=info[2], label=info[1]) for info in STATE_INFO]
    ax.legend(handles=legend_els, fontsize=8, loc='upper left', framealpha=0.7)

    plt.tight_layout()
    if save:
        savefig(fig, figdir, 'state_raster_4state', exp_name)
        plt.close(fig)
        print(f"  Saved: state_raster_4state")
    return fig


def plot_raster_custom_order(d, sort_idxs, sort_str="custom order",
                             figsize=(13, 5), save=False, legend=True,
                             figdir=None, exp_name='', backend='pcolormesh'):
    """
    Plot a per-sample state raster with a caller-supplied sample ordering.

    Unlike plot_raster() — which sorts by final state / first-memorized epoch —
    this function lets you pass any permutation of sample indices so you can
    order rows by an external criterion (e.g. nearest-Hamming distance to
    training set, arrival time, or cluster assignment).

    Parameters
    ----------
    d         : dict  — output of load_data(); must contain 'epochs', 'is_valid',
                        'is_mem', 'has_ambiguous'
    sort_idxs : (N,) array-like of int  — sample indices in desired row order
                (bottom→top); e.g. np.argsort(some_metric)
    sort_str  : str  — label describing the sort order, shown in y-axis
    figsize   : (w, h) tuple
    save      : bool  — if True, save PNG+PDF via savefig() helper
    figdir    : str or None  — output directory (required when save=True)
    exp_name  : str  — experiment name, used in title and save filename
    backend   : 'pcolormesh' or 'imshow'
                'pcolormesh' — log-scale x-axis, variable-width columns per checkpoint gap.
                'imshow'     — linear axis in log10(epoch) space; each checkpoint gets one
                               equal-width pixel column. Visually equivalent to pcolormesh
                               and always produces a single raster image in the PDF
                               (Illustrator-friendly without needing rasterized=True).

    Returns
    -------
    fig : matplotlib Figure
    """
    epochs    = d['epochs']
    is_valid  = d['is_valid']
    is_mem    = d['is_mem']
    has_ambig = d['has_ambiguous']
    T, N      = is_valid.shape

    state        = build_state(is_valid, is_mem, has_ambig)   # (T, N)
    state_sorted = state[:, sort_idxs]                         # (T, N) reordered

    cmap = mcolors.ListedColormap([info[2] for info in STATE_INFO])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    if backend == 'imshow':
        # x-axis is linear in log10(epoch) space so each column gets equal width,
        # which is visually equivalent to pcolormesh on a log-scale x-axis.
        log_epochs = np.log10(np.asarray(epochs, dtype=float) + 1)
        half = (log_epochs[1] - log_epochs[0]) / 2  # half-pixel for extent
        extent = [log_epochs[0] - half, log_epochs[-1] + half, 0, N]
        ax.imshow(state_sorted.T.astype(float), aspect='auto', origin='lower',
                  cmap=cmap, norm=norm, extent=extent, interpolation='nearest',
                  rasterized=True)
        # Annotate x-axis with actual epoch numbers (pick ~8 evenly spaced ticks)
        tick_idxs = np.linspace(0, T - 1, min(T, 8), dtype=int)
        ax.set_xticks(log_epochs[tick_idxs])
        ax.set_xticklabels([f'{epochs[i]:,}' for i in tick_idxs], rotation=30, ha='right')
        ax.set_xlim(extent[0], extent[1])
    else:
        x_lin   = pcolormesh_edges(epochs)
        y_edges = np.arange(N + 1)
        mesh = ax.pcolormesh(x_lin, y_edges, state_sorted.T.astype(float),
                      cmap=cmap, norm=norm, rasterized=True, 
                      antialiased=False,zorder=0, )
                    #   shading="nearest",   # or "auto", but nearest is often safer for raster-like data
                    #     edgecolors="none",
                    #     linewidth=0,
        mesh.set_rasterized(True)
        ax.set_rasterization_zorder(1)
        ax.set_xscale('log')
        ax.set_xlim(x_lin[0], x_lin[-1])

    ax.set_ylim(0, N)
    ax.set_xlabel('Training step', fontsize=10)
    ax.set_ylabel(f'Sample index (sorted: {sort_str})', fontsize=9)
    title = f'Per-sample state raster — {exp_name}' if exp_name else 'Per-sample state raster'
    ax.set_title(title, fontsize=9)

    if legend:
        legend_els = [Patch(color=info[2], label=info[1]) for info in STATE_INFO]
        ax.legend(handles=legend_els, fontsize=8, loc='upper left', framealpha=0.7)

    plt.tight_layout()
    if save:
        savefig(fig, figdir, 'state_raster_custom_order', exp_name)
        plt.close(fig)
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp_name', required=True)
    p.add_argument('--saveroot', default=DEFAULT_SAVEROOT)
    p.add_argument('--figdir',   default=DEFAULT_FIGDIR)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    print(f"\nPlotting: {args.exp_name}")
    d = load_data(args.exp_name, args.saveroot)
    plot_overview(d, args.exp_name, args.figdir)
    plot_transitions(d, args.exp_name, args.figdir)
    plot_raster(d, args.exp_name, args.figdir)
    print("Done.")
