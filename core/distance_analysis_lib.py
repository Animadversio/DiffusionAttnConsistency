"""
core/distance_analysis_lib.py

Utilities for analyzing Hamming / L2 distance of generated samples to the
training set across training checkpoints, and relating distance to
memorization / validity transitions.

Typical usage in a notebook
---------------------------
    import numpy as np
    from core.distance_analysis_lib import (
        load_dist_data, load_evo_data,
        build_state,
        ham_at_transition_by_window,
        plot_ham_v2m_windows,
        compute_transition_matrix,
        plot_transition_heatmap_both,
    )

    d   = load_dist_data(exp_name, saveroot)
    evo = load_evo_data(exp_name, saveroot)
    state = build_state(evo['is_valid'], evo['is_mem'], evo['has_ambiguous'])
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── State constants ───────────────────────────────────────────────────────────
#  0  invalid + quant-ambiguous  (orange)
#  1  invalid + rule-error       (red)
#  2  valid novel                (green)
#  3  memorized                  (purple)
STATE_LABELS = ['Invalid (ambig)', 'Invalid (rule)', 'Valid novel', 'Memorized']
STATE_COLORS = ['#ff7f0e', '#d62728', '#2ca02c', '#9467bd']

DEFAULT_SAVEROOT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_dist_data(exp_name, saveroot=DEFAULT_SAVEROOT):
    """
    Load dist_to_train.npz for an experiment.

    Returns dict with keys:
      epochs_sub      (T,)      subsampled epoch numbers
      nearest_hamming (T, N)    min Hamming distance to training set
      nearest_l2      (T, N)    min L2 distance (continuous)
      nearest_idx_ham (T, N)    index of nearest training sample (Hamming)
      traj_state      (N,)      0=stuck / 1=flicker / 2=novel / 3=mem
      first_mem_ep    (N,)      epoch of first memorization (-1=never)
      first_valid_ep  (N,)      epoch of first validity (-1=never)
      flicker_score_tail      (N,)  std(is_valid) in last 1/5 of training
      flicker_score_post_mem  (N,)  std(is_valid) after mem onset
      mem_onset_epoch         scalar  epoch when global mem ratio >= 10%
    """
    path = os.path.join(saveroot, exp_name, 'evolution_analysis', 'dist_to_train.npz')
    npz  = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def load_evo_data(exp_name, saveroot=DEFAULT_SAVEROOT):
    """
    Load evolution_metrics.npz for an experiment.

    Returns dict with keys:
      epochs          (T,)      all checkpoint epoch numbers
      is_valid        (T, N)    rule validity per sample per checkpoint
      is_mem          (T, N)    exact memorization per sample per checkpoint
      has_ambiguous   (T, N)    any bit with |x| < 0.1
      ambig_count     (T, N)    number of ambiguous bits
      mean_confidence (T, N)    mean |x| per sample
      change_rate     (T-1,)    fraction of samples that flipped >= 1 bit
      arrival_epochs  (N_train,) epoch each training pattern first appeared (-1=never)
    """
    path = os.path.join(saveroot, exp_name, 'evolution_analysis', 'evolution_metrics.npz')
    npz  = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


# ── State classification ──────────────────────────────────────────────────────

def build_state(is_valid, is_mem, has_ambiguous):
    """
    Build 4-state classification array from boolean masks.

    Convention:
      0  invalid + quant-ambiguous  (~valid &  ambiguous)
      1  invalid + rule-error       (~valid & ~ambiguous)
      2  valid novel                ( valid & ~mem)
      3  memorized                  ( valid &  mem)

    Parameters
    ----------
    is_valid      : (T, N) bool
    is_mem        : (T, N) bool
    has_ambiguous : (T, N) bool

    Returns
    -------
    state : (T, N) int8
    """
    state = np.zeros(is_valid.shape, dtype=np.int8)
    state[~is_valid &  has_ambiguous] = 0
    state[~is_valid & ~has_ambiguous] = 1
    state[ is_valid & ~is_mem]        = 2
    state[ is_valid &  is_mem]        = 3
    return state


# ── Transition masks ──────────────────────────────────────────────────────────

def make_transition_mask(is_valid, is_mem, src, dst):
    """
    Boolean mask (T-1, N) for samples transitioning from state `src` to `dst`
    between checkpoint t and t+1.

    States: 0=ambig-invalid, 1=rule-invalid, 2=valid-novel, 3=memorized
    For convenience src/dst can also be string shorthands:
      'v'  = valid novel (state 2)
      'm'  = memorized   (state 3)
      'i'  = any invalid (state 0 or 1)
      'ia' = ambig-invalid (state 0)
      'ir' = rule-invalid  (state 1)
    """
    def _mask(s):
        if   s == 'v':  return  is_valid[:-1] & ~is_mem[:-1]
        elif s == 'm':  return  is_mem[:-1]
        elif s == 'i':  return ~is_valid[:-1]
        elif s == 'ia': return ~is_valid[:-1]   # has_ambig not available here
        elif s == 'ir': return ~is_valid[:-1]
        elif s == 0:    return ~is_valid[:-1]
        elif s == 2:    return  is_valid[:-1] & ~is_mem[:-1]
        elif s == 3:    return  is_mem[:-1]
        raise ValueError(f"Unknown state {s!r}")

    def _mask_next(s):
        if   s == 'v':  return  is_valid[1:] & ~is_mem[1:]
        elif s == 'm':  return  is_mem[1:]
        elif s == 'i':  return ~is_valid[1:]
        elif s == 'ia': return ~is_valid[1:]
        elif s == 'ir': return ~is_valid[1:]
        elif s == 0:    return ~is_valid[1:]
        elif s == 2:    return  is_valid[1:] & ~is_mem[1:]
        elif s == 3:    return  is_mem[1:]
        raise ValueError(f"Unknown state {s!r}")

    return _mask(src) & _mask_next(dst)


# ── Hamming at valid→memorized transitions ────────────────────────────────────

def ham_at_transition_by_window(ham, epochs_sub, is_valid, is_mem,
                                n_windows=6, log_time=True,
                                src='v', dst='m'):
    """
    Split training into time windows and, within each window, collect the
    Hamming distance to the nearest training sample at the moment a sample
    transitions from state `src` → state `dst`.

    Default: valid-novel → memorized (src='v', dst='m').

    Parameters
    ----------
    ham        : (T, N) int16   nearest Hamming to training set
    epochs_sub : (T,)   int64   epoch at each checkpoint (must align with ham)
    is_valid   : (T, N) bool
    is_mem     : (T, N) bool
    n_windows  : int            number of time windows
    log_time   : bool           split windows in log-epoch space
    src, dst   : str or int     source/destination state (see make_transition_mask)

    Returns
    -------
    windows : list of dicts, one per time window, each with:
        'ham'           : (K,) int16   Hamming values in this window
        'epochs'        : (K,) int64   epoch of each transition
        'ep_lo'         : float        window start epoch
        'ep_hi'         : float        window end epoch
        'label'         : str
        'n_transitions' : int
    trans_mask : (T-1, N) bool  full transition mask (for reference)
    """
    T, N = is_valid.shape
    assert ham.shape == (T, N), \
        f"ham shape {ham.shape} must match is_valid shape {(T, N)}"

    trans = make_transition_mask(is_valid, is_mem, src, dst)  # (T-1, N)
    ham_before = ham[:-1]                                      # (T-1, N)

    # time window edges
    if log_time:
        pos_epochs = epochs_sub[epochs_sub > 0]
        ep_min = np.log10(pos_epochs.min())
        ep_max = np.log10(epochs_sub.max())
        edges  = np.logspace(ep_min, ep_max, n_windows + 1)
    else:
        edges  = np.linspace(epochs_sub.min(), epochs_sub.max(), n_windows + 1)

    windows = []
    for i in range(n_windows):
        lo, hi   = edges[i], edges[i + 1]
        time_sel = (epochs_sub[:-1] >= lo) & (epochs_sub[:-1] < hi)  # (T-1,)
        trans_win = trans & time_sel[:, None]                          # (T-1, N)

        ham_win = ham_before[trans_win]
        ep_win  = epochs_sub[np.where(trans_win)[0]]

        if log_time:
            label = f"ep ≈ [{lo:.0f}, {hi:.0f})"
        else:
            label = f"ep [{lo:.0f}, {hi:.0f})"

        windows.append(dict(
            ham=ham_win, epochs=ep_win,
            ep_lo=lo, ep_hi=hi, label=label,
            n_transitions=len(ham_win),
        ))

    return windows, trans


def ham_at_transition_summary(windows):
    """Print a quick text summary of Hamming values per window."""
    print(f"{'Window':<35} {'N':>6}  {'median':>7}  {'mean':>7}  {'%at0':>6}")
    for w in windows:
        h = w['ham']
        if len(h) == 0:
            print(f"  {w['label']:<33}      0")
            continue
        pct0 = 100 * (h == 0).mean()
        print(f"  {w['label']:<33} {len(h):>6}  {np.median(h):>7.1f}  "
              f"{h.mean():>7.2f}  {pct0:>5.1f}%")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ham_v2m_windows(windows, bins=np.arange(0, 19),
                         ax=None, title='', palette='plasma'):
    """
    Histogram of Hamming-at-transition for each time window.

    Parameters
    ----------
    windows : list of dicts from ham_at_transition_by_window()
    bins    : array-like   histogram bin edges (default 0..18 integer bins)
    ax      : matplotlib Axes or None
    title   : str
    palette : str   matplotlib colormap name for window colors

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    colors = getattr(cm, palette)(np.linspace(0.1, 0.9, len(windows)))
    for w, c in zip(windows, colors):
        if len(w['ham']) == 0:
            continue
        ax.hist(w['ham'], bins=bins, density=True, alpha=0.55,
                color=c, label=f"{w['label']}  (n={w['n_transitions']})",
                histtype='stepfilled', linewidth=0.8, edgecolor=c)

    ax.set_xlabel(
        'Hamming distance to nearest training sample\n'
        '(measured at checkpoint just before transition)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title or 'Hamming at valid→memorized transition, by time window')
    ax.legend(fontsize=7, framealpha=0.8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def plot_ham_over_time_by_state(ham, epochs_sub, traj_state,
                                ax=None, title='', smooth=10):
    """
    Mean nearest-Hamming over training, split by final trajectory state.

    Parameters
    ----------
    ham         : (T, N) int16
    epochs_sub  : (T,) int64
    traj_state  : (N,) uint8   0=stuck/1=flicker/2=novel/3=mem
    smooth      : int          rolling window for smoothing (in checkpoints)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    x = epochs_sub + 1   # avoid log(0)
    for s, (lbl, col) in enumerate(zip(STATE_LABELS, STATE_COLORS)):
        mask = traj_state == s
        if mask.sum() == 0:
            continue
        h = ham[:, mask].astype(np.float32)
        mu  = h.mean(axis=1)
        std = h.std(axis=1)
        if smooth > 1:
            from numpy.lib.stride_tricks import sliding_window_view
            pad = smooth // 2
            mu  = np.convolve(mu,  np.ones(smooth) / smooth, mode='same')
            std = np.convolve(std, np.ones(smooth) / smooth, mode='same')
        ax.plot(x, mu, color=col, lw=1.8, label=f'{lbl} (n={mask.sum()})')
        ax.fill_between(x, mu - std, mu + std, color=col, alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel('Training step', fontsize=10)
    ax.set_ylabel('Nearest Hamming distance', fontsize=10)
    ax.set_title(title or 'Hamming to training set by final trajectory state')
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


# ── Transition matrix ─────────────────────────────────────────────────────────

def compute_transition_matrix(is_valid, is_mem, has_ambiguous=None):
    """
    Compute the (T-1, 4, 4) transition count and probability matrices.

    Convention
    ----------
    T_count[t, i, j] = number of samples in state i at checkpoint t
                        that are in state j at checkpoint t+1

    T_prob[t, i, j]  = T_count[t, i, j] / (# samples in state i at t)
                        = 0 when source population is 0

    States (rows = source, columns = destination):
      0  invalid + quant-ambiguous
      1  invalid + rule-error
      2  valid novel
      3  memorized

    Parameters
    ----------
    is_valid      : (T, N) bool
    is_mem        : (T, N) bool
    has_ambiguous : (T, N) bool or None  (if None, state 0 == state 1)

    Returns
    -------
    T_count : (T-1, 4, 4) int32
    T_prob  : (T-1, 4, 4) float32
    """
    if has_ambiguous is None:
        has_ambiguous = np.zeros_like(is_valid)

    state = build_state(is_valid, is_mem, has_ambiguous)  # (T, N) int8
    T, N  = state.shape

    T_count = np.zeros((T - 1, 4, 4), dtype=np.int32)
    T_prob  = np.zeros((T - 1, 4, 4), dtype=np.float32)

    for i in range(4):
        src_mask = (state[:-1] == i)               # (T-1, N)
        src_pop  = src_mask.sum(axis=1)            # (T-1,)
        for j in range(4):
            cnt = (src_mask & (state[1:] == j)).sum(axis=1)   # (T-1,)
            T_count[:, i, j] = cnt
            T_prob[:, i, j]  = np.where(src_pop > 0, cnt / src_pop, 0.0)

    return T_count, T_prob


def plot_transition_heatmap_both(mat_4x4, normalize=True,
                                 title='', figsize=(10, 4), fmt_count='.0f'):
    """
    Show a (4, 4) transition matrix as two side-by-side heatmaps:
      left  = raw counts (unnormalized)
      right = row-normalized probabilities

    Parameters
    ----------
    mat_4x4   : (4, 4) array  — counts (can be averaged over time)
    normalize : bool           — if False, only show counts on left
    title     : str
    figsize   : tuple
    fmt_count : str            — fmt string for count annotations

    Returns
    -------
    fig, axes
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required for plot_transition_heatmap_both")

    counts = np.array(mat_4x4)
    with np.errstate(invalid='ignore'):
        probs  = counts / counts.sum(axis=1, keepdims=True)
        probs  = np.nan_to_num(probs)

    ncols = 2 if normalize else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    kw = dict(xticklabels=STATE_LABELS, yticklabels=STATE_LABELS,
              linewidths=0.5, linecolor='white')

    sns.heatmap(counts, ax=axes[0], annot=True, fmt=fmt_count,
                cmap='Blues', **kw)
    axes[0].set_title('Transition counts', fontsize=11)
    axes[0].set_xlabel('State at t+1', fontsize=9)
    axes[0].set_ylabel('State at t', fontsize=9)

    if normalize:
        sns.heatmap(probs, ax=axes[1], annot=True, fmt='.2f',
                    cmap='Oranges', vmin=0, vmax=1, **kw)
        axes[1].set_title('Transition probabilities (row-normalized)', fontsize=11)
        axes[1].set_xlabel('State at t+1', fontsize=9)
        axes[1].set_ylabel('State at t', fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig, axes


def aggregate_transition_matrix(T_count, epochs, ep_lo=None, ep_hi=None):
    """
    Sum T_count over a time window [ep_lo, ep_hi) and return a (4,4) matrix.

    Parameters
    ----------
    T_count : (T-1, 4, 4) int32
    epochs  : (T,)  int64   checkpoint epochs (length T, transitions are between t and t+1)
    ep_lo   : float or None  start epoch (None = beginning)
    ep_hi   : float or None  end epoch   (None = end)

    Returns
    -------
    agg : (4, 4) int32
    """
    trans_epochs = epochs[:-1]   # epoch at start of each transition interval
    mask = np.ones(len(trans_epochs), dtype=bool)
    if ep_lo is not None:
        mask &= trans_epochs >= ep_lo
    if ep_hi is not None:
        mask &= trans_epochs <  ep_hi
    return T_count[mask].sum(axis=0)
