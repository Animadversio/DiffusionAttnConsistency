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

def make_transition_mask(is_valid, is_mem, src, dst, has_ambiguous=None):
    """
    Boolean mask (T-1, N) for samples transitioning from state `src` to `dst`
    between checkpoint t and t+1.

    States: 0=ambig-invalid, 1=rule-invalid, 2=valid-novel, 3=memorized
    For convenience src/dst can also be string shorthands:
      'v'  = valid novel     (state 2)
      'm'  = memorized       (state 3)
      'i'  = any invalid     (state 0 or 1)
      'ia' = ambig-invalid   (state 0, requires has_ambiguous)
      'ir' = rule-invalid    (state 1, requires has_ambiguous)

    Parameters
    ----------
    has_ambiguous : (T, N) bool or None
        If None, 'ia' / 'ir' / state 0 / state 1 all fall back to ~is_valid.
    """
    # pull slices once for clarity
    v,  v1  = is_valid[:-1],       is_valid[1:]
    m,  m1  = is_mem[:-1],         is_mem[1:]
    if has_ambiguous is not None:
        a,  a1 = has_ambiguous[:-1], has_ambiguous[1:]
    else:
        a = a1 = None

    def _at(s, vt, mt, at):
        """State mask at a single time slice."""
        if   s in ('v',  2):  return  vt & ~mt
        elif s in ('m',  3):  return  mt
        elif s in ('i',):     return ~vt
        elif s in ('ia', 0):
            return (~vt &  at) if at is not None else ~vt
        elif s in ('ir', 1):
            return (~vt & ~at) if at is not None else ~vt
        raise ValueError(f"Unknown state {s!r}")

    return _at(src, v, m, a) & _at(dst, v1, m1, a1)


# ── Hamming at valid→memorized transitions ────────────────────────────────────

def ham_at_transition_by_window(ham, epochs_sub, is_valid, is_mem,
                                n_windows=6, log_time=True,
                                src='v', dst='m', has_ambiguous=None):
    """
    Split training into time windows and, within each window, collect the
    Hamming distance to the nearest training sample at the moment a sample
    transitions from state `src` → state `dst`.

    Default: valid-novel → memorized (src='v', dst='m').

    Parameters
    ----------
    ham           : (T, N) int16   nearest Hamming to training set
    epochs_sub    : (T,)   int64   epoch at each checkpoint (must align with ham)
    is_valid      : (T, N) bool
    is_mem        : (T, N) bool
    n_windows     : int            number of time windows
    log_time      : bool           split windows in log-epoch space
    src, dst      : str or int     source/destination state (see make_transition_mask)
    has_ambiguous : (T, N) bool or None  required for 'ia'/'ir'/state-0/1 transitions

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

    trans = make_transition_mask(is_valid, is_mem, src, dst, has_ambiguous)  # (T-1, N)
    ham_before = ham[:-1]                                                     # (T-1, N)

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


def ham_at_transition_single_window(ham, epochs_sub, is_valid, is_mem,
                                    ep_lo=None, ep_hi=None,
                                    src='v', dst='m', has_ambiguous=None):
    """
    Collect Hamming distances at transitions from state `src` → `dst`
    within a single epoch interval [ep_lo, ep_hi).

    Parameters
    ----------
    ham           : (T, N) int16   nearest Hamming to training set
    epochs_sub    : (T,)   int64   epoch at each checkpoint
    is_valid      : (T, N) bool
    is_mem        : (T, N) bool
    ep_lo         : float or None  start epoch (None = beginning)
    ep_hi         : float or None  end epoch   (None = end)
    src, dst      : str or int     source/destination state (see make_transition_mask)
    has_ambiguous : (T, N) bool or None  required for 'ia'/'ir'/state-0/1 transitions

    Returns
    -------
    result : dict with keys:
        'ham'           : (K,) int16   Hamming values in window
        'epochs'        : (K,) int64   epoch of each transition
        'ep_lo'         : float or None
        'ep_hi'         : float or None
        'label'         : str
        'n_transitions' : int
        'median'        : float
        'mean'          : float
        'pct_zero'      : float        fraction with Hamming == 0
    trans_mask : (T-1, N) bool  full transition mask (before time windowing)
    """
    T, N = is_valid.shape
    assert ham.shape == (T, N), \
        f"ham shape {ham.shape} must match is_valid shape {(T, N)}"

    trans      = make_transition_mask(is_valid, is_mem, src, dst, has_ambiguous)  # (T-1, N)
    ham_before = ham[:-1]                                                          # (T-1, N)
    ham_after  = ham[1:]                                                           # (T-1, N)

    time_sel = np.ones(T - 1, dtype=bool)
    if ep_lo is not None:
        time_sel &= epochs_sub[:-1] >= ep_lo
    if ep_hi is not None:
        time_sel &= epochs_sub[:-1] <  ep_hi

    trans_win = trans & time_sel[:, None]
    ham_win   = ham_before[trans_win]
    ham_win_after = ham_after[trans_win]
    ep_win    = epochs_sub[np.where(trans_win)[0]]

    lo_str = f"{ep_lo:.0f}" if ep_lo is not None else "start"
    hi_str = f"{ep_hi:.0f}" if ep_hi is not None else "end"
    label  = f"ep [{lo_str}, {hi_str})"

    result = dict(
        ham=ham_win,              # Hamming just BEFORE the transition (at t)
        ham_after=ham_win_after,  # Hamming just AFTER  the transition (at t+1)
        epochs=ep_win,
        ep_lo=ep_lo, ep_hi=ep_hi, label=label,
        n_transitions=len(ham_win),
        median=float(np.median(ham_win))      if len(ham_win) else float('nan'),
        mean=float(ham_win.mean())            if len(ham_win) else float('nan'),
        pct_zero=float((ham_win == 0).mean()) if len(ham_win) else float('nan'),
    )
    return result, trans


def bits_changed_at_transition(n_bits_changed, epochs_sub,
                               is_valid, is_mem,
                               ep_lo=None, ep_hi=None,
                               src='v', dst='m', has_ambiguous=None,
                               bit_change_rate=None):
    """
    For each transition event of type src→dst within [ep_lo, ep_hi), collect:
      - number of bits flipped between t and t+1          (from evo n_bits_changed)
      - optionally per-position flip rates at those steps  (from evo bit_change_rate)

    All arrays must come from evolution_metrics.npz and share the same T.

    Parameters
    ----------
    n_bits_changed : (T-1, N) int16   bits flipped per sample between t and t+1
    epochs_sub     : (T,)    int64    checkpoint epochs (length T; transitions span T-1)
    is_valid       : (T, N)  bool
    is_mem         : (T, N)  bool
    ep_lo, ep_hi   : float or None    epoch window (None = open-ended)
    src, dst       : str or int       transition states (see make_transition_mask)
    has_ambiguous  : (T, N) bool or None
    bit_change_rate: (T-1, D) float32 or None
                     per-position flip frequency; if provided, returns mean
                     per-position rate over selected transitions

    Returns
    -------
    result : dict with keys:
        'n_bits'         : (K,) int16   bits flipped for each transition event
        'epochs'         : (K,) int64   epoch (t) of each event
        'n_transitions'  : int
        'median'         : float
        'mean'           : float
        'pct_one_bit'    : float        fraction of single-bit-flip transitions
        'bit_pos_rate'   : (D,) float32 or None
                           mean per-position flip rate over selected transitions
                           (only if bit_change_rate is provided)
    trans_mask : (T-1, N) bool   full transition mask (before time windowing)
    """
    T, N = is_valid.shape
    assert n_bits_changed.shape == (T - 1, N), \
        f"n_bits_changed shape {n_bits_changed.shape} must be (T-1, N) = {(T-1, N)}"

    trans    = make_transition_mask(is_valid, is_mem, src, dst, has_ambiguous)  # (T-1, N)
    time_sel = np.ones(T - 1, dtype=bool)
    if ep_lo is not None:
        time_sel &= epochs_sub[:-1] >= ep_lo
    if ep_hi is not None:
        time_sel &= epochs_sub[:-1] <  ep_hi

    trans_win  = trans & time_sel[:, None]                   # (T-1, N)
    n_bits_win = n_bits_changed[trans_win]                   # (K,) int16
    ep_win     = epochs_sub[np.where(trans_win)[0]]          # (K,) int64

    # per-position flip rate averaged over selected (t, i) events
    bit_pos = None
    if bit_change_rate is not None:
        # For each selected time step t, weight by number of transitioning samples
        t_idx = np.where(trans_win.any(axis=1))[0]          # unique t's with events
        if len(t_idx):
            rates = bit_change_rate[t_idx]                   # (t_uniq, D)
            # weight by event count at each t
            counts = trans_win[t_idx].sum(axis=1)            # (t_uniq,)
            bit_pos = (rates * counts[:, None]).sum(axis=0) / counts.sum()
            bit_pos = bit_pos.astype(np.float32)

    result = dict(
        n_bits=n_bits_win,
        epochs=ep_win,
        n_transitions=len(n_bits_win),
        median=float(np.median(n_bits_win))          if len(n_bits_win) else float('nan'),
        mean=float(n_bits_win.mean())                if len(n_bits_win) else float('nan'),
        pct_one_bit=float((n_bits_win == 1).mean())  if len(n_bits_win) else float('nan'),
        bit_pos_rate=bit_pos,
    )
    return result, trans


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

def plot_ham_before_after(ham_before, ham_after, n_bits,
                          ax=None, title='',
                          max_points=400, jitter=0.18,
                          cmap='Spectral_r', rng=None):
    """
    Paired jitter plot showing Hamming distance BEFORE and AFTER a transition,
    with lines connecting each pair colored by the number of bits that flipped.

    Parameters
    ----------
    ham_before : (K,) int16   Hamming to training set just before transition
    ham_after  : (K,) int16   Hamming to training set just after  transition
    n_bits     : (K,) int16   bits flipped during the transition
    ax         : matplotlib Axes or None
    title      : str
    max_points : int           subsample to this many pairs (avoids crowding)
    jitter     : float         horizontal jitter width
    cmap       : str           colormap for n_bits (default: RdPu — low=pale, high=saturated)
    rng        : np.random.Generator or None  for reproducible subsampling

    Returns
    -------
    fig, ax

    Stats annotation
    ----------------
    Wilcoxon signed-rank test (paired, two-sided) on ham_before vs ham_after.
    Annotated on the figure as W=..., p=...
    """
    from scipy.stats import wilcoxon
    import matplotlib.colors as mcolors

    ham_before = np.asarray(ham_before, dtype=np.float32)
    ham_after  = np.asarray(ham_after,  dtype=np.float32)
    n_bits     = np.asarray(n_bits,     dtype=np.float32)
    K = len(ham_before)
    assert len(ham_after) == K and len(n_bits) == K

    # subsample
    if K > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(K, size=max_points, replace=False)
        ham_before = ham_before[idx]
        ham_after  = ham_after[idx]
        n_bits     = n_bits[idx]
        n_shown    = max_points
    else:
        n_shown = K

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    # color by n_bits
    vmin, vmax = 1, max(int(n_bits.max()), 2)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_  = plt.get_cmap(cmap)
    colors = cmap_(norm(n_bits))   # (n_shown, 4)

    rng2 = np.random.default_rng(42)
    jb = rng2.uniform(-jitter, jitter, n_shown)
    ja = rng2.uniform(-jitter, jitter, n_shown)

    x_before = np.zeros(n_shown) + jb
    x_after  = np.ones(n_shown)  + ja

    # draw connecting lines first (behind points)
    for xb, xa, yb, ya, c in zip(x_before, x_after, ham_before, ham_after, colors):
        ax.plot([xb, xa], [yb, ya], color=c, alpha=0.25, lw=0.7, zorder=1)

    # scatter points
    ax.scatter(x_before, ham_before, c=colors, s=12, zorder=2,
               edgecolors='none', alpha=0.6)
    ax.scatter(x_after,  ham_after,  c=colors, s=12, zorder=2,
               edgecolors='none', alpha=0.6, marker='D')

    # mean ± SEM markers (shifted slightly inward to avoid collision)
    mu_b, sem_b = ham_before.mean(), ham_before.std() / np.sqrt(n_shown)
    mu_a, sem_a = ham_after.mean(),  ham_after.std()  / np.sqrt(n_shown)
    x_mb, x_ma = -0.12, 1.12   # shifted outward from the jitter cloud
    ax.errorbar(x_mb, mu_b, yerr=sem_b, fmt='o', color='steelblue',
                ms=8, capsize=5, lw=2.5, zorder=4,
                label=f'mean before={mu_b:.2f} ± {sem_b:.2f}')
    ax.errorbar(x_ma, mu_a, yerr=sem_a, fmt='D', color='tomato',
                ms=8, capsize=5, lw=2.5, zorder=4,
                label=f'mean after={mu_a:.2f} ± {sem_a:.2f}')

    # stats test + compact annotation
    diff = ham_before - ham_after
    if (diff != 0).any():
        stat, pval = wilcoxon(ham_before, ham_after, alternative='two-sided')
        pstr = f'p={pval:.2e}' if pval >= 1e-300 else 'p<1e-300'
    else:
        pstr = 'p=n/a (all diffs=0)'

    annot = (f'n={K}  {pstr}\n'
             f'Δham {diff.mean():+.2f}±{diff.std():.2f}\n'
             f'bits {n_bits.mean():.1f}±{n_bits.std():.1f}')
    ax.text(0.97, 0.97, annot,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8,
            bbox=dict(fc='white', ec='gray', alpha=0.85, boxstyle='round,pad=0.3'))

    # print full stats to console
    print(f'[{title}]  n={K}  {pstr}')
    print(f'  Hamming before: mean={ham_before.mean():.2f} med={np.median(ham_before):.1f} std={ham_before.std():.2f}')
    print(f'  Hamming after:  mean={ham_after.mean():.2f} med={np.median(ham_after):.1f} std={ham_after.std():.2f}')
    print(f'  Δ (before-after): mean={diff.mean():+.2f} med={np.median(diff):+.2f} std={diff.std():.2f}')
    print(f'  Bits flipped: mean={n_bits.mean():.2f} med={np.median(n_bits):.1f} std={n_bits.std():.2f} '
          f'| 1-bit={100*(n_bits==1).mean():.1f}% 2-bit={100*(n_bits==2).mean():.1f}% >4={100*(n_bits>4).mean():.1f}%')

    # colorbar for n_bits
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('bits flipped', fontsize=8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before\ntransition', 'After\ntransition'], fontsize=10)
    ax.set_ylabel('Nearest Hamming distance to training set', fontsize=10)
    ax.set_title(title or f'Hamming before/after transition  (showing {n_shown}/{K})')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def plot_ham_before_after_from_window(ham, epochs_sub, is_valid, is_mem,
                                      n_bits_changed,
                                      ep_lo=None, ep_hi=None,
                                      src='v', dst='m', has_ambiguous=None,
                                      ax=None, title='', max_points=400, **kwargs):
    """
    High-level wrapper: selects transition events via ham_at_transition_single_window
    and bits_changed_at_transition, then calls plot_ham_before_after.

    Parameters
    ----------
    ham, epochs_sub, is_valid, is_mem : from dist_to_train.npz / evolution_metrics.npz
    n_bits_changed : (T-1, N) int16   from evo['n_bits_changed']
    ep_lo, ep_hi   : epoch window
    src, dst       : transition states
    has_ambiguous  : (T, N) bool or None
    **kwargs       : passed to plot_ham_before_after (jitter, cmap, rng, ...)
    """
    ham_res, _ = ham_at_transition_single_window(
        ham, epochs_sub, is_valid, is_mem,
        ep_lo=ep_lo, ep_hi=ep_hi, src=src, dst=dst,
        has_ambiguous=has_ambiguous,
    )
    bits_res, _ = bits_changed_at_transition(
        n_bits_changed, epochs_sub, is_valid, is_mem,
        ep_lo=ep_lo, ep_hi=ep_hi, src=src, dst=dst,
        has_ambiguous=has_ambiguous,
    )
    if title == '':
        lo_str = f"{ep_lo:.0f}" if ep_lo is not None else "start"
        hi_str = f"{ep_hi:.0f}" if ep_hi is not None else "end"
        title  = f"{src}→{dst}  ep [{lo_str}, {hi_str})"

    return plot_ham_before_after(
        ham_res['ham'], ham_res['ham_after'], bits_res['n_bits'],
        ax=ax, title=title, max_points=max_points, **kwargs,
    )


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


def ham_by_state_window(ham, epochs_sub, is_valid, is_mem,
                        sample_type='all', ep_lo=None, ep_hi=None,
                        has_ambiguous=None):
    """
    Extract Hamming distances for samples of a given state type within
    an epoch window, flattened across all (t, i) entries in that window.

    Parameters
    ----------
    ham          : (T, N) int16   nearest Hamming to training set
    epochs_sub   : (T,)   int64   checkpoint epochs
    is_valid     : (T, N) bool
    is_mem       : (T, N) bool
    sample_type  : str
        'all'  — every sample at every checkpoint in the window
        'v'    — valid novel   (is_valid & ~is_mem)
        'm'    — memorized     (is_mem)
        'i'    — any invalid   (~is_valid)
        'ir'   — rule-invalid  (~is_valid & ~has_ambiguous)  requires has_ambiguous
        'ia'   — ambig-invalid (~is_valid &  has_ambiguous)  requires has_ambiguous
    ep_lo, ep_hi : float or None  epoch window (None = open-ended)
    has_ambiguous: (T, N) bool or None

    Returns
    -------
    ham_vals : (K,) int16   Hamming values for all matching (t, i) entries
    """
    T, N = is_valid.shape

    # time mask
    time_sel = np.ones(T, dtype=bool)
    if ep_lo is not None:
        time_sel &= epochs_sub >= ep_lo
    if ep_hi is not None:
        time_sel &= epochs_sub <  ep_hi

    # state mask (T, N)
    if   sample_type == 'all':
        state_mask = np.ones((T, N), dtype=bool)
    elif sample_type == 'v':
        state_mask =  is_valid & ~is_mem
    elif sample_type == 'm':
        state_mask =  is_mem
    elif sample_type == 'i':
        state_mask = ~is_valid
    elif sample_type in ('ir', 'ia'):
        if has_ambiguous is None:
            raise ValueError(f"has_ambiguous required for sample_type={sample_type!r}")
        if sample_type == 'ir':
            state_mask = ~is_valid & ~has_ambiguous
        else:
            state_mask = ~is_valid &  has_ambiguous
    else:
        raise ValueError(f"Unknown sample_type {sample_type!r}. "
                         f"Choose from: 'all','v','m','i','ir','ia'")

    combined = state_mask & time_sel[:, None]   # (T, N)
    return ham[combined]


def plot_ham_compare(ham_a, ham_b, label_a='group A', label_b='group B',
                     ax=None, title='', colors=('#4C72B0', '#DD8452')):
    """
    Compare two Hamming distance distributions with a dodged histogram
    and Mann-Whitney U test annotation.

    Parameters
    ----------
    ham_a, ham_b : array-like  int   Hamming values for each group
    label_a, label_b : str           legend labels
    ax           : matplotlib Axes or None
    title        : str               figure title (test stats appended as subtitle)
    colors       : tuple of 2 str   bar colors for group A and B

    Returns
    -------
    fig, ax
    """
    import pandas as pd
    import seaborn as sns
    from scipy.stats import mannwhitneyu

    ham_a = np.asarray(ham_a).ravel()
    ham_b = np.asarray(ham_b).ravel()

    stat, pval = mannwhitneyu(ham_a, ham_b, alternative='two-sided')
    pstr = f'p={pval:.2e}' if pval >= 1e-300 else 'p<1e-300'

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    df = pd.concat([
        pd.DataFrame({'ham': ham_a, 'type': label_a}),
        pd.DataFrame({'ham': ham_b, 'type': label_b}),
    ], ignore_index=True)

    sns.histplot(data=df, x='ham', hue='type',
                 multiple='dodge', stat='probability',
                 common_norm=False, discrete=True, shrink=0.85,
                 palette={label_a: colors[0], label_b: colors[1]},
                 legend=True, ax=ax)

    stats_line = (f'Mann-Whitney U={stat:.2e}  {pstr}  |  '
                  f'{label_a}: μ={ham_a.mean():.2f}  '
                  f'{label_b}: μ={ham_b.mean():.2f}')
    full_title = f'{title}\n{stats_line}' if title else stats_line
    ax.set_title(full_title, fontsize=9)
    ax.set_xlabel('Hamming distance to nearest training sample', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig, ax
