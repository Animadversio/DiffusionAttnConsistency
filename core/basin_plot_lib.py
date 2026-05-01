"""
Plotting utilities for attractor basin analysis.

Main entry point
----------------
plot_basin_profiles(all_stacks, t_vals, epochs, epoch_labels, epoch_colors,
                    title='', figsize=(13, 10), B=2000, seed=42)

    3×3 grid: columns = (invalid, valid_novel, other_train),
              rows    = (exact bit match, Hamming distance, denoiser L2)
    Shading   = 5–95% bootstrap CI of the mean across training samples.

Helper
------
load_per_sample_stacks(cache_dir, ep, direction, n_samples, sigma, n_points)
    Loads per-line NPZ cache files and returns a dict of (N, T) arrays.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ── Default style constants ────────────────────────────────────────────────────

DIRECTIONS = ['invalid', 'valid_novel', 'other_train']

COL_TITLES = {
    'invalid':     'Toward invalid\n(Hamming-1)',
    'valid_novel': 'Toward valid-novel\n(Hamming-2)',
    'other_train': 'Toward other training\n(nearest Hamming)',
}
ROW_TITLES = [
    'Exact bit match\n(all bits)',
    'Hamming distance\n(# bits changed from $x_a$)',
    r'Denoiser L2  $\|D(x,\sigma)-x_a\|$',
]
METRICS = ['exact_match', 'hamming', 'dist_from_start']


# ── Data loading ───────────────────────────────────────────────────────────────

def load_per_sample_stacks(cache_dir, ep, direction, n_samples=30,
                            sigma=1.0, n_points=150):
    """
    Load per-line NPZ cache files for one (epoch, direction) pair.

    Returns dict with keys 'exact_match', 'hamming', 'dist_from_start',
    each a (n_samples, n_points) float32 array.
    """
    prefix = f"ep{ep:06d}_sig{sigma:.4f}"
    stacks = {}
    for idx in range(n_samples):
        fname = os.path.join(
            cache_dir,
            f"{prefix}_{direction}_xa{idx}_sig{sigma:.4f}_n{n_points}.npz"
        )
        raw = np.load(fname)
        x_start    = raw['x_line'][0]           # (D,)
        D_out      = raw['D_out']               # (T, D)
        sign_start = np.sign(x_start).astype(np.int8)
        sign_D     = np.sign(D_out).astype(np.int8)

        for key, val in [
            ('exact_match',     (sign_D == sign_start[None, :]).all(axis=1).astype(np.float32)),
            ('hamming',         (sign_D != sign_start[None, :]).sum(axis=1).astype(np.float32)),
            ('dist_from_start', np.linalg.norm(D_out - x_start[None, :], axis=1).astype(np.float32)),
        ]:
            stacks.setdefault(key, []).append(val)

    return {k: np.stack(v, axis=0) for k, v in stacks.items()}  # (N, T)


def load_all_stacks(cache_dir, epochs, directions=None, n_samples=30,
                    sigma=1.0, n_points=150):
    """
    Convenience wrapper: load stacks for all (epoch, direction) combos.

    Returns all_stacks[epoch][direction] = dict of (N, T) arrays.
    """
    if directions is None:
        directions = DIRECTIONS
    all_stacks = {}
    for ep in epochs:
        all_stacks[ep] = {}
        for direction in directions:
            all_stacks[ep][direction] = load_per_sample_stacks(
                cache_dir, ep, direction, n_samples=n_samples,
                sigma=sigma, n_points=n_points,
            )
    return all_stacks


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def _bootstrap_ci_mean(stk, B=2000, lo_pct=5, hi_pct=95, rng=None):
    """
    Bootstrap 5–95% CI of the mean.

    stk : (N, T) float array
    Returns lo, hi : (T,) arrays
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = stk.shape[0]
    idxs = rng.integers(0, N, size=(B, N))
    boot_means = stk[idxs, :].mean(axis=1)  # (B, T)
    return np.percentile(boot_means, lo_pct, axis=0), np.percentile(boot_means, hi_pct, axis=0)


# ── Main plot function ─────────────────────────────────────────────────────────

def plot_basin_profiles(
    all_stacks,
    t_vals,
    epochs,
    epoch_labels=None,
    epoch_colors=None,
    directions=None,
    col_titles=None,
    title='',
    figsize=(13, 10),
    B=2000,
    seed=42,
    ax_array=None,
):
    """
    3×3 basin profile plot.

    Columns : invalid / valid_novel / other_train (or custom directions)
    Rows    : exact bit match / Hamming distance / denoiser L2
    Shading : 5–95% bootstrap CI of the mean (B resamples)

    Parameters
    ----------
    all_stacks    : dict  all_stacks[epoch][direction][metric] = (N, T) array
    t_vals        : (T,) float  — shared t-axis
    epochs        : list of int — epochs to plot (one color each)
    epoch_labels  : dict ep→str  (default: "ep {ep}")
    epoch_colors  : dict ep→color  (default: C0, C1, ...)
    directions    : list of 3 direction keys (default: DIRECTIONS)
    col_titles    : dict direction→str  (default: COL_TITLES)
    title         : suptitle string
    figsize       : figure size
    B             : bootstrap resamples
    seed          : random seed for bootstrap
    ax_array      : optional (3,3) axes array — if provided, plot into it

    Returns
    -------
    fig, axes
    """
    if directions is None:
        directions = DIRECTIONS
    if col_titles is None:
        col_titles = COL_TITLES
    if epoch_labels is None:
        epoch_labels = {ep: f"ep {ep}" for ep in epochs}
    if epoch_colors is None:
        epoch_colors = {ep: f"C{i}" for i, ep in enumerate(epochs)}

    rng = np.random.default_rng(seed)

    if ax_array is None:
        fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True)
    else:
        fig = ax_array[0, 0].get_figure()
        axes = ax_array

    for ri, metric in enumerate(METRICS):
        for ci, direction in enumerate(directions):
            ax = axes[ri, ci]
            for ep in epochs:
                stk   = all_stacks[ep][direction][metric]
                mean  = stk.mean(axis=0)
                lo, hi = _bootstrap_ci_mean(stk, B=B, rng=rng)
                color = epoch_colors[ep]
                ax.plot(t_vals, mean, color=color, lw=2.0, label=epoch_labels[ep])
                ax.fill_between(t_vals, lo, hi, color=color, alpha=0.25)

            ax.axvline(0,    color='k',    lw=0.8, ls='--', alpha=0.4)
            ax.axvline(1,    color='gray', lw=0.8, ls=':',  alpha=0.4)
            if metric == 'exact_match':
                ax.set_ylim(-0.05, 1.05)
            if ri == 0:
                ax.set_title(col_titles[direction], fontsize=11, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(ROW_TITLES[ri], fontsize=10)
            if ri == 2:
                ax.set_xlabel('t  (0 = $x_a$,  1 = endpoint)', fontsize=9)

    axes[0, 2].legend(fontsize=8, loc='upper right')
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig, axes
