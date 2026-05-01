"""
Attractor basin measurement for trained diffusion models on discrete binary data.

For a given training sample x_a, we measure the denoiser D(x,σ) along three
1D lines through the high-dimensional space:
  1. Toward the nearest Hamming-1 invalid sample  (parity broken, one bit flip)
  2. Toward the nearest Hamming-2 valid-novel sample (parity preserved, two-bit flip)
  3. Toward the nearest other training sample      (by Hamming distance)

Line parameterization: x(t) = x_a + t * (x_target - x_a)
  t=0 → x_a (training sample / basin center)
  t=1 → x_target (endpoint)
  t ∈ [-0.5, 2.0] by default (slightly behind x_a, well past target)

Three basin metrics computed at each t:
  exact_match   : sign(D(x(t))) == sign(x_a) for ALL bits  → cleanest binary basin boundary
  bit_agreement : fraction of bits where sign agrees        → graded version
  dist_from_start: ||D(x(t)) - x_a||                       → L2 metric

Basin half-width = t* where exact_match first becomes False (going from t=0 outward).
"""

import os
import numpy as np
import torch
from typing import Optional


# ── Neighbor construction ─────────────────────────────────────────────────────

def get_nearest_invalid_neighbor(x_a: np.ndarray, rule_params: dict) -> np.ndarray:
    """
    Return a Hamming-1 invalid neighbor of x_a by flipping one bit that breaks parity.

    For parity rules: flips bit 0 of group 0 (always breaks parity of that group).
    For row-K rules: flips the first bit in the first row.

    x_a : (D,) float {-1, +1}
    Returns x_inv : (D,) float {-1, +1}
    """
    x_inv = x_a.copy()
    if rule_params.get('rule') == 'parity':
        # flip bit 0 of group 0 — breaks that group's parity product
        x_inv[0] = -x_inv[0]
    else:
        # generic: flip first bit
        x_inv[0] = -x_inv[0]
    return x_inv


def get_nearest_valid_novel_neighbor(
    x_a: np.ndarray,
    rule_params: dict,
    train_codes: Optional[set] = None,
) -> np.ndarray:
    """
    Return a Hamming-2 valid-novel neighbor of x_a by flipping two bits within the
    same parity group (product unchanged → parity preserved).

    Tries flipping bits (0,1), (0,2), (1,2) within group 0 until a non-training
    sample is found.  Falls back to group 1 if needed.

    x_a        : (D,) float {-1, +1}
    train_codes: set of int — integer codes of training samples (optional check)
    Returns x_novel : (D,) float {-1, +1}
    """
    group_size = rule_params.get('group_size', 3)

    def bit_code(x):
        bits = (x > 0).astype(np.int64)
        n = len(bits)
        return int(sum(bits[i] << i for i in range(n)))

    # Try pairs within each group
    n_groups = len(x_a) // group_size
    for g in range(n_groups):
        base = g * group_size
        for i in range(group_size):
            for j in range(i + 1, group_size):
                x_cand = x_a.copy()
                x_cand[base + i] = -x_cand[base + i]
                x_cand[base + j] = -x_cand[base + j]
                if train_codes is None or bit_code(x_cand) not in train_codes:
                    return x_cand

    # Fallback: return first candidate regardless
    x_cand = x_a.copy()
    x_cand[0] = -x_cand[0]
    x_cand[1] = -x_cand[1]
    return x_cand


def get_nearest_other_train(
    x_a: np.ndarray,
    x_train: np.ndarray,
    x_a_idx: Optional[int] = None,
) -> tuple:
    """
    Find the nearest other training sample by Hamming distance.

    x_a      : (D,) float {-1, +1}
    x_train  : (N, D) float {-1, +1}
    x_a_idx  : index of x_a in x_train (excluded from search)

    Returns (x_other, hamming_dist, other_idx)
    """
    q_a = (x_a > 0).astype(np.int8)
    q_train = (x_train > 0).astype(np.int8)
    hamming = (q_a[None, :] != q_train).sum(axis=1)  # (N,)
    if x_a_idx is not None:
        hamming[x_a_idx] = 999
    else:
        # exclude exact matches
        hamming[hamming == 0] = 999
    nearest_idx = int(hamming.argmin())
    return x_train[nearest_idx], int(hamming[nearest_idx]), nearest_idx


def load_rule_params(exp_dir: str) -> dict:
    """Load rule parameters from args.json in exp_dir."""
    import json
    args_path = os.path.join(exp_dir, 'args.json')
    if not os.path.exists(args_path):
        return {'rule': 'unknown'}
    with open(args_path) as f:
        args = json.load(f)
    if 'group_size' in args:
        return {
            'rule': 'parity',
            'group_size': int(args['group_size']),
            'parity_val': int(args.get('parity_val', 1)),
        }
    return {'rule': args.get('rule', 'unknown')}


# ── Line profile measurement ──────────────────────────────────────────────────

def measure_line_profile(
    model,
    sigma: float,
    x_start: np.ndarray,
    x_end: np.ndarray,
    t_vals: Optional[np.ndarray] = None,
    n_points: int = 150,
    t_range: tuple = (-0.5, 2.0),
    device: str = 'cpu',
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> dict:
    """
    Evaluate the denoiser along the line x(t) = x_start + t*(x_end - x_start).

    Parameters
    ----------
    model    : EDM model (has .forward or __call__)
    sigma    : noise level
    x_start  : (D,) float — origin of line (training sample)
    x_end    : (D,) float — endpoint (invalid / valid-novel / other-train)
    t_vals   : if provided, used directly; else linspace(t_range, n_points)
    n_points : number of t values
    t_range  : (t_min, t_max)
    device   : 'cpu' or 'cuda'

    Returns dict
    ------
    t_vals        : (n_points,)
    x_line        : (n_points, D)  — input points
    D_out         : (n_points, D)  — denoiser output
    score         : (n_points, D)  — (D-x)/σ²
    dist_from_start : (n_points,) — ||D_out - x_start||
    bit_agreement : (n_points,)   — fraction of bits sign(D)==sign(x_start)
    exact_match   : (n_points,)   bool — all bits agree with x_start
    proj_pull     : (n_points,)   — score projected onto direction toward x_start
                                    positive = score points back toward x_start
    direction_l2  : float         — ||x_end - x_start||
    t_basin_lo    : float         — largest t < 0 where exact_match still True
    t_basin_hi    : float         — smallest t > 0 where exact_match first False
    basin_width_t : float         — t_basin_hi - t_basin_lo (in t units)
    basin_width_l2: float         — basin_width_t * direction_l2 (in L2 units)
    """
    if t_vals is None:
        t_vals = np.linspace(t_range[0], t_range[1], n_points, dtype=np.float32)

    direction = (x_end - x_start).astype(np.float32)
    direction_l2 = float(np.linalg.norm(direction))

    # Check cache
    if cache_dir and cache_key:
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.join(cache_dir, f"{cache_key}_sig{sigma:.4f}_n{len(t_vals)}.npz")
        if os.path.exists(fname):
            c = np.load(fname, allow_pickle=True)
            return _compute_derived(dict(c), x_start, direction, direction_l2, t_vals)

    # Build line points
    x_line = x_start[None, :] + t_vals[:, None] * direction[None, :]  # (n_points, D)
    x_line_f32 = x_line.astype(np.float32)

    # Model forward pass (batch all points together)
    from core.vector_field_lib import eval_score
    score_np, D_np = eval_score(model, x_line_f32, sigma, device)

    result = dict(
        t_vals=t_vals,
        x_line=x_line_f32,
        D_out=D_np.astype(np.float32),
        score=score_np.astype(np.float32),
        direction_l2=np.float32(direction_l2),
    )

    if cache_dir and cache_key:
        np.savez_compressed(fname, **result)

    return _compute_derived(result, x_start, direction, direction_l2, t_vals)


def _compute_derived(result, x_start, direction, direction_l2, t_vals):
    D_out  = result['D_out']    # (n_points, D)
    score  = result['score']    # (n_points, D)
    D = D_out.shape[1]

    sign_start = np.sign(x_start).astype(np.int8)  # (D,)
    sign_D     = np.sign(D_out).astype(np.int8)     # (n_points, D)

    # Exact bit match: all bits agree
    exact_match   = (sign_D == sign_start[None, :]).all(axis=1)          # (n_points,) bool
    # Bit agreement fraction
    bit_agreement = (sign_D == sign_start[None, :]).mean(axis=1)         # (n_points,)
    # L2 distance of denoiser output from x_start
    dist_from_start = np.linalg.norm(D_out - x_start[None, :], axis=1)  # (n_points,)
    # Score projected onto -direction (pointing back toward x_start)
    if direction_l2 > 0:
        v_back = -direction / direction_l2
        proj_pull = (score * v_back[None, :]).sum(axis=1)                # (n_points,)
    else:
        proj_pull = np.zeros(len(t_vals), dtype=np.float32)

    # Basin boundaries (in t units)
    t_basin_lo, t_basin_hi = _find_basin_bounds(t_vals, exact_match)

    out = dict(result)
    out.update(dict(
        t_vals=t_vals,
        direction_l2=np.float32(direction_l2),
        exact_match=exact_match,
        bit_agreement=bit_agreement.astype(np.float32),
        dist_from_start=dist_from_start.astype(np.float32),
        proj_pull=proj_pull.astype(np.float32),
        t_basin_lo=np.float32(t_basin_lo),
        t_basin_hi=np.float32(t_basin_hi),
        basin_width_t=np.float32(t_basin_hi - t_basin_lo),
        basin_width_l2=np.float32((t_basin_hi - t_basin_lo) * direction_l2),
    ))
    return out


def _find_basin_bounds(t_vals, exact_match):
    """
    Find t_lo (last t < 0 with match True going backward from 0)
    and t_hi (first t > 0 with match False going forward from 0).
    If match never fails in positive direction, t_hi = t_vals[-1].
    """
    t0_idx = int(np.argmin(np.abs(t_vals)))  # index closest to t=0

    # Forward: find first False after t=0
    t_hi = float(t_vals[-1])
    for i in range(t0_idx, len(t_vals)):
        if not exact_match[i]:
            t_hi = float(t_vals[i])
            break

    # Backward: find last True before t=0 (going negative)
    t_lo = float(t_vals[0])
    for i in range(t0_idx, -1, -1):
        if not exact_match[i]:
            t_lo = float(t_vals[i])
            break

    return t_lo, t_hi


# ── Per-sample basin measurement ──────────────────────────────────────────────

def measure_basin_three_directions(
    model,
    sigma: float,
    x_a: np.ndarray,
    x_a_idx: int,
    x_train: np.ndarray,
    train_codes: set,
    rule_params: dict,
    n_points: int = 150,
    t_range: tuple = (-0.5, 2.0),
    device: str = 'cpu',
    cache_dir: Optional[str] = None,
    cache_prefix: Optional[str] = None,
) -> dict:
    """
    Measure basin profiles along all three directions for one training sample.

    Returns dict with keys 'invalid', 'valid_novel', 'other_train', each
    containing the full profile dict from measure_line_profile plus:
      'x_end'        : the endpoint used
      'hamming_dist' : Hamming distance from x_a to x_end
    """
    x_invalid   = get_nearest_invalid_neighbor(x_a, rule_params)
    x_valid_nov = get_nearest_valid_novel_neighbor(x_a, rule_params, train_codes)
    x_other, h_other, other_idx = get_nearest_other_train(x_a, x_train, x_a_idx)

    results = {}
    for name, x_end, h_dist in [
        ('invalid',     x_invalid,   1),
        ('valid_novel', x_valid_nov, 2),
        ('other_train', x_other,     h_other),
    ]:
        ck = f"{cache_prefix}_{name}_xa{x_a_idx}" if cache_prefix else None
        prof = measure_line_profile(
            model, sigma, x_a, x_end,
            n_points=n_points, t_range=t_range,
            device=device, cache_dir=cache_dir, cache_key=ck,
        )
        prof['x_end']        = x_end
        prof['hamming_dist'] = h_dist
        prof['other_idx']    = other_idx if name == 'other_train' else None
        results[name] = prof

    return results


# ── Batch measurement across training samples ─────────────────────────────────

def measure_basin_batch(
    model,
    sigma: float,
    x_train: np.ndarray,
    train_codes: set,
    rule_params: dict,
    n_samples: int = 50,
    n_points: int = 150,
    t_range: tuple = (-0.5, 2.0),
    device: str = 'cpu',
    cache_dir: Optional[str] = None,
    cache_prefix: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Measure basin profiles for n_samples training samples, aggregate statistics.

    Returns dict with:
      per_sample   : list of per-sample dicts (from measure_basin_three_directions)
      summary      : dict with mean/std of basin_width_t, basin_width_l2 etc.
                     per direction
      t_vals       : shared t-axis
      mean_profiles: mean of exact_match, bit_agreement, dist_from_start per direction
    """
    N = min(n_samples, len(x_train))
    indices = np.arange(N)  # first N samples
    per_sample = []

    for i, idx in enumerate(indices):
        if verbose:
            print(f"  Sample {i+1}/{N} (train idx={idx})")
        x_a = x_train[idx]
        prof = measure_basin_three_directions(
            model, sigma, x_a, int(idx), x_train, train_codes, rule_params,
            n_points=n_points, t_range=t_range, device=device,
            cache_dir=cache_dir, cache_prefix=cache_prefix,
        )
        per_sample.append(prof)

    # Aggregate
    t_vals = per_sample[0]['invalid']['t_vals']
    summary = {}
    mean_profiles = {}

    for direction in ('invalid', 'valid_novel', 'other_train'):
        widths_t  = [s[direction]['basin_width_t']  for s in per_sample]
        widths_l2 = [s[direction]['basin_width_l2'] for s in per_sample]
        hi_t      = [s[direction]['t_basin_hi']     for s in per_sample]
        lo_t      = [s[direction]['t_basin_lo']     for s in per_sample]
        h_dists   = [s[direction]['hamming_dist']   for s in per_sample]

        summary[direction] = dict(
            basin_width_t_mean  = float(np.mean(widths_t)),
            basin_width_t_std   = float(np.std(widths_t)),
            basin_width_l2_mean = float(np.mean(widths_l2)),
            basin_width_l2_std  = float(np.std(widths_l2)),
            t_basin_hi_mean     = float(np.mean(hi_t)),
            t_basin_hi_std      = float(np.std(hi_t)),
            t_basin_lo_mean     = float(np.mean(lo_t)),
            t_basin_lo_std      = float(np.std(lo_t)),
            hamming_dist_mean   = float(np.mean(h_dists)),
        )

        for metric in ('exact_match', 'bit_agreement', 'dist_from_start', 'proj_pull'):
            stacked = np.stack([s[direction][metric] for s in per_sample], axis=0)
            mean_profiles.setdefault(direction, {})[metric] = stacked.mean(axis=0)
            mean_profiles[direction][f'{metric}_std'] = stacked.std(axis=0)

    return dict(
        per_sample=per_sample,
        summary=summary,
        mean_profiles=mean_profiles,
        t_vals=t_vals,
        n_samples=N,
        sigma=sigma,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def load_per_sample_stacks_from_cache(
    cache_dir: str,
    epoch: int,
    direction: str,
    n_samples: int = 30,
    sigma: float = 1.0,
    n_points: int = 150,
) -> dict:
    """
    Load per-sample line-cache NPZs and return stacked arrays.

    Returns dict with keys 'exact_match', 'hamming', 'dist_from_start',
    each of shape (n_samples, n_points).
    """
    prefix = f"ep{epoch:06d}_sig{sigma:.4f}"
    stacks = {}
    for idx in range(n_samples):
        fname = os.path.join(cache_dir,
            f"{prefix}_{direction}_xa{idx}_sig{sigma:.4f}_n{n_points}.npz")
        raw = np.load(fname)
        x_start = raw['x_line'][0]
        D_out   = raw['D_out']
        sign_start = np.sign(x_start).astype(np.int8)
        sign_D     = np.sign(D_out).astype(np.int8)
        for key, val in [
            ('exact_match',
             (sign_D == sign_start[None, :]).all(axis=1).astype(np.float32)),
            ('hamming',
             (sign_D != sign_start[None, :]).sum(axis=1).astype(np.float32)),
            ('dist_from_start',
             np.linalg.norm(D_out - x_start[None, :], axis=1).astype(np.float32)),
        ]:
            stacks.setdefault(key, []).append(val)
    return {k: np.stack(v, axis=0) for k, v in stacks.items()}


def basin_plot_profiles(
    cache_dir: str,
    epochs: list,
    epoch_labels: Optional[dict] = None,
    epoch_colors: Optional[dict] = None,
    sigma: float = 1.0,
    n_samples: int = 30,
    n_points: int = 150,
    n_bootstrap: int = 2000,
    ci: tuple = (5, 95),
    directions: tuple = ('invalid', 'valid_novel', 'other_train'),
    col_titles: Optional[list] = None,
    title: Optional[str] = None,
    figsize: tuple = (13, 10),
) -> 'plt.Figure':
    """
    Plot 3×3 basin profile grid (directions as columns, metrics as rows).

    Rows:
      0 — Exact bit match (all bits agree with x_a)
      1 — Hamming distance (# bits changed from x_a)
      2 — Denoiser L2 distance from x_a

    Columns: invalid, valid_novel, other_train directions.

    Shading: bootstrap CI of the mean (ci=(lo_pct, hi_pct)).

    Parameters
    ----------
    cache_dir    : directory containing per-line NPZ cache files
    epochs       : list of int — checkpoint epochs to overlay
    epoch_labels : dict {epoch: str} — line labels; defaults to "ep {epoch}"
    epoch_colors : dict {epoch: color} — line colors; defaults to matplotlib tab10
    sigma        : noise level used during caching
    n_samples    : number of training samples (must match cache)
    n_points     : number of t-values (must match cache)
    n_bootstrap  : bootstrap resamples for CI
    ci           : (lo_pct, hi_pct) percentiles for CI band

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    if epoch_labels is None:
        epoch_labels = {ep: f"ep {ep:,}" for ep in epochs}
    if epoch_colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(epochs)))
        epoch_colors = {ep: colors[i] for i, ep in enumerate(epochs)}
    if col_titles is None:
        col_titles = [
            'Toward invalid\n(Hamming-1)',
            'Toward valid-novel\n(Hamming-2)',
            'Toward other training\n(nearest Hamming)',
        ]

    row_titles = [
        'Exact bit match\n(all bits)',
        'Hamming distance\n(# bits changed from $x_a$)',
        r'Denoiser L2  $\|D(x,\sigma)-x_a\|$',
    ]
    metrics = ['exact_match', 'hamming', 'dist_from_start']

    # Load all stacks
    all_stacks = {}
    for ep in epochs:
        all_stacks[ep] = {}
        for direction in directions:
            all_stacks[ep][direction] = load_per_sample_stacks_from_cache(
                cache_dir, ep, direction, n_samples, sigma, n_points)

    # t_vals from first file
    first_ep  = epochs[0]
    first_dir = directions[0]
    prefix = f"ep{first_ep:06d}_sig{sigma:.4f}"
    t_vals = np.load(os.path.join(cache_dir,
        f"{prefix}_{first_dir}_xa0_sig{sigma:.4f}_n{n_points}.npz"))['t_vals']

    fig, axes = plt.subplots(3, len(directions), figsize=figsize, sharex=True)

    for ri, metric in enumerate(metrics):
        for ci_idx, direction in enumerate(directions):
            ax = axes[ri, ci_idx]
            for ep in epochs:
                stk = all_stacks[ep][direction][metric]  # (N, T)
                N = stk.shape[0]
                mean = stk.mean(axis=0)
                idxs = rng.integers(0, N, size=(n_bootstrap, N))
                boot = stk[idxs, :].mean(axis=1)
                lo_band = np.percentile(boot, ci[0], axis=0)
                hi_band = np.percentile(boot, ci[1], axis=0)
                color = epoch_colors[ep]
                ax.plot(t_vals, mean, color=color, lw=2.0, label=epoch_labels[ep])
                ax.fill_between(t_vals, lo_band, hi_band, color=color, alpha=0.25)

            ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.axvline(1, color='gray', lw=0.8, ls=':', alpha=0.4)
            if metric == 'exact_match':
                ax.set_ylim(-0.05, 1.05)
            if ri == 0:
                ax.set_title(col_titles[ci_idx], fontsize=11, fontweight='bold')
            if ci_idx == 0:
                ax.set_ylabel(row_titles[ri], fontsize=10)
            if ri == 2:
                ax.set_xlabel('t  (0=x_a, 1=endpoint)', fontsize=9)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    axes[0, -1].legend(fontsize=8, loc='upper right')

    if title is None:
        title = (f"Attractor basin profiles  σ={sigma}  N={n_samples} samples  "
                 f"(shading: {ci[0]}–{ci[1]}% CI of mean, bootstrap)")
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    return fig
