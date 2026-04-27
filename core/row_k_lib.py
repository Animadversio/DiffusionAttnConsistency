"""
row_k_lib.py

Dataset generation and evaluation for per-row / global-K binary grid rules.

An n×n binary grid has values in {-1, +1}.  Three constraint families:

  row_k         : every row has exactly K entries equal to +1  (K fixed)
  row_variable_k: each row independently draws its K from K_list  (rows can differ)
  global_k      : a single K is drawn from K_list per sample,
                  then ALL rows of that sample use that same K

Encoding convention (matching exact_k_lib.py / parity_lib.py):
  active   = +1
  inactive = -1
  #active in a row = (row.sum() + n) / 2

Valid set sizes:
  row_k               : C(n, K)^n
  row_variable_k      : (Σ_{k ∈ K_list} C(n,k))^n
  global_k            : Σ_{k ∈ K_list} C(n,k)^n

Example (n=6):
  K=2          : C(6,2)^6 = 15^6  = 11,390,625
  K=3          : C(6,3)^6 = 20^6  = 64,000,000
  K_list={1,5} row_variable_k : (6+6)^6 = 12^6 = 2,985,984
  K_list={1,5} global_k       : 6^6 + 6^6       =    93,312
  K_list={3,4} row_variable_k : (20+15)^6        = 1,838,265,625
  K_list={3,4} global_k       : 20^6 + 15^6      =   75,390,625
"""

import numpy as np
from math import comb, prod
from itertools import product as iproduct


# ---------------------------------------------------------------------------
# Low-level row sampling helpers
# ---------------------------------------------------------------------------

def _sample_row_exact_k(n, K, N=1):
    """
    Sample N rows of length n, each with exactly K entries equal to +1.

    Returns
    -------
    np.ndarray of shape (N, n), dtype int8
    """
    x = np.full((N, n), -1, dtype=np.int8)
    idx = np.argsort(np.random.rand(N, n), axis=1)[:, :K]
    np.put_along_axis(x, idx, 1, axis=1)
    return x  # (N, n)


# ---------------------------------------------------------------------------
# Dataset sampling — row_k  (fixed K per row)
# ---------------------------------------------------------------------------

def sample_row_k_batch(N, n, K):
    """
    Sample N grids of shape (n, n) where every row has exactly K active entries.

    Parameters
    ----------
    N : int  -- number of samples
    n : int  -- grid size (n rows × n cols)
    K : int  -- number of +1 entries per row  (0 ≤ K ≤ n)

    Returns
    -------
    np.ndarray of shape (N, n, n), dtype int8
    """
    # Each grid: n independent rows of length n with exactly K ones
    rows = _sample_row_exact_k(n, K, N=N * n)        # (N*n, n)
    return rows.reshape(N, n, n)


def sample_row_k_dataset(N, n, K):
    """
    Sample N *unique* grids where every row has exactly K active entries.

    Raises ValueError if N > valid_set_size_row_k(n, K).
    """
    vs = valid_set_size_row_k(n, K)
    if N > vs:
        raise ValueError(f"Cannot sample {N} unique grids: only {vs} valid samples exist.")
    return _sample_ensuring_uniqueness(N, lambda M: sample_row_k_batch(M, n, K))


# ---------------------------------------------------------------------------
# Dataset sampling — row_variable_k  (per-row K drawn independently from K_list)
# ---------------------------------------------------------------------------

def sample_row_variable_k_batch(N, n, K_list):
    """
    Sample N grids of shape (n, n).
    Each row independently draws its K from K_list (uniform), then places K active entries.

    Parameters
    ----------
    N      : int
    n      : int
    K_list : sequence of int  -- allowed K values per row

    Returns
    -------
    np.ndarray of shape (N, n, n), dtype int8
    """
    K_arr = np.array(K_list, dtype=int)
    # Draw K for each (sample, row) pair
    k_choices = K_arr[np.random.randint(len(K_arr), size=(N, n))]  # (N, n)
    grids = np.full((N, n, n), -1, dtype=np.int8)
    for sample_i in range(N):
        for row_j in range(n):
            k = int(k_choices[sample_i, row_j])
            cols = np.random.choice(n, size=k, replace=False)
            grids[sample_i, row_j, cols] = 1
    return grids


def sample_row_variable_k_dataset(N, n, K_list):
    """
    Sample N *unique* grids where each row draws its K from K_list independently.
    """
    vs = valid_set_size_row_variable_k(n, K_list)
    if N > vs:
        raise ValueError(f"Cannot sample {N} unique grids: only {vs} valid samples exist.")
    return _sample_ensuring_uniqueness(N, lambda M: sample_row_variable_k_batch(M, n, K_list))


# ---------------------------------------------------------------------------
# Dataset sampling — global_k  (one K drawn per sample, all rows use same K)
# ---------------------------------------------------------------------------

def sample_global_k_batch(N, n, K_list):
    """
    Sample N grids of shape (n, n).
    A single K is drawn from K_list (uniform) per sample; all n rows of that
    sample then have exactly K active entries.

    Parameters
    ----------
    N      : int
    n      : int
    K_list : sequence of int

    Returns
    -------
    grids   : np.ndarray of shape (N, n, n), dtype int8
    k_labels: np.ndarray of shape (N,), dtype int  -- which K was used per sample
    """
    K_arr = np.array(K_list, dtype=int)
    k_labels = K_arr[np.random.randint(len(K_arr), size=N)]  # (N,)
    grids = np.full((N, n, n), -1, dtype=np.int8)
    for k_val in K_arr:
        mask = k_labels == k_val
        if mask.sum() == 0:
            continue
        n_k = int(mask.sum())
        grids[mask] = sample_row_k_batch(n_k, n, k_val)
    return grids, k_labels


def sample_global_k_dataset(N, n, K_list):
    """
    Sample N *unique* grids where all rows share a globally drawn K ∈ K_list.
    Returns (grids, k_labels).
    """
    vs = valid_set_size_global_k(n, K_list)
    if N > vs:
        raise ValueError(f"Cannot sample {N} unique grids: only {vs} valid samples exist.")

    collected, seen, labels = [], set(), []
    while len(collected) < N:
        batch = max(N - len(collected), N // 2)
        grids, ks = sample_global_k_batch(batch, n, K_list)
        for g, k in zip(grids, ks):
            t = tuple(g.flatten().tolist())
            if t not in seen:
                seen.add(t)
                collected.append(g)
                labels.append(int(k))
                if len(collected) == N:
                    break
    return np.array(collected), np.array(labels)


# ---------------------------------------------------------------------------
# Uniqueness helper
# ---------------------------------------------------------------------------

def _sample_ensuring_uniqueness(N, sample_func):
    collected, seen = [], set()
    while len(collected) < N:
        batch_size = max(N - len(collected), N // 2)
        grids = sample_func(batch_size)
        for g in grids:
            t = tuple(g.flatten().tolist())
            if t not in seen:
                seen.add(t)
                collected.append(g)
                if len(collected) == N:
                    break
    return np.array(collected)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def per_row_counts(x_flat, n):
    """
    Number of active (+1) entries per row for each sample.

    Parameters
    ----------
    x_flat : np.ndarray of shape (N, n²), values in {-1, +1}
    n      : int

    Returns
    -------
    np.ndarray of shape (N, n), dtype int  -- #active per row
    """
    x = x_flat.reshape(-1, n, n)
    return ((x + 1) // 2).sum(axis=2).astype(int)   # (N, n)


def check_row_k_batch(x_flat, n, K):
    """
    True if every row of each sample has exactly K active entries.

    Parameters
    ----------
    x_flat : np.ndarray (N, n²), values in {-1, +1}
    n, K   : int

    Returns
    -------
    np.ndarray of shape (N,), dtype bool
    """
    counts = per_row_counts(x_flat, n)   # (N, n)
    return (counts == K).all(axis=1)


def check_row_variable_k_batch(x_flat, n, K_list):
    """
    True if every row of each sample has a count ∈ K_list.

    Parameters
    ----------
    x_flat  : np.ndarray (N, n²), values in {-1, +1}
    n       : int
    K_list  : sequence of int

    Returns
    -------
    np.ndarray of shape (N,), dtype bool
    """
    counts = per_row_counts(x_flat, n)   # (N, n)
    K_set = set(K_list)
    row_ok = np.array([[c in K_set for c in row] for row in counts])  # (N, n)
    return row_ok.all(axis=1)


def check_global_k_batch(x_flat, n, K_list):
    """
    True if all rows of a sample share the SAME K value AND that K ∈ K_list.

    Parameters
    ----------
    x_flat  : np.ndarray (N, n²), values in {-1, +1}
    n       : int
    K_list  : sequence of int

    Returns
    -------
    valid  : np.ndarray (N,) bool
    k_used : np.ndarray (N,) int  -- inferred K per sample (-1 if invalid)
    """
    counts = per_row_counts(x_flat, n)          # (N, n)
    K_set = set(K_list)
    valid = np.zeros(len(x_flat), dtype=bool)
    k_used = np.full(len(x_flat), -1, dtype=int)
    for i, row_counts in enumerate(counts):
        unique_counts = set(row_counts.tolist())
        if len(unique_counts) == 1:
            k = unique_counts.pop()
            if k in K_set:
                valid[i] = True
                k_used[i] = k
    return valid, k_used


# ---------------------------------------------------------------------------
# Snap continuous outputs → {-1, +1}
# ---------------------------------------------------------------------------

def snap_to_binary(x_cont, eps=0.15):
    """
    Round a continuous array to {-1, +1}.  Entries not within eps of ±1 → NaN.

    Parameters
    ----------
    x_cont : np.ndarray of any shape, dtype float
    eps    : float

    Returns
    -------
    np.ndarray, same shape, float (NaN where ambiguous)
    """
    out = np.full_like(x_cont, np.nan)
    out[np.abs(x_cont - 1) < eps]  =  1.0
    out[np.abs(x_cont + 1) < eps]  = -1.0
    return out


# ---------------------------------------------------------------------------
# Evaluation functions (continuous input from diffusion model)
# ---------------------------------------------------------------------------

def _base_snap(x_cont, n, eps):
    """Shared snap + nan-filter logic. Returns (snapped_flat, nan_mask, non_nan_snapped)."""
    x = np.asarray(x_cont).reshape(len(x_cont), -1)   # (N, n²)
    snapped = snap_to_binary(x, eps=eps)
    nan_mask = np.isnan(snapped).any(axis=1)
    return snapped, nan_mask, snapped[~nan_mask].astype(int)


def _per_row_stats(snapped_valid, n, K_set):
    """
    Per-row statistics on non-nan snapped samples.

    Returns
    -------
    counts         : (M, n) int   — active count per row
    per_row_ok     : (M, n) bool  — each row independently satisfies the rule
    per_row_ratio  : float        — fraction of (sample, row) pairs satisfying rule
    k_hist         : dict {k: int}  — histogram of per-row active counts (all rows, all samples)
    """
    counts = per_row_counts(snapped_valid, n)                    # (M, n)
    per_row_ok = np.array([[c in K_set for c in row] for row in counts])  # (M, n)
    per_row_ratio = float(per_row_ok.mean())
    k_hist = {int(k): int(v)
              for k, v in zip(*np.unique(counts, return_counts=True))}
    return counts, per_row_ok, per_row_ratio, k_hist


def evaluate_row_k_samples(x_cont, n, K, eps=0.15):
    """
    Evaluate continuous diffusion samples against the row-K rule (fixed K).

    Returns
    -------
    dict with keys
      nan_ratio          : fraction of samples with any unsnappable cell
      per_row_k_hist     : {k: count} histogram of active counts across all rows
      per_row_valid_ratio: fraction of (sample, row) pairs satisfying K constraint
      row_k_valid        : fraction of non-nan samples where ALL rows satisfy K (cond. on non-nan)
      full_valid_ratio   : row_k_valid * (1 - nan_ratio)
      valid_int          : (M, n²) int  snapped samples passing ALL-rows check
    """
    snapped, nan_mask, vs = _base_snap(x_cont, n, eps)
    nan_ratio = float(nan_mask.mean())
    M = len(vs)
    empty = dict(nan_ratio=nan_ratio, per_row_k_hist={}, per_row_valid_ratio=0.0,
                 row_k_valid=0.0, full_valid_ratio=0.0, valid_int=vs)
    if M == 0:
        return empty

    K_set = {K}
    counts, per_row_ok, per_row_ratio, k_hist = _per_row_stats(vs, n, K_set)
    sample_ok = per_row_ok.all(axis=1)

    return dict(
        nan_ratio=nan_ratio,
        per_row_k_hist=k_hist,
        per_row_valid_ratio=per_row_ratio,
        row_k_valid=float(sample_ok.mean()),
        full_valid_ratio=float(sample_ok.mean() * (1 - nan_ratio)),
        valid_int=vs[sample_ok],
    )


def evaluate_row_variable_k_samples(x_cont, n, K_list, eps=0.15):
    """
    Evaluate against row-variable-K rule (each row K ∈ K_list independently).

    Returns
    -------
    dict with keys
      nan_ratio          : fraction of samples with any unsnappable cell
      per_row_k_hist     : {k: count} histogram of active counts across all rows
      per_row_valid_ratio: fraction of (sample, row) pairs with count ∈ K_list
      row_var_k_valid    : fraction of non-nan samples where ALL rows satisfy rule
      full_valid_ratio   : row_var_k_valid * (1 - nan_ratio)
      valid_int          : (M, n²) int  snapped samples passing ALL-rows check
    """
    snapped, nan_mask, vs = _base_snap(x_cont, n, eps)
    nan_ratio = float(nan_mask.mean())
    M = len(vs)
    empty = dict(nan_ratio=nan_ratio, per_row_k_hist={}, per_row_valid_ratio=0.0,
                 row_var_k_valid=0.0, full_valid_ratio=0.0, valid_int=vs)
    if M == 0:
        return empty

    K_set = set(K_list)
    counts, per_row_ok, per_row_ratio, k_hist = _per_row_stats(vs, n, K_set)
    sample_ok = per_row_ok.all(axis=1)

    return dict(
        nan_ratio=nan_ratio,
        per_row_k_hist=k_hist,
        per_row_valid_ratio=per_row_ratio,
        row_var_k_valid=float(sample_ok.mean()),
        full_valid_ratio=float(sample_ok.mean() * (1 - nan_ratio)),
        valid_int=vs[sample_ok],
    )


def evaluate_global_k_samples(x_cont, n, K_list, eps=0.15):
    """
    Evaluate against global-K rule (all rows share same K, K ∈ K_list).

    Returns
    -------
    dict with keys
      nan_ratio           : fraction of samples with any unsnappable cell
      per_row_k_hist      : {k: count} histogram of active counts across all rows
      per_row_valid_ratio : fraction of (sample, row) pairs with count ∈ K_list
      row_consistency     : fraction of non-nan samples where ALL rows have same count
                            (regardless of K_list membership)
      global_k_valid      : fraction of non-nan samples satisfying global-K rule
      full_valid_ratio    : global_k_valid * (1 - nan_ratio)
      per_k_breakdown     : {k: count} of valid samples that used each K value
      valid_int           : (M, n²) int  snapped samples passing global-K check
    """
    snapped, nan_mask, vs = _base_snap(x_cont, n, eps)
    nan_ratio = float(nan_mask.mean())
    M = len(vs)
    empty = dict(nan_ratio=nan_ratio, per_row_k_hist={}, per_row_valid_ratio=0.0,
                 row_consistency=0.0, global_k_valid=0.0, full_valid_ratio=0.0,
                 per_k_breakdown={k: 0 for k in K_list}, valid_int=vs)
    if M == 0:
        return empty

    K_set = set(K_list)
    counts, per_row_ok, per_row_ratio, k_hist = _per_row_stats(vs, n, K_set)

    # row_consistency: all rows in a sample have the same count (any count)
    row_counts_min = counts.min(axis=1)
    row_counts_max = counts.max(axis=1)
    consistent = (row_counts_min == row_counts_max)

    # global_k_valid: consistent AND the shared count is in K_list
    global_ok, k_used = check_global_k_batch(vs, n, K_list)

    per_k = {k: int((k_used[global_ok] == k).sum()) for k in K_list}

    return dict(
        nan_ratio=nan_ratio,
        per_row_k_hist=k_hist,
        per_row_valid_ratio=per_row_ratio,
        row_consistency=float(consistent.mean()),
        global_k_valid=float(global_ok.mean()),
        full_valid_ratio=float(global_ok.mean() * (1 - nan_ratio)),
        per_k_breakdown=per_k,
        valid_int=vs[global_ok],
    )


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def valid_set_size_row_k(n, K):
    """C(n, K)^n — number of valid grids for fixed K per row."""
    return comb(n, K) ** n


def valid_set_size_row_variable_k(n, K_list):
    """(Σ_{k ∈ K_list} C(n,k))^n — each row draws K independently from K_list."""
    return sum(comb(n, k) for k in K_list) ** n


def valid_set_size_global_k(n, K_list):
    """Σ_{k ∈ K_list} C(n,k)^n — one global K drawn per sample."""
    return sum(comb(n, k) ** n for k in K_list)


def expected_memorization_ratio(N, n, rule, K=None, K_list=None):
    """
    N / valid_set_size for the given rule.

    Parameters
    ----------
    N      : int  -- training set size
    n      : int  -- grid size
    rule   : str  -- 'row_k' | 'row_variable_k' | 'global_k'
    K      : int  -- required for 'row_k'
    K_list : list -- required for 'row_variable_k' and 'global_k'
    """
    if rule == "row_k":
        vs = valid_set_size_row_k(n, K)
    elif rule == "row_variable_k":
        vs = valid_set_size_row_variable_k(n, K_list)
    elif rule == "global_k":
        vs = valid_set_size_global_k(n, K_list)
    else:
        raise ValueError(f"Unknown rule: {rule}")
    return N / vs


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    n = 6
    N = 4096

    print("=== Valid set sizes (n=6) ===")
    print(f"  row_k K=1          : {valid_set_size_row_k(n,1):>20,}  (=row-only baseline)")
    print(f"  row_k K=2          : {valid_set_size_row_k(n,2):>20,}")
    print(f"  row_k K=3          : {valid_set_size_row_k(n,3):>20,}")
    print(f"  row_variable_k K={{1,5}}: {valid_set_size_row_variable_k(n,[1,5]):>20,}")
    print(f"  row_variable_k K={{1,4}}: {valid_set_size_row_variable_k(n,[1,4]):>20,}")
    print(f"  row_variable_k K={{3,4,5,6}}: {valid_set_size_row_variable_k(n,[3,4,5,6]):>20,}")
    print(f"  global_k K={{1,5}}  : {valid_set_size_global_k(n,[1,5]):>20,}")
    print(f"  global_k K={{2,3}}  : {valid_set_size_global_k(n,[2,3]):>20,}")
    print(f"  global_k K={{3,4}}  : {valid_set_size_global_k(n,[3,4]):>20,}")

    print("\n=== Memorization ratios (N=4096) ===")
    for rule, kwargs in [
        ("row_k",          {"K": 2}),
        ("row_k",          {"K": 3}),
        ("row_variable_k", {"K_list": [1,5]}),
        ("row_variable_k", {"K_list": [3,4]}),
        ("global_k",       {"K_list": [1,5]}),
        ("global_k",       {"K_list": [2,3]}),
        ("global_k",       {"K_list": [3,4]}),
    ]:
        mr = expected_memorization_ratio(N, n, rule, **kwargs)
        print(f"  {rule:<20} {str(kwargs):<25} mem_ratio = {mr:.3e}")

    print("\n=== Sampling correctness ===")
    # row_k
    t0 = time.time()
    grids = sample_row_k_batch(N, n, K=2)
    elapsed = time.time() - t0
    flat = grids.reshape(N, -1)
    ok = check_row_k_batch(flat, n, K=2)
    print(f"  row_k K=2: {ok.mean():.4f} valid, {elapsed*1000:.1f}ms")

    # row_variable_k
    t0 = time.time()
    grids_var = sample_row_variable_k_batch(N, n, K_list=[1,5])
    elapsed = time.time() - t0
    flat_var = grids_var.reshape(N, -1)
    ok_var = check_row_variable_k_batch(flat_var, n, K_list=[1,5])
    counts_var = per_row_counts(flat_var, n)
    print(f"  row_variable_k {{1,5}}: {ok_var.mean():.4f} valid, {elapsed*1000:.1f}ms")
    print(f"    K distribution: {dict(zip(*np.unique(counts_var, return_counts=True)))}")

    # global_k
    t0 = time.time()
    grids_gl, k_labs = sample_global_k_batch(N, n, K_list=[2,3])
    elapsed = time.time() - t0
    flat_gl = grids_gl.reshape(N, -1)
    ok_gl, k_used = check_global_k_batch(flat_gl, n, K_list=[2,3])
    print(f"  global_k {{2,3}}: {ok_gl.mean():.4f} valid, {elapsed*1000:.1f}ms")
    print(f"    K label distribution: {dict(zip(*np.unique(k_labs, return_counts=True)))}")

    print("\n=== Evaluate on noisy samples ===")
    true_grids = sample_row_k_batch(100, n, K=3).astype(float)
    noisy = true_grids + np.random.randn(*true_grids.shape) * 0.05
    res = evaluate_row_k_samples(noisy, n, K=3, eps=0.2)
    print(f"  row_k K=3 (noisy): nan={res['nan_ratio']:.3f}, row_k_valid={res['row_k_valid']:.3f}")

    print("\n=== Dataset uniqueness ===")
    ds = sample_row_k_dataset(1000, n, K=2)
    n_uniq = len(set(tuple(r.tolist()) for r in ds.reshape(1000,-1)))
    print(f"  row_k K=2 dataset: {n_uniq}/1000 unique")
