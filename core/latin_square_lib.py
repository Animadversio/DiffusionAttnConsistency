"""
latin_square_lib.py

Dataset generation and evaluation for the Latin Square rule:
  An n×n grid filled with symbols {0,...,n-1}, each appearing exactly once
  in every row and every column.

Encoding (primary):
  Integer encoding, normalized to [-1, +1]:
    x_norm = 2 * v / (n - 1) - 1
  For n=6: {0,1,2,3,4,5} → {-1.0, -0.6, -0.2, +0.2, +0.6, +1.0}
  Flat representation: shape (n²,), dtype float32.

Valid set sizes (known):
  n=1:  1
  n=2:  2
  n=3:  12
  n=4:  576
  n=5:  161,280
  n=6:  812,851,200

Sampling strategy: permutation-based (fast, non-uniform).
  Start from a canonical cyclic Latin square, then randomly permute
  rows, columns, and symbols. Covers all 576 squares for n=4, and
  ~46% of all squares for n=6 (one isotopy class per base square).

Score / Energy analysis:
  The sum-per-row and sum-per-col constraints are degree-1 (like exact-K),
  but the all-distinct constraint requires higher-order interactions.
  This places Latin squares between exact-K and high-G parity in difficulty.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Known valid set sizes
# ---------------------------------------------------------------------------

LATIN_SQUARE_COUNTS = {1: 1, 2: 2, 3: 12, 4: 576, 5: 161_280, 6: 812_851_200}


def valid_set_size(n):
    """Total number of valid n×n Latin squares (exact, for n ≤ 6)."""
    if n not in LATIN_SQUARE_COUNTS:
        raise ValueError(f"Valid set size not known for n={n}. Known: {list(LATIN_SQUARE_COUNTS)}")
    return LATIN_SQUARE_COUNTS[n]


def expected_memorization_ratio(N, n):
    """N / valid_set_size(n). Values > 1 mean full coverage is inevitable."""
    return N / valid_set_size(n)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def make_cyclic_latin_square(n):
    """
    Canonical cyclic Latin square: base[i, j] = (i + j) % n.

    Returns
    -------
    np.ndarray of shape (n, n), dtype int
    """
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return (i_idx + j_idx) % n


def random_permute_latin_square(ls):
    """
    Randomly permute rows, columns, and symbols of a Latin square.
    The result is always a valid Latin square.

    Parameters
    ----------
    ls : np.ndarray of shape (n, n), integer grid

    Returns
    -------
    np.ndarray of shape (n, n), integer grid
    """
    n = ls.shape[0]
    ls = ls[np.random.permutation(n), :]          # permute rows
    ls = ls[:, np.random.permutation(n)]           # permute columns
    sym_perm = np.random.permutation(n)
    ls = sym_perm[ls]                              # permute symbols
    return ls


def sample_latin_square(n):
    """
    Sample one n×n Latin square (integer grid).

    Returns
    -------
    np.ndarray of shape (n, n), dtype int
    """
    return random_permute_latin_square(make_cyclic_latin_square(n))


def sample_latin_square_vec(N, n):
    """
    Sample N Latin squares, returned as a flat (N, n²) integer array.

    Parameters
    ----------
    N : int
    n : int

    Returns
    -------
    np.ndarray of shape (N, n*n), dtype int
    """
    return np.array([sample_latin_square(n).ravel() for _ in range(N)])


def sample_ensuring_uniqueness(N, sample_func):
    """
    Collect exactly N unique samples using rejection / batch sampling.

    Parameters
    ----------
    N           : int
    sample_func : callable(N=int) -> np.ndarray of shape (N, D)

    Returns
    -------
    np.ndarray of shape (N, D)
    """
    collected = []
    seen = set()
    while len(collected) < N:
        batch_size = max(N - len(collected), N // 2)
        x_batch = sample_func(N=batch_size)
        for row in x_batch:
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                collected.append(row)
                if len(collected) == N:
                    break
    return np.array(collected)


def sample_latin_square_dataset(N, n):
    """
    Sample N unique n×n Latin squares as a flat (N, n²) integer array.

    Parameters
    ----------
    N : int
    n : int

    Returns
    -------
    np.ndarray of shape (N, n*n), dtype int
    """
    vs = valid_set_size(n)
    if N > vs:
        raise ValueError(
            f"Cannot sample {N} unique {n}×{n} Latin squares: only {vs} valid squares exist."
        )
    return sample_ensuring_uniqueness(N, sample_func=lambda N: sample_latin_square_vec(N, n))


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------

def encode_latin_square(ls_int_flat, n):
    """
    Normalize integer values {0,...,n-1} to floats in [-1, +1].

    Parameters
    ----------
    ls_int_flat : np.ndarray of shape (..., n²), integer
    n           : int

    Returns
    -------
    np.ndarray of same shape, float
    """
    return 2.0 * ls_int_flat / (n - 1) - 1.0


def valid_float_values(n):
    """
    The n valid float values for integer encoding of an n-symbol grid.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    return np.array([2.0 * v / (n - 1) - 1.0 for v in range(n)])


def snap_to_integer(x_cont, n, eps=0.15):
    """
    Snap continuous float values to the nearest valid integer encoding.
    Cells too far from any valid value are set to NaN.

    Parameters
    ----------
    x_cont : np.ndarray of shape (N, n²) or (n²,)
    n      : int
    eps    : float, tolerance (gap between adjacent values is 2/(n-1); eps=0.15 works for n≥4)

    Returns
    -------
    np.ndarray of same shape, dtype float. NaN where snapping fails.
    """
    vf = valid_float_values(n)  # (n,)
    was_1d = x_cont.ndim == 1
    x = x_cont[None] if was_1d else x_cont  # (N, D)
    N, D = x.shape

    # distances: (N, D, n)
    dists = np.abs(x[:, :, None] - vf[None, None, :])
    nearest_idx = dists.argmin(axis=2)    # (N, D)
    nearest_dist = dists.min(axis=2)      # (N, D)

    result = nearest_idx.astype(float)
    result[nearest_dist > eps] = np.nan

    return result[0] if was_1d else result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_latin_square_batch(x_int_flat, n):
    """
    Vectorized check for a batch of integer-encoded (snapped) flat samples.

    Parameters
    ----------
    x_int_flat : np.ndarray of shape (M, n²), integer (no NaNs)
    n          : int

    Returns
    -------
    row_valid : np.ndarray of shape (M,), bool — all n rows are valid permutations
    col_valid : np.ndarray of shape (M,), bool — all n cols are valid permutations
    """
    M = len(x_int_flat)
    grids = x_int_flat.reshape(M, n, n)      # (M, n, n)
    expected = np.arange(n)                   # (n,)

    # Sort each row; compare to [0,...,n-1]
    rows_sorted = np.sort(grids, axis=2)                                # (M, n, n)
    row_ok = (rows_sorted == expected[None, None, :]).all(axis=2)       # (M, n)
    row_valid = row_ok.all(axis=1)                                       # (M,)

    # Sort each column; compare to [0,...,n-1]
    cols_sorted = np.sort(grids, axis=1)                                # (M, n, n)
    col_ok = (cols_sorted == expected[None, :, None]).all(axis=1)       # (M, n)
    col_valid = col_ok.all(axis=1)                                       # (M,)

    return row_valid, col_valid


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_latin_square_samples(x_flat_cont, n, eps=0.15):
    """
    Evaluate a batch of continuous flat samples from the diffusion model.

    Parameters
    ----------
    x_flat_cont : np.ndarray of shape (N, n²), continuous floats ≈ [-1, +1]
    n           : int
    eps         : float, snapping tolerance

    Returns
    -------
    dict with keys:
        nan_ratio        — fraction with any cell failing to snap
        row_valid_ratio  — fraction (of snappable) where all rows are valid
        col_valid_ratio  — fraction (of snappable) where all cols are valid
        full_valid_ratio — fraction (of snappable) satisfying both
        valid_int        — np.ndarray (M, n²) int, the snapped valid samples
    """
    N = len(x_flat_cont)
    x_int = snap_to_integer(x_flat_cont, n, eps=eps)   # (N, n²), NaN where bad

    nan_mask = np.isnan(x_int).any(axis=1)
    nan_ratio = float(nan_mask.mean())

    valid_int = x_int[~nan_mask].astype(int)
    M = len(valid_int)

    if M == 0:
        return dict(nan_ratio=nan_ratio, row_valid_ratio=0.0,
                    col_valid_ratio=0.0, full_valid_ratio=0.0,
                    valid_int=valid_int)

    row_valid, col_valid = check_latin_square_batch(valid_int, n)
    full_valid = row_valid & col_valid

    return dict(
        nan_ratio=nan_ratio,
        row_valid_ratio=float(row_valid.mean()),
        col_valid_ratio=float(col_valid.mean()),
        full_valid_ratio=float(full_valid.mean()),
        valid_int=valid_int,
    )


# ---------------------------------------------------------------------------
# Memorization helper
# ---------------------------------------------------------------------------

def compute_memorization(train_flat_int, gen_flat_int):
    """
    Fraction of generated valid (integer-snapped) samples that appear in the training set.

    Parameters
    ----------
    train_flat_int : np.ndarray of shape (N_train, n²), integer
    gen_flat_int   : np.ndarray of shape (M, n²), integer (no NaNs)

    Returns
    -------
    float in [0, 1]
    """
    if len(gen_flat_int) == 0:
        return 0.0
    train_set = set(tuple(row) for row in train_flat_int)
    count = sum(1 for row in gen_flat_int if tuple(row) in train_set)
    return count / len(gen_flat_int)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Cyclic base squares ===")
    for n in [3, 4, 6]:
        base = make_cyclic_latin_square(n)
        print(f"n={n}:\n{base}\n")

    print("=== Single sample validity ===")
    for n in [3, 4, 5, 6]:
        ls = sample_latin_square(n)
        flat = ls.ravel()
        row_v, col_v = check_latin_square_batch(flat[None], n)
        print(f"n={n}: row_valid={row_v[0]}, col_valid={col_v[0]}")

    print("\n=== Batch sampling & evaluation (N=200) ===")
    for n in [4, 6]:
        x_int = sample_latin_square_vec(200, n)
        x_norm = encode_latin_square(x_int, n)
        # Add tiny noise to simulate model output
        x_noisy = x_norm + np.random.randn(*x_norm.shape) * 0.05
        metrics = evaluate_latin_square_samples(x_noisy, n, eps=0.15)
        print(f"n={n}: nan={metrics['nan_ratio']:.3f} | row={metrics['row_valid_ratio']:.3f} "
              f"| col={metrics['col_valid_ratio']:.3f} | full={metrics['full_valid_ratio']:.3f}")

    print("\n=== Unique dataset ===")
    for n, N in [(4, 128), (4, 576), (6, 256)]:
        ds = sample_latin_square_dataset(N=N, n=n)
        n_unique = len(set(tuple(r) for r in ds))
        print(f"n={n}, N={N}: got {n_unique} unique samples (requested {N})")

    print("\n=== Encoding round-trip ===")
    for n in [4, 5, 6]:
        x_int = sample_latin_square_vec(100, n)
        x_norm = encode_latin_square(x_int, n)
        x_back = snap_to_integer(x_norm, n, eps=0.01).astype(int)
        match = (x_back == x_int).all()
        print(f"n={n}: round-trip ok={match}")

    print("\n=== Memorization helper ===")
    n = 4
    train = sample_latin_square_dataset(N=50, n=n)
    gen   = sample_latin_square_dataset(N=100, n=n)
    mem = compute_memorization(train, gen)
    print(f"n={n}: mem_ratio={mem:.3f} (train=50, gen=100 from 576 valid squares)")

    print("\n=== Valid set sizes & mem ratios (N=4096) ===")
    N = 4096
    for n_val in [3, 4, 5, 6]:
        vs = valid_set_size(n_val)
        mr = expected_memorization_ratio(N, n_val)
        print(f"n={n_val}: valid={vs:>12,}  mem_ratio(N={N}) = {mr:.3e}  D_int={n_val**2}")
