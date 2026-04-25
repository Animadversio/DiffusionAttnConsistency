"""
exact_k_lib.py

Dataset generation and evaluation for the Exact-K rule:
  A D-dimensional binary vector (e.g. 6x6 image) must contain exactly K
  entries equal to +1 (and D-K entries equal to -1).

Convention (matching parity_lib.py):
  - Values in {-1, +1}
  - "pixel = 1"  ->  x_i = +1
  - "pixel = 0"  ->  x_i = -1
  - Exact-K condition: sum(x) == 2*K - D

Valid set size: C(D, K)  (binomial coefficient)

Energy / Score analysis
-----------------------
A natural continuous-space energy for the exact-K distribution is:

    E(x) = (1/2) * sum_i (x_i^2 - 1)^2
          + lambda_k * (sum_i x_i - (2K - D))^2

Term 1: local boolean-cube regularizer (same as parity energy)
Term 2: quadratic penalty on the global sum deviation

Score (gradient w.r.t. x_i):

    d/dx_i E = 2*x_i*(x_i^2 - 1)  +  2*lambda_k*(sum_j x_j - (2K - D))

The second term is a GLOBAL SUM — degree 1 in each x_i.
This contrasts sharply with parity, whose score involves a degree-G product.
A single attention head can compute a global sum in one step, so exact-K
is expected to be much easier to learn than high-G parity.
"""

import numpy as np
from math import comb
import torch as th


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_exact_k(D, K):
    """
    Sample one vector x in {-1,+1}^D with exactly K entries equal to +1.

    Parameters
    ----------
    D : int  -- vector length (e.g. 36 for a 6x6 image)
    K : int  -- number of +1 entries (0 <= K <= D)

    Returns
    -------
    x : np.ndarray of shape (D,), dtype int
    """
    x = np.full(D, -1, dtype=int)
    x[np.random.choice(D, size=K, replace=False)] = 1
    return x


def sample_exact_k_vec(N, D, K):
    """
    Sample N vectors in {-1,+1}^D each with exactly K entries equal to +1.
    Vectorized via the argsort trick (no Python loop over N).

    Parameters
    ----------
    N : int  -- number of samples
    D : int  -- vector length
    K : int  -- number of +1 entries

    Returns
    -------
    x : np.ndarray of shape (N, D), dtype int
    """
    x = np.full((N, D), -1, dtype=int)
    # For each row, draw K positions uniformly without replacement
    idx = np.argsort(np.random.rand(N, D), axis=1)[:, :K]
    np.put_along_axis(x, idx, 1, axis=1)
    return x


def sample_ensuring_uniqueness(N, sample_func):
    """
    Collect exactly N unique samples using rejection / batch sampling.
    Identical pattern to parity_lib.py.

    Parameters
    ----------
    N           : int  -- number of unique samples desired
    sample_func : callable(N=int) -> np.ndarray of shape (N, D)

    Returns
    -------
    np.ndarray of shape (N, D)
    """
    collected_samples = []
    seen = set()

    while len(collected_samples) < N:
        batch_size = max(N - len(collected_samples), N // 2)
        x_batch = sample_func(N=batch_size)
        batch_tuples = set(tuple(row) for row in x_batch)
        new_samples = batch_tuples - seen
        seen.update(new_samples)
        for sample_tuple in new_samples:
            collected_samples.append(np.array(sample_tuple))
            if len(collected_samples) == N:
                break

    return np.array(collected_samples)


def sample_exact_k_dataset(N, D, K):
    """
    Sample N *unique* vectors in {-1,+1}^D with exactly K ones.

    Parameters
    ----------
    N : int  -- number of unique samples
    D : int  -- vector length
    K : int  -- number of +1 entries

    Returns
    -------
    np.ndarray of shape (N, D), dtype int
    """
    if N > comb(D, K):
        raise ValueError(
            f"Cannot sample {N} unique vectors: only C({D},{K})={comb(D,K)} valid samples exist."
        )
    return sample_ensuring_uniqueness(N, sample_func=lambda N: sample_exact_k_vec(N, D, K))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def ones_count(x):
    """
    Count the number of +1 entries per sample.

    Parameters
    ----------
    x : np.ndarray of shape (..., D) in {-1, +1}

    Returns
    -------
    np.ndarray of shape (...,) with integer counts
    """
    return (x.sum(axis=-1) + x.shape[-1]) // 2


def exact_k_check(x, K):
    """
    Boolean array: True if a sample has exactly K ones.

    Parameters
    ----------
    x : np.ndarray of shape (N, D) in {-1, +1}
    K : int

    Returns
    -------
    np.ndarray of shape (N,), dtype bool
    """
    return ones_count(x) == K


def exact_k_accuracy(x, K):
    """
    Fraction of samples satisfying the exact-K constraint.

    Parameters
    ----------
    x : np.ndarray of shape (N, D) in {-1, +1}
    K : int

    Returns
    -------
    float in [0, 1]
    """
    return exact_k_check(x, K).mean()


# ---------------------------------------------------------------------------
# Combinatorics / memorization helpers
# ---------------------------------------------------------------------------

def valid_set_size(D, K):
    """
    Total number of valid samples: C(D, K).

    Parameters
    ----------
    D : int  -- vector length
    K : int  -- number of +1 entries

    Returns
    -------
    int
    """
    return comb(D, K)


def expected_memorization_ratio(N, D, K):
    """
    Expected sample-level memorization ratio if the model learns the true
    uniform distribution over all C(D, K) valid samples.

    Defined as N / C(D, K).  Values > 1 mean the training set covers the
    full valid set (complete memorization is inevitable).

    Parameters
    ----------
    N : int  -- training set size
    D : int  -- vector length
    K : int  -- number of +1 entries

    Returns
    -------
    float
    """
    return N / valid_set_size(D, K)


# ---------------------------------------------------------------------------
# Binarization utility (torch, same as parity_lib.py)
# ---------------------------------------------------------------------------

def round_to_pos_neg_one(x, eps=1e-2):
    """
    Binarize a continuous tensor to {-1, +1}.
    Entries not within eps of ±1 are left as NaN (invalid).

    Parameters
    ----------
    x   : torch.Tensor
    eps : float  -- tolerance

    Returns
    -------
    torch.Tensor  same shape as x
    """
    result = th.full_like(x, float('nan'))
    result[th.abs(x - 1) < eps] = 1.0
    result[th.abs(x + 1) < eps] = -1.0
    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    D = 36   # 6x6 image

    print("=== single sample ===")
    for K in [1, 6, 18, 30, 35]:
        x = sample_exact_k(D, K)
        cnt = ones_count(x.reshape(1, -1))[0]
        ok  = exact_k_check(x.reshape(1, -1), K)[0]
        print(f"  K={K:2d}: ones={cnt}, check={ok}")

    print("\n=== batch accuracy (should be 1.0) ===")
    for K in [1, 6, 12, 18, 24, 30, 35]:
        x = sample_exact_k_vec(N=1000, D=D, K=K)
        acc = exact_k_accuracy(x, K)
        print(f"  K={K:2d}: accuracy={acc:.4f}")

    print("\n=== unique dataset ===")
    K = 18
    N = 4096
    x_ds = sample_exact_k_dataset(N=N, D=D, K=K)
    n_unique = len(np.unique(x_ds, axis=0))
    print(f"  Requested {N}, got {n_unique} unique samples (K={K})")

    print("\n=== valid set sizes and memorization ratios (N=4096) ===")
    for K in [1, 2, 3, 6, 9, 12, 18, 24, 30, 33, 35, 36]:
        vs  = valid_set_size(D, K)
        mr  = expected_memorization_ratio(N, D, K)
        print(f"  K={K:2d}: C({D},{K:2d}) = {vs:>15,}  mem_ratio = {mr:.3e}")
