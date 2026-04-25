# Latin Square Learning Benchmark

## Overview

An **n×n Latin square** is a grid filled with n symbols {0,...,n-1} where each symbol appears exactly once in every row and every column. This benchmark studies whether a DiT diffusion model can learn to generate valid Latin squares from a finite training set.

Latin squares sit in a middle ground of constraint complexity:
- **Exact-K**: 1 global sum constraint — degree 1, trivially easy
- **Latin square**: 2n local permutation constraints (n row + n col) — intermediate
- **Parity (group-G)**: nonlinear degree-G product — hardest

## Valid Set Sizes

| n | # valid Latin squares | D (int enc.) | D (onehot) | Recommended N | Mem ratio |
|---|---|---|---|---|---|
| 4 | 576 | 16 | 64 | 128 or 432 | 22% or 75% |
| 5 | 161,280 | 25 | 125 | 4096 | 2.5% |
| 6 | 812,851,200 | 36 | 216 | 4096 | 5e-6 |

For n=6 with D=36, the architecture and training setup are directly comparable to the exact-K and parity benchmarks (same D, same DiT-mini config).

## Encoding

**Primary: integer encoding**
- Each cell value ∈ {0,...,n-1} is normalized to [-1,+1]:
  `x_norm = 2 * v / (n-1) - 1`
- For n=6: {0,1,2,3,4,5} → {-1.0, -0.6, -0.2, +0.2, +0.6, +1.0}
- Flat representation: shape (n²,)
- Snapping tolerance eps=0.15 (gap between adjacent valid floats is 0.4 for n=6)

**Optional: one-hot encoding**
- Each cell → n-bit one-hot vector
- D = n³ (n=6: 216, n=4: 64)
- Binary like parity/exact-K, easier binarization

## Sampling Strategy

**Permutation-based (fast, non-uniform):**
1. Start from cyclic base: `base[i][j] = (i+j) % n`
2. Randomly permute row order, column order, and symbol mapping
3. For n=4: covers all 576 squares (one isotopy class)
4. For n=6: covers ~373M / 812M squares (~46% of all valid squares)

This is sufficient for rule-learning experiments. For uniformly random Latin squares, the Jacobson-Matthews algorithm would be needed.

## Evaluation Metrics

| Metric | Description |
|---|---|
| `eval/nan_ratio` | Fraction of samples that couldn't be snapped to the integer grid |
| `eval/row_valid_ratio` | Fraction of valid-snapped samples where ALL n rows are valid permutations |
| `eval/col_valid_ratio` | Fraction of valid-snapped samples where ALL n columns are valid permutations |
| `eval/full_valid_ratio` | Fraction satisfying BOTH row and col constraints (true Latin squares) |
| `eval/sample_mem_ratio` | Fraction of valid gen. samples appearing verbatim in training set |

## Energy / Score Analysis

A natural energy for this distribution is:

```
E(x) = λ_cube * Σ_i (x_i² - 1)²               # local boolean-cube (for binary)
      + λ_row * Σ_r (Σ_{j in row r} x_j - target_sum)²   # row sum constraints
      + λ_col * Σ_c (Σ_{i in col c} x_i - target_sum)²   # col sum constraints
```

But sum constraints alone are necessary but not sufficient for Latin squares — the all-distinct condition requires higher-order interactions. The model must learn to represent permutation constraints implicitly.

## Scientific Questions

1. Does DiT learn Latin squares at all? (And how fast vs. exact-K?)
2. Does the model learn row and column constraints simultaneously or sequentially?
3. Does n=4 with N=128 (partial coverage) exhibit generalization or memorization?
4. How does the difficulty scale with n?

## File Structure

```
core/latin_square_lib.py          # Dataset lib (sampling, encoding, evaluation)
scripts/DiT_learn_latin_sq_CLI.py # Training script
bash/DiT_edm_learn_latin_sq.sh    # SLURM array job
```

## Relationship to Other Benchmarks

All experiments use the same DiT-mini architecture: 6 layers, 6 heads, hidden size 384, patch_size=1.

| Rule | D | Valid set | Constraint type |
|---|---|---|---|
| Parity G=1 | 36 | 2^35 ≈ 34B | Degree-1 product (trivial) |
| Exact-K=18 | 36 | 9.1B | Degree-1 sum (easy) |
| Latin sq. n=6 | 36 | 813M | Permutation (intermediate?) |
| Parity G=36 | 36 | 2 | Degree-36 product (hardest) |
