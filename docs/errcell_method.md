# Error Cell Confidence Analysis — Method

## Overview

We test whether cells that cause rule violations in a diffusion model's decoded output
are systematically less confident than correct cells. The analysis applies to Latin
square samples decoded from two encoding families: one-hot and scalar.

---

## Step 1: Decoding and Confidence Estimation

### One-hot encoding

The model outputs an `(N, n, n²)` tensor where the `n` channels over each of the `n²`
cells form a soft one-hot vector. The confidence for cell `j` in sample `i` is:

```
confidence_ij = (max_channel_ij − inactive) / (active − inactive)
```

This normalizes the range `[inactive, active]` to `[0, 1]`.
- `confidence = 1.0` → perfectly sharp one-hot (maximum certainty)
- `confidence = 0.0` → completely flat channel distribution (no information)

The decoded integer is `argmax` over channels (no thresholding for this analysis).

Encoding variants and their active/inactive values:

| Variant | active | inactive | RMS (σ_data) |
|---------|--------|----------|--------------|
| `{-1,+1}` (pm1) | 1.0 | −1.0 | 1.000 |
| `{0,1}` (zero_one) | 1.0 | 0.0 | 0.408 |
| zero-mean | (n−1)/n | −1/n | 0.373 |

### Scalar encoding

The model outputs a scalar per cell in `[0, 1]`. Valid levels are equally spaced:
`v_k = k / (n−1)` for `k = 0, …, n−1`. The confidence is:

```
confidence_ij = 1 − dist_to_nearest / half_spacing
```

where `half_spacing = 1 / (n−1) / 2`.
- `confidence = 1.0` → cell value exactly on a valid level
- `confidence = 0.0` → cell value at the midpoint between two valid levels
- `confidence < 0.0` → farther than half-spacing from any level (unsnappable)

For decoding, each cell is snapped to its nearest valid level if within `eps` of it
(default `eps = 0.15`). Samples where any cell fails to snap are excluded.

---

## Step 2: Identifying Error Cells

A cell at position `(r, c)` in sample `i` is called an **error cell** if its decoded
symbol appears more than once in row `r` **or** more than once in column `c`.

Algorithm:
```
for each row r:
    for each symbol v in 0..n-1:
        find samples where v appears ≥ 2 times in row r
        mark all occurrences of v in row r as error cells

for each column c:
    for each symbol v in 0..n-1:
        find samples where v appears ≥ 2 times in column c
        mark all occurrences of v in column c as error cells
```

This correctly identifies all cells participating in a constraint violation
(both copies of a duplicated symbol are flagged, not just one).

---

## Step 3: Uncertainty Comparison

We define **uncertainty** as `1 − confidence` and split all cells from the
last `n_ckpts` checkpoints into two groups:

- **Correct cells**: cells whose decoded symbol does not cause any violation
- **Error cells**: cells whose decoded symbol participates in a row or column duplicate

We compare the uncertainty distributions via histograms (zoomed to `[0, 0.10]`)
and summary statistics.

---

## Key Results (n=6, last 10 checkpoints)

| Metric | {-1,+1} onehot (correct) | {-1,+1} onehot (error) | scalar (correct) | scalar (error) |
|--------|--------------------------|------------------------|------------------|----------------|
| N cells | 717,680 | 19,600 | 677,434 | 56,714 |
| Error rate | — | 2.7% | — | 7.7% |
| Mean uncertainty | 0.0004 | 0.0104 | 0.0047 | 0.0117 |
| Std uncertainty | 0.0032 | 0.0317 | 0.0059 | 0.0339 |
| % unc < 0.01 (conf > 0.99) | 99.1% | 67.6% | 90.6% | 65.3% |
| % unc < 0.05 (conf > 0.95) | 100.0% | 98.7% | 100.0% | 98.3% |

**Finding**: Error cells have ~10× higher uncertainty std and ~25× lower rate of near-perfect
confidence (> 0.99) compared to correct cells. This holds for both encoding families.

---

## Code

- `scripts/analyze_error_cell_confidence.py` — main analysis and figures
  - `find_error_cells(decoded_int, n)` — identify error cell mask
  - `onehot_cell_confidence(samples_oh, active, inactive)` — one-hot confidence
  - `scalar_cell_confidence(samples_sc, n, eps)` — scalar confidence
- `scripts/visualize_error_samples.py` — board + uncertainty map per example

---

## Usage

```bash
# One-hot encoding (n=6, last 10 checkpoints)
python scripts/analyze_error_cell_confidence.py \
    --exp_names DiT_mini_latinSq_n6_N4096_onehot \
                DiT_mini_latinSq_n6_N4096_scalar \
    --labels "{-1,+1} onehot n=6" "scalar n=6" \
    --n_sizes 6 6 --n_ckpts 10 \
    --outpath /tmp/errcell_n6.png
```

Outputs: `.png`, `.pdf` (uncertainty histogram), `_stats.md`, `_stats.tex` (tables).
