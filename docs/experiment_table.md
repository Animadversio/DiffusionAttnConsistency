# Experiment Inventory Table

All runs use length-36 sequences (or 6×6 grids). N = training set size.

---

## Rule Family 1: Even Parity (D=36, binary ±1)

Rule: within each group of G consecutive bits, the product = +1 (even parity).
Valid space = 2^(36/G − 1).

### DiT models (diffusion)

| Model    | Architecture | N values                            | G values trained           | # runs |
|----------|--------------|-------------------------------------|----------------------------|--------|
| DiT-nano | 3L 6H 384D   | 4096                                | 2, 3, 4, 6, 9, 12, 18, 36 | 8      |
| DiT-mini | 6L 6H 384D   | 1024, 2048, 4096, 8192, 16384, 65536| 2, 3, 4, 6, 9, 12, 18, 36 | 48     |
| DiT-S    | 12L 6H 384D  | 4096                                | 2, 3, 4, 6, 9, 12, 18, 36 | 8      |
| DiT-B    | 12L 12H 768D | 4096                                | 2, 3, 4, 6, 9, 12, 18, 36 | 8      |

### GPT models (autoregressive)

| Model    | Architecture | N values                    | G values trained           | # runs |
|----------|--------------|-----------------------------|----------------------------|--------|
| GPT-nano | 3L 3H 192D   | 4096                        | 2, 3, 4, 6, 9, 12, 18, 36 | 8      |
| GPT-mini | 6L 6H 384D   | 4096, 8192, 16384, 32768    | 2, 3, 4, 6, 9, 12, 18, 36 | 32     |
| GPT-B    | 12L 12H 768D | 32768                       | 2, 3, 4, 6, 9, 12, 18, 36 | 8      |

**Subtotal: 120 runs**

---

## Rule Family 2: Exact-K (D=36, binary ±1)

Rule: exactly K of 36 bits are +1. Valid space = C(36, K).
All DiT-mini, N=4096.

| K  | Valid space       | # runs |
|----|-------------------|--------|
| 3  | C(36,3) = 7,140   | 1      |
| 4  | C(36,4) = 58,905  | 1      |
| 6  | C(36,6) = 1.95M   | 1      |
| 8  | C(36,8) = 30.3M   | 1      |
| 9  | C(36,9) = 94.1M   | 1      |
| 12 | C(36,12) = 1.25B  | 1      |
| 18 | C(36,18) = 9.1B   | 1      |

**Subtotal: 7 runs**

---

## Rule Family 3: Multi-valued Grid Rules (n×n, values 0..n−1)

Encoding: scalar (uniform {−1..+1}) or one-hot (n² channels × n classes).

### Latin square variants (row+col, row-only, row+col+block)

| Model    | Rule                    | Valid space        | Encoding              | # runs |
|----------|-------------------------|--------------------|-----------------------|--------|
| DiT-mini | Row-only (row perm.)    | (6!)^6 = 2.18B     | scalar                | 1      |
| DiT-mini | Row-only                | 2.18B              | onehot zero-mean σ=auto | 1    |
| DiT-mini | Latin square (row+col)  | ~812M              | scalar                | 1      |
| DiT-mini | Latin square            | ~812M              | onehot {−1,+1} σ=1   | 1      |
| DiT-mini | Latin square            | ~812M              | onehot zero-mean σ=auto | 1    |
| DiT-mini | Latin square            | ~812M              | onehot {0,1} σ=auto  | 1      |
| DiT-mini | Latin square n=5        | 161,280            | scalar                | 1      |
| DiT-mini | Latin square n=5        | 161,280            | onehot                | 1      |
| DiT-B    | Latin square n=5        | 161,280            | scalar                | 1      |
| DiT-B    | Latin square n=5        | 161,280            | onehot                | 1      |
| DiT-B    | Latin square n=6        | ~812M              | scalar                | 1      |
| DiT-B    | Latin square n=6        | ~812M              | onehot {−1,+1} σ=1   | 1      |
| DiT-mini | Sudoku 6×6 (row+col+2×3 block) | 28,200,960 | scalar           | 1      |
| DiT-mini | Sudoku 6×6              | 28,200,960         | onehot zero-mean σ=auto | 1    |

**Subtotal: 14 runs**

---

## Rule Family 4: Row-K Binary Grid Rules (6×6, binary ±1)

Rule: each row of a 6×6 ±1 binary grid satisfies a count constraint on active (+1) cells.
All DiT-mini, N=4096, onehot zero-mean σ=auto.

### Fixed-K (all rows must have exactly K active cells)

| Rule    | K   | Valid space (per-row C(6,K)^6) | # runs |
|---------|-----|--------------------------------|--------|
| row_k   | K=2 | C(6,2)^6 = 11.4M              | 1      |
| row_k   | K=3 | C(6,3)^6 = 64M                | 1      |

### Per-row Variable-K (each row independently draws K from K_list)

| Rule       | K_list    | Valid space (approx)           | # runs |
|------------|-----------|--------------------------------|--------|
| row_var_k  | {1,5}     | (C(6,1)+C(6,5))^6 = 12^6 = 3M | 1      |
| row_var_k  | {3,4}     | (C(6,3)+C(6,4))^6 = 35^6 = 1.8B | 1    |
| row_var_k  | {0,2,4,6} | (1+15+15+1)^6 = 32^6 = 1B     | 1      |
| row_var_k  | {3,4,5,6} | (20+15+6+1)^6 = 42^6 = 5.5B   | 1      |

### Global-K (one K drawn per sample; all rows use same K)

| Rule      | K_list | Valid space (sum of C(6,K)^6) | # runs |
|-----------|--------|-------------------------------|--------|
| global_k  | {1,5}  | 2 × C(6,1)^6 = 93k            | 1      |
| global_k  | {2,4}  | 2 × C(6,2)^6 = 22.8M          | 1      |

**Subtotal: 8 runs**

---

## Grand Total

| Rule family                  | # runs |
|------------------------------|--------|
| Even parity (DiT)            | 72     |
| Even parity (GPT)            | 48     |
| Exact-K (DiT-mini)           | 7      |
| Multi-valued grid (DiT)      | 14     |
| Row-K binary grid (DiT-mini) | 8      |
| **Total**                    | **149** |

---

## Notes

- All DiT runs: 1M training steps, N=4096 unless noted.
- All GPT runs: 100k training steps, parity rule only, N varies (4096–32768).
- GPT architecture: autoregressive transformer (next-token prediction on ±1 sequences).
- DiT architecture: diffusion transformer (EDM2 noise schedule).
- "Valid space" = number of distinct valid samples under the rule.
