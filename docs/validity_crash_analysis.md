# Validity Crash Analysis: Late-Training Rule Validity Degradation

## Summary

Several training runs exhibit a "validity crash" — a drop in rule validity (full_valid_ratio) during
late training despite continued smooth loss decrease. The phenomenon has two distinct manifestations
depending on encoding type.

---

## Affected Runs

| Run | Encoding | Peak valid (step) | Final valid | Mem@final | Crash type |
|---|---|---|---|---|---|
| `DiT_mini_latinSq_n6_N4096_scalar` | scalar | ~80% @50k | ~59% @1M | ~40% | Memorization |
| `DiT_B_latinSq_n6_N4096_scalar` | scalar | ~92% @105k | ~91% @720k | ~84% | Memorization (mild) |
| `DiT_mini_latinSq_n6_N4096_onehot_zeromean_autoSD` | onehot | ~87% @273k | ~71% @1M | <0.1% | Confident-wrong |
| `DiT_mini_latinSq_n6_N4096_onehot_zeroone_autoSD` | onehot | ~83% @473k | ~78% @1M | <0.1% | Confident-wrong |
| `DiT_mini_latinSq_n6_N4096_onehot` (pm1) | onehot | ~81% @801k | ~76% @1M | <0.1% | Mild oscillation |
| `DiT_B_latinSq_n6_N4096_onehot` | onehot | ~87% @220k | ~60% @690k | <0.1% | Confident-wrong |
| `DiT_mini_rowOnly_n6_N4096_onehot_zeromean_autoSD` | onehot | ~100% @26k | ~83% @900k | <0.1% | Confident-wrong |

Runs **not** exhibiting significant crashes:
- `DiT_B_latinSq_n5_N4096_scalar/onehot` — n=5 is easier; scalar stays near 99%, onehot near 94%
- `DiT_mini_sudoku6x6_N4096_onehot_zeromean_autoSD` — stays near 90% validity throughout

---

## Mechanism 1: Memorization-driven degradation (scalar encoding)

### Pattern
- Rule validity peaks early (50k–100k steps), then decays steadily
- Memorization ratio (fraction of generated valid boards found in training set) rises in lockstep
- Loss continues to decrease smoothly throughout
- Nan ratio stays near zero (model remains confident)

### Correlation with memorization (mini n6 scalar example)
| Step | Net valid | Mem ratio |
|---|---|---|
| 10k | 28% | 0% |
| 50k | **80%** (peak) | 0% |
| 300k | 68% | 2% |
| 600k | 53% | 19% |
| 1M | 59% | **40%** |

### Interpretation
The model gradually shifts from generating diverse novel valid boards toward reproducing training
examples. As memorization grows, fewer generated samples are truly novel, but memorized boards
trivially satisfy rules — validity of novel (non-memorized) boards degrades.

The diffusion loss objective does not distinguish between memorized reproductions and novel valid
samples, so it provides no signal against memorization.

---

## Mechanism 2: Confident-wrong degradation (onehot encoding)

### Pattern
- Validity peaks, then decays or oscillates on a degraded plateau
- Memorization stays near 0% throughout (no memorization)
- Loss continues to decrease smoothly
- Nan ratio stays low (model is still confident — not hedging)
- Crashes appear as sudden drops (~5–20% validity in one eval step) followed by partial recovery

### What changes at a crash (from sample analysis)
Analyzed checkpoints before/after the worst crash in `zeromean_autoSD`
(step 681632→682656, Δvalid = −29%):

| Metric | Before | After |
|---|---|---|
| Nan ratio (boards) | 40.5% | 70.4% |
| Cond valid (of decoded) | 93.7% | 85.2% |
| Mean confidence | 0.980 | 0.962 |
| Pct cells < 0.99 conf | 2.9% | 8.2% |

Two sub-effects:
1. **Nan spike**: confidence drops slightly on many cells, pushing them below the decoding threshold.
   These boards were previously counted as valid (when confident), now counted as nan.
2. **Cond valid drop**: among decoded boards, row violation rate roughly doubles (~2.5% → ~5.8% per row),
   uniformly across all rows. Column violations remain exactly 0% at all checkpoints.

### Key findings from deep sample analysis
- **Column constraints are perfectly satisfied** at all time points including during crashes.
  The model internalizes column structure more robustly than row structure.
- **Row violations are the sole source of rule failures** in onehot runs.
- **Failing boards are confidently wrong**: confidence of cells in violated rows (0.9987) is
  virtually indistinguishable from cells in passing rows (0.9990). The model has no uncertainty
  signal for its own errors.
- **No detectable precursor in loss, nan, or mem**: cross-correlation analysis on smoothed traces
  shows loss has r≈0.10 correlation with validity changes. The validity drops are decoupled from
  all other observable training metrics.
- **Mid-range values duplicate preferentially**: values 1, 3, 4 appear as row duplicates more often
  than 0, 2, 5 — a structural bias present consistently before and after crashes.

### Loss-validity decoupling
On log step scale:
- Loss and validity both improve rapidly in phase 1 (0 → ~30k steps)
- After ~300k steps, loss decay *accelerates* (steeper log-log slope: −0.03 → −0.30)
  while validity begins to degrade
- This divergence is consistent with H3 (below)

### Hypotheses
**H1 (most likely): Fixed-seed eval stochasticity**
`eval_fix_noise_seed=True` — same noise is used every eval. A single weight update can slightly
shift the denoising trajectory for borderline boards. The crash is a genuine model shift, but the
*magnitude* (29% drop) is amplified by the fixed seed always hitting the same borderline cases.

**H2: Value-position marginal bias amplification**
The model develops a mild preference for certain value-position mappings (mid-range values more
likely in certain positions), increasing row collision probability for those values as training
progresses.

**H3: Loss-validity decoupling (supported by data)**
After the early learning phase, the EDM denoising loss improves fastest at intermediate noise levels
(σ ∈ [0.1, 10]), while final sample quality (σ → 0) may not improve proportionally or may
slightly degrade. The loss and validity objectives become partially decoupled.
*Note: this hypothesis requires per-sigma loss analysis to confirm — no model checkpoints were saved
for these runs, making direct testing impossible.*

---

## Comparison: scalar vs onehot

| Property | Scalar | Onehot |
|---|---|---|
| Memorization | Strong (40-80% late training) | Negligible (<0.1%) |
| Validity crash cause | Memorization displaces novel generation | Confident wrong predictions |
| Loss correlation | Decoupled (loss ↓, valid ↓) | Decoupled (loss ↓, valid oscillates) |
| Column constraint | Sometimes violated | Never violated (even during crashes) |
| Recovery from crash | Partial, slow | Yes — oscillates around plateau |
| Failure mode | Model "collapses" to training data | Model output distribution shifts subtly |

The scalar encoding is more vulnerable to memorization because the continuous output space
makes it easier for the diffusion model to reproduce exact training samples.
The onehot encoding's discrete argmax structure limits exact memorization but allows
confident-but-wrong predictions in a way that scalar quantization (snap-to-grid) does not.

---

## Relevant scripts and data

- Training: `scripts/DiT_learn_latin_sq_CLI.py`
- Sample analysis: `scripts/analyze_error_cell_confidence.py`, `scripts/visualize_error_samples.py`
- TensorBoard plots: `scripts/plot_tb_curves.py` (groups: `latinsq_encoding`, `latinsq_rules`, `latinsq_B`)
- Training data: `{exp_dir}/training_data_tsr.pt` (integer boards, N=4096)
- Eval samples: `{exp_dir}/samples/samples_epoch_{step:06d}.pt` (shape: `(2048, C, n, n)`)

No model checkpoints were saved for these runs (`ckpts/` directory is empty in all runs).
Future experiments should enable checkpoint saving to allow per-sigma loss analysis.
