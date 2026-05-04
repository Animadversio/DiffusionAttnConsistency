# DiffusionAttnConsistency — Parity / Rule Learning Analysis

Research codebase for studying how diffusion models (DiT) and autoregressive models (GPT)
learn discrete rules (parity, exact-k, Latin square, row-k) from finite datasets, and
how rule learning vs. memorization emerge across training.

---

## Repo layout

```
core/           importable Python libraries
scripts/        CLI scripts and batch analysis tools
notebooks/      Jupyter exploration notebooks
outputs/        generated tables / cached results  (CSV, pickle)
figures/        saved publication figures
logs/           SLURM .err/.out files, organized by experiment type
Writings/       LaTeX paper drafts
```

---

## Core libraries (`core/`)

### Onset & timeseries analysis

| File | Purpose | Key functions |
|------|---------|---------------|
| `onset_lib.py` | Detect rule-learning / memorization onset from TensorBoard or CSV logs | `first_sustained_crossing`, `load_eval_timeseries`, `get_onsets`, `collect_onsets` |
| `dynamics_plot_lib.py` | Build and plot training-dynamics heatmaps (valid frac / mem ratio / innovation vs log-step × G) | `build_timeheatmap`, `plot_dynamics_heatmap`, `interp_row` |
| `run_registry.py` | Scan SAVEROOT, parse args.json, compute support_size and stat_mem_frac for all parity runs | `scan_parity_runs` |
| `fit_lib.py` | Power-law fitting on log-log axes, NaN-safe, returns slope/intercept/R²/n | `fit_loglog`, `fit_loglog_segment`, `plot_loglog_fit` |
| `gpt_eval_lib.py` | Load GPT checkpoints, compute per-position CE loss on train / valid-novel / boolean-cube splits | `load_gpt_config`, `load_gpt_checkpoint`, `compute_ce_loss`, `compute_per_position_loss`, `build_all_test_sets` |

### Rule / dataset libraries

| File | Rule type | Key functions |
|------|-----------|---------------|
| `parity_lib.py` | Group parity | `sample_parity`, `sample_group_parity_vec`, `sample_ensuring_uniqueness` |
| `exact_k_lib.py` | Exact-k (hamming weight = k) | `sample_exact_k_dataset`, `exact_k_accuracy`, `valid_set_size`, `expected_memorization_ratio` |
| `latin_square_lib.py` | Latin square / Sudoku | `sample_latin_square_dataset`, `evaluate_latin_square_samples`, `compute_memorization` |
| `row_k_lib.py` | Row-k / global-k counting rules | `sample_row_k_dataset`, `evaluate_row_k_samples`, `valid_set_size_row_k` |

### Model libraries

| File | Purpose |
|------|---------|
| `DiT_model_lib.py` | 1D DiT architecture (parity/discrete variant) |
| `diffusion_edm_lib.py` | EDM training loop, preconditioner wrappers (`EDMLoss`, `EDMDiTPrecondWrapper`) |
| `diffusion_nn_lib.py` | MLP / UNet-style score networks |
| `network_edm_lib.py` | Song/Dhariwal UNet (image) |

### Analysis libraries

| File | Purpose |
|------|---------|
| `basin_lib.py` / `basin_plot_lib.py` | Attractor basin measurement and visualization |
| `distance_analysis_lib.py` | Hamming distance at state transitions, transition matrices |
| `attention_analysis_lib.py` | Attention map entropy, spatial variance, head visualization |
| `vector_field_lib.py` | Evaluate and visualize diffusion score/denoiser vector fields on 2D planes |

---

## Key scripts (`scripts/`)

### Training CLIs

| Script | Purpose |
|--------|---------|
| `DiT_learn_parity_CLI.py` | Train DiT on group-parity task |
| `DiT_learn_exact_k_CLI.py` | Train DiT on exact-k task |
| `DiT_learn_latin_sq_CLI.py` | Train DiT on Latin square task |
| `DiT_learn_row_k_CLI.py` | Train DiT on row-k task |
| `GPT_learn_parity_CLI.py` | Train GPT on group-parity task |

### Analysis & plotting

| Script | Purpose |
|--------|---------|
| `extract_onset_table.py` | Batch extract rule/mem onset steps → `outputs/parity_onset_table.{csv,pkl}` |
| `plot_GPT_CE_analysis.py` | CE loss curves + per-position heatmaps for GPT runs; `plot_ce_figure()` importable |
| `plot_tb_curves.py` | General TensorBoard curve plotter (multi-run overlay) |
| `compare_DiT_GPT_parity.py` | Side-by-side DiT vs GPT onset comparison |
| `analyze_GPT_checkpoints_CE.py` | CE analysis from saved GPT checkpoints |
| `parity_memorization_eval_cli.py` | Offline memorization eval for parity samples |

---

## Onset detection convention

```
Rule onset : Sample_Accuracy  > acc_thresh  for n_consec=5 consecutive eval points
Mem onset  : Sample_Mem_Ratio > mem_thresh  for n_consec=5 consecutive eval points
Returns np.nan if never reached within the run.
```

Default thresholds: `acc_thresh=0.90`, `mem_thresh=0.10`.

**Stat mem frac** (data-adaptive baseline):
```python
support_size  = (2**(G-1)) ** (D // G)   # valid sequences (each group: even parity)
stat_mem_frac = N_train / support_size    # fraction of support covered by training data
adaptive_mem_thresh = 0.10 + stat_mem_frac
```

---

## Data format

Experiments are stored under `SAVEROOT`:
```
SAVEROOT = /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning
```

Each experiment folder follows the naming convention:
```
{arch}_{size}_parity_N{N}_D{D}_G{G}_even[_{suffix}]
```
e.g. `DiT_mini_parity_N4096_D36_G6_even`, `GPT_mini_parity_N4096_D36_G6_even_lr1e3`

Contains: `args.json`, `config.json`, `tensorboard/` (newer runs) or `mem_eval_stats.csv` (older DiT runs).

### Model scales

**DiT variants** (hidden_size × depth × heads):
- mini: 384 × 6 × 6
- B: 768 × 12 × 12

**GPT variants** (n_embd × n_layer × n_head, approx params):
- nano: 384 × 3 × 6, ~11M
- mini: 384 × 6 × 6, ~22M
- B: 768 × 12 × 12, ~86M

---

## Color conventions (publication figures)

| Split | Color | Style |
|-------|-------|-------|
| Train | `#2166ac` blue | solid |
| Valid-novel | `#d73027` red | solid |
| Boolean cube | `#555555` gray | dashed |
| DiT model | `#1b7837` green | — |
| GPT model | `#762a83` purple | — |

---

## Outputs

| File | Contents |
|------|---------|
| `outputs/parity_onset_table.csv` | 153 parity runs × onset steps at 5 mem + 3 acc thresholds |
| `outputs/parity_onset_table.pkl` | Same as pickle for faster loading |
| `figures/GPT_parity_learn_dissection/` | GPT CE analysis figures (PDF + PNG) |
