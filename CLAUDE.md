# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository studies attention consistency, memorization, and rule-learning behavior in diffusion models — primarily DiT (Diffusion Transformers) trained on synthetic structured datasets. The original focus was parity functions; the project has since expanded to a family of "rule-learning" benchmarks (exact-K counts, Latin squares, Sudoku, row-K binary grid rules) that span a spectrum of constraint complexity, from degree-1 global sums to high-degree nonlinear products. A GPT baseline on parity is also included for comparison with autoregressive models.

## Core Architecture

### Main Components
- **core/**: Core library modules
  - `DiT_model_lib.py`: DiT (Diffusion Transformer) model implementations
  - `diffusion_edm_lib.py`: EDM (Elucidating Diffusion Models) framework
  - `diffusion_esm_edm_lib.py`: Extended EDM with explicit score matching (`EDMDeltaGMMScoreLoss`)
  - `diffusion_basics_lib.py`: Basic diffusion utilities
  - `diffusion_nn_lib.py`: NN backbones (e.g. `UNetBlockStyleMLP_backbone_NoFirstNorm`) used as alternatives to DiT
  - `network_edm_lib.py`: Neural network components for EDM
  - `attention_analysis_lib.py`: Attention map analysis utilities
  - `parity_lib.py`: Parity rule — group-G parity on ±1 bit vectors
  - `exact_k_lib.py`: Exact-K rule — vectors with exactly K active (+1) entries
  - `latin_square_lib.py`: Latin square rule — n×n grids whose rows and cols are permutations
  - `row_k_lib.py`: Row-K binary grid rules — `row_k`, `row_variable_k`, `global_k` over n×n ±1 grids

### Scripts
- **scripts/**: Executable training, evaluation, and analysis scripts
  - Training CLIs (one per rule):
    - `DiT_learn_parity_CLI.py` — DiT on group-G parity
    - `DiT_learn_exact_k_CLI.py` — DiT on exact-K
    - `DiT_learn_latin_sq_CLI.py` — DiT on Latin squares (scalar / one-hot encodings)
    - `DiT_learn_row_k_CLI.py` — DiT on row-K binary grid rules
    - `GPT_learn_parity_CLI.py` — GPT-2 baseline on parity
    - `train_flower.py`, `train_flower_latents.py` — image-domain (flower) DiT training
  - Evaluation CLIs:
    - `parity_memorization_eval_cli.py`, `exact_k_memorization_eval_cli.py`
  - Analysis / plotting:
    - `analyze_error_cell_confidence.py`, `analyze_latinsq_rowcol.py`,
      `analyze_onehot_margin.py`, `analyze_onehot_violations.py`,
      `analyze_sample_errors.py`, `analyze_scalar_quantization.py`,
      `training_data_nn_analysis.py`
    - `plot_onehot_margin_pubfig.py`, `plot_row_k_crash.py`, `plot_tb_curves.py`
    - `visualize_error_samples.py`, `visualize_rule_variant_boards.py`
  - `attn_map_massprod_script.py`: Batch attention map generation

### Bash launchers (SLURM)
- **bash/**: SLURM array-job launchers, one per rule family
  - `DiT_edm_learn_parity.sh`, `DiT_edm_learn_exact_k.sh`,
    `DiT_edm_learn_latin_sq.sh`, `DiT_edm_learn_row_k.sh`,
    `GPT_learn_parity.sh`, `DiT_train_flower.sh`

### Docs
- **docs/**: Method writeups and benchmark notes
  - `experiment_table.md` — consolidated results table across all rules (validity / memorization)
  - `latin_square_benchmark.md` — Latin square benchmark design
  - `validity_crash_analysis.md` — late-training validity degradation (memorization vs confident-wrong)
  - `errcell_method.md` / `errcell_method.tex` — per-cell error / confidence analysis method

### Notebooks
- **notebooks/**: Analysis and experimentation notebooks
  - Parity: `20250731_parity_diffusion_learning.ipynb`, `20250731_parity_func_eval.ipynb`,
    `20250731_parity_func_sampling.ipynb`, `20250804_parity_sample_memorization_eval.ipynb`,
    `20250807_parity_cross_run_synopsis_eval.ipynb`,
    `20250826_parity_DiT_eval_export.ipynb`, `20250826_parity_GPT_eval.ipynb`,
    `20250829_parity_train_dynam_export.ipynb`
  - Energy / sampling: `20250803_prod_energy_func.ipynb`,
    `multipoint_energy_sampling.ipynb`, `pyro_energy_sampling.ipynb`
  - GPT baseline: `20250801_parity_GPT_train.ipynb`
  - Attention: `attn_head_structure.ipynb`, `DiT_attn_visualize.ipynb`,
    `DiT_attn_visualize_massprod.ipynb`
  - Image domain: `flower_dataset_gen.ipynb`

### Tables
- **tables/**: Aggregated CSVs (e.g. `parity_near_neighbor_stats_df.csv`)

## Common Development Commands

### Training Models
```bash
# SLURM array jobs (one per rule family)
sbatch bash/DiT_edm_learn_parity.sh
sbatch bash/DiT_edm_learn_exact_k.sh
sbatch bash/DiT_edm_learn_latin_sq.sh
sbatch bash/DiT_edm_learn_row_k.sh
sbatch bash/GPT_learn_parity.sh

# Direct training (parity)
python scripts/DiT_learn_parity_CLI.py --record_frequency 0 --eval_sampling_steps 35 \
    --eval_fix_noise_seed --eval_sample_size 2048 --eval_batch_size 512 \
    --lr 1e-4 --batch_size 256 --sample_num 4096 --sample_len 36 --group_size 12 \
    --parity 0 --exp_name DiT_mini_parity_N4096_D36_G12_even \
    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 \
    --nsteps 1000000

# Direct training (row-K binary grid rules)
python scripts/DiT_learn_row_k_CLI.py --sample_num 4096 --n_size 6 \
    --rule row_variable_k --K_list 3 4 \
    --exp_name DiT_mini_rowVarK34_n6_N4096 \
    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 \
    --nsteps 1000000 --save_ckpts --num_ckpts 40

# Train on flower dataset
bash bash/DiT_train_flower.sh
```

### Running Notebooks
```bash
# Environment setup (typically done on compute nodes)
module load python
mamba activate torch2

# Launch Jupyter
jupyter notebook
```

### Analysis and Evaluation
```bash
# Generate attention maps
python scripts/attn_map_massprod_script.py

# Memorization evaluation
python scripts/parity_memorization_eval_cli.py     # parity
python scripts/exact_k_memorization_eval_cli.py    # exact-K

# Late-training crash / row-K plots
python scripts/plot_row_k_crash.py
python scripts/plot_tb_curves.py

# One-hot margin / error-cell analyses (Latin square)
python scripts/analyze_onehot_margin.py
python scripts/analyze_error_cell_confidence.py
python scripts/analyze_latinsq_rowcol.py
```

## Key Experiment Patterns

### Rule Families and Difficulty Spectrum
Constraints are ordered roughly by score-function complexity:

| Family       | Constraint                                    | Score degree     |
|--------------|-----------------------------------------------|------------------|
| Exact-K      | `sum(x) == 2K - D` (one global sum)           | degree 1         |
| Row-K        | per-row count constraints on n×n ±1 grid      | degree 1 per row |
| Latin square | row + col all-distinct on n×n grid            | intermediate     |
| Sudoku 6×6   | row + col + 2×3 block all-distinct            | intermediate     |
| Parity (G)   | nonlinear degree-G product per group of G bits| degree G         |

See `docs/experiment_table.md` for the consolidated validity / memorization table across families.

### Model Naming Convention
- Parity: `DiT_{size}_parity_N{samples}_D{dims}_G{group_size}_{parity_type}`
- Other rules: `DiT_{size}_{rule}_n{n}_N{samples}[_{encoding}]`,
  e.g. `DiT_mini_latinSq_n6_N4096_onehot_zeromean_autoSD`,
  `DiT_mini_rowVarK0246_n6_N4096`
- Sizes: `nano` (3L), `mini` (6L 6H 384D), `S` (12L), `B` (12L 12H 768D)

### Encodings (Latin square / row-K)
- **scalar**: cell value normalized to [-1, +1] (gap 2/(n-1))
- **one-hot** variants: `{-1,+1}` (pm1), `{0,1}` (zero_one), zero-mean — see `docs/errcell_method.md`
- Scalar tends to memorize more; one-hot avoids memorization but fails via confident-wrong cells.

### Evaluation Metrics
- **Rule validity** (per-rule check, `*_check` functions in each `*_lib.py`)
- **Memorization ratio**: fraction of generated valid samples already in the training set
- **Per-row / per-column / per-group** correctness (sub-rule diagnostics)
- **Attention patterns**: transformer attention map analyses
- **Late-training crash** (validity decay): see `docs/validity_crash_analysis.md`

## Environment and Dependencies

### Python Environment
- PyTorch 2.x, transformers (for GPT baseline)
- Standard ML libraries: numpy, matplotlib, seaborn, pandas
- Custom utilities: `circuit_toolkit`, `easydict`
- Diffusion-specific: custom EDM implementation in `core/`

### Compute Environment
- SLURM-based HPC cluster (Kempner H100 partition)
- Conda/Mamba environment `torch2`
- Each bash launcher unsets `CONDA_PREFIX` / `CONDA_DEFAULT_ENV` and uses the absolute python path
  `/n/home12/binxuwang/.conda/envs/torch2/bin/python`

## File Organization Patterns

### Experiment Outputs
```
/n/holylfs06/.../DiffusionParityLearning/{exp_name}/
├── args.json             # Experiment arguments
├── config.json           # Model configuration
├── training_data_tsr.pt  # Training data tensor
├── samples/              # Generated samples by epoch
├── ckpts/                # Model checkpoints (with --save_ckpts; --num_ckpts geomspaced)
├── tb/                   # TensorBoard logs (per-callback metrics, grad_norm, etc.)
└── *.csv                 # Evaluation statistics
```

### Generated Figures
```
figures/
├── {exp_name}_sample_parity_eval.pdf/png
├── {exp_name}_sample_parity_memorization_eval.pdf/png
├── {exp_name}_row_k_crash.pdf/png
└── evaluation summary files
```
