#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --array=1-8%8
#SBATCH --account=binxuwang
#SBATCH -o analyze_evolution_parity_rep2_%A_%a.out
#SBATCH -e analyze_evolution_parity_rep2_%A_%a.err

echo "Task: $SLURM_ARRAY_TASK_ID"

param_list=\
'DiT_mini_parity_N4096_D36_G2_even_rep2
DiT_mini_parity_N4096_D36_G3_even_rep2
DiT_mini_parity_N4096_D36_G4_even_rep2
DiT_mini_parity_N4096_D36_G6_even_rep2
DiT_mini_parity_N4096_D36_G9_even_rep2
DiT_mini_parity_N4096_D36_G12_even_rep2
DiT_mini_parity_N4096_D36_G18_even_rep2
DiT_mini_parity_N4096_D36_G36_even_rep2'

exp_name=$(echo "$param_list" | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "exp_name=$exp_name"

unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python

cd /n/home12/binxuwang/Github/DiffusionAttnConsistency

# Step 1: compute evolution metrics
$PYTHON scripts/analyze_sample_evolution.py --exp_name "$exp_name"

# Step 2: generate raster + overview + transition plots
$PYTHON scripts/plot_sample_evolution.py --exp_name "$exp_name"

echo "Done: $exp_name"
