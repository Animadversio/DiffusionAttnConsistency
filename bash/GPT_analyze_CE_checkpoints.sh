#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -p kempner_h100
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --array=1-2%2
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o logs/GPT_analyze_CE_%A_%a.out
#SBATCH -e logs/GPT_analyze_CE_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

# Analyze GPT mini N4096 G6 baseline checkpoints:
# compute per-split and per-position CE loss at every saved checkpoint.
#
# Array:
#   1 → GPT_mini_parity_N4096_D36_G6_even_lr1e4  (LR-sweep baseline rep)
#   2 → GPT_mini_parity_N4096_D36_G6_even_wd1e2  (WD-sweep baseline rep)

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

param_list=\
'--exp_names GPT_mini_parity_N4096_D36_G6_even_lr1e4
--exp_names GPT_mini_parity_N4096_D36_G6_even_wd1e2
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "Running: $param_name"

# ── Environment ───────────────────────────────────────────────────────────────
module load python
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

# ── Run ───────────────────────────────────────────────────────────────────────
cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
mkdir -p logs

/n/home12/binxuwang/.conda/envs/torch2/bin/python \
    scripts/analyze_GPT_checkpoints_CE.py \
    $param_name \
    --n_eval 4096 \
    --batch_size 512 \
    --device cuda
