#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -p kempner_h100
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --array=1-8%8
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o eval_sigma_loss_parity_rep2_%A_%a.out
#SBATCH -e eval_sigma_loss_parity_rep2_%A_%a.err

echo "Task: $SLURM_ARRAY_TASK_ID"

# param_list: one entry per line (1-indexed via sed)
# Format: exp_name  group_size  parity_val
param_list=\
'DiT_mini_parity_N4096_D36_G2_even_rep2  2  0
DiT_mini_parity_N4096_D36_G3_even_rep2  3  0
DiT_mini_parity_N4096_D36_G4_even_rep2  4  0
DiT_mini_parity_N4096_D36_G6_even_rep2  6  0
DiT_mini_parity_N4096_D36_G9_even_rep2  9  0
DiT_mini_parity_N4096_D36_G12_even_rep2 12 0
DiT_mini_parity_N4096_D36_G18_even_rep2 18 0
DiT_mini_parity_N4096_D36_G36_even_rep2 36 0'

line=$(echo "$param_list" | sed -n "${SLURM_ARRAY_TASK_ID}p")
exp_name=$(echo "$line" | awk '{print $1}')
group_size=$(echo "$line" | awk '{print $2}')
parity_val=$(echo "$line" | awk '{print $3}')

echo "exp_name=$exp_name  group_size=$group_size  parity_val=$parity_val"

# Fix conda environment contamination
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
mamba activate torch2 2>/dev/null || true

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
SCRIPT_DIR=/n/home12/binxuwang/Github/DiffusionAttnConsistency

cd $SCRIPT_DIR

$PYTHON scripts/eval_sigma_loss_CLI.py \
    --exp_name "$exp_name" \
    --rule parity \
    --group_size "$group_size" \
    --parity_val "$parity_val" \
    --ckpts all \
    --n_sigma 50 \
    --batch_size 1024 \
    --noise_reps 2 \
    --device cuda
