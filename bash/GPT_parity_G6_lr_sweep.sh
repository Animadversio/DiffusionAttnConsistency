#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -p kempner_h100
#SBATCH -c 16
#SBATCH --mem=75G
#SBATCH --gres=gpu:1
#SBATCH --array=1-6%6
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o logs/GPT_parity_G6_lr_sweep_%A_%a.out
#SBATCH -e logs/GPT_parity_G6_lr_sweep_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

# Learning rate sweep for GPT-mini parity G6 N=4096, 1e5 steps
#
# Baseline (existing): lr=1e-4, wd=0.01, batch=256, n_embd=384, n_layer=6, n_head=6
#
# Array index → (lr, exp_name):
#   1 → 1e-5   ..._lr1e5
#   2 → 3e-5   ..._lr3e5
#   3 → 1e-4   ..._lr1e4   (new rep of baseline)
#   4 → 3e-4   ..._lr3e4
#   5 → 1e-3   ..._lr1e3
#   6 → 3e-3   ..._lr3e3

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

param_list=\
'--lr 1e-5  --exp_name GPT_mini_parity_N4096_D36_G6_even_lr1e5
--lr 3e-5  --exp_name GPT_mini_parity_N4096_D36_G6_even_lr3e5
--lr 1e-4  --exp_name GPT_mini_parity_N4096_D36_G6_even_lr1e4
--lr 3e-4  --exp_name GPT_mini_parity_N4096_D36_G6_even_lr3e4
--lr 1e-3  --exp_name GPT_mini_parity_N4096_D36_G6_even_lr1e3
--lr 3e-3  --exp_name GPT_mini_parity_N4096_D36_G6_even_lr3e3
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "Running: $param_name"

# ── Environment setup ─────────────────────────────────────────────────────────
module load python
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
mamba activate torch2 2>/dev/null || true
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

# ── Training ──────────────────────────────────────────────────────────────────
cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
mkdir -p logs

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/GPT_learn_parity_CLI.py \
    --sample_num 4096  --sample_len 36 \
    --group_size 6     --parity 0 \
    --n_embd 384 --n_layer 6 --n_head 6 \
    --nsteps 100000    --batch_size 256 \
    --weight_decay 0.01 \
    --eval_sample_size 2048  --eval_batch_size 1024 \
    --temperature 1.0 \
    --record_frequency 0 \
    --save_ckpts --num_ckpts 40 \
    --use_tensorboard --tb_log_every 100 \
    $param_name
