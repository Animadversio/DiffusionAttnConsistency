#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -p kempner_h100
#SBATCH -c 16
#SBATCH --mem=75G
#SBATCH --gres=gpu:1
#SBATCH --array=1-7%7
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o logs/GPT_parity_G6_wd_sweep_%A_%a.out
#SBATCH -e logs/GPT_parity_G6_wd_sweep_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

# Weight decay sweep for GPT-mini parity G6 N=4096, 1e5 steps
# lr fixed at 1e-4 (baseline value)
#
# Key question: does WD affect rule-learning vs. memorization dynamics?
# wd=0 is the critical control — does the baseline WD=0.01 actually matter?
#
# Array index → (weight_decay, exp_name):
#   1 → 0      ..._wd0
#   2 → 1e-3   ..._wd1e3
#   3 → 1e-2   ..._wd1e2   (re-run of baseline)
#   4 → 1e-1   ..._wd1e1
#   5 → 3e-1   ..._wd3e1
#   6 → 1.0    ..._wd1e0

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

param_list=\
'--weight_decay 0    --exp_name GPT_mini_parity_N4096_D36_G6_even_wd0
--weight_decay 1e-3  --exp_name GPT_mini_parity_N4096_D36_G6_even_wd1e3
--weight_decay 1e-2  --exp_name GPT_mini_parity_N4096_D36_G6_even_wd1e2
--weight_decay 5e-2  --exp_name GPT_mini_parity_N4096_D36_G6_even_wd5e2
--weight_decay 1e-1  --exp_name GPT_mini_parity_N4096_D36_G6_even_wd1e1
--weight_decay 3e-1  --exp_name GPT_mini_parity_N4096_D36_G6_even_wd3e1
--weight_decay 1.0   --exp_name GPT_mini_parity_N4096_D36_G6_even_wd1e0
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
    --lr 1e-4 \
    --eval_sample_size 2048  --eval_batch_size 1024 \
    --temperature 1.0 \
    --record_frequency 0 \
    --save_ckpts --num_ckpts 40 \
    --use_tensorboard --tb_log_every 100 \
    $param_name
