#!/bin/bash
#SBATCH -t 15:00:00
#SBATCH -p kempner_h100
#SBATCH -c 16
#SBATCH --mem=75G
#SBATCH --gres=gpu:1
#SBATCH --array=1-6%6
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o logs/DiT_parity_G6_wd_sweep_%A_%a.out
#SBATCH -e logs/DiT_parity_G6_wd_sweep_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

# Weight decay sweep for DiT-mini parity G6 N=4096 at lr=1e-4
# Baseline (rep2 / lr_sweep lr1e4): wd=0, Adam, no weight decay
#
# Sweep: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
#   wd=1e-6: essentially zero — sanity baseline with AdamW
#   wd=1e-5: very weak, negligible effect expected
#   wd=1e-4: weak regularization
#   wd=1e-3: moderate (common in NLP transformers)
#   wd=1e-2: strong — likely slows memorization
#   wd=5e-2: aggressive (ViT-scale) — may suppress memorization entirely
#
# Optimizer: AdamW (weight_decay > 0 → AdamW auto-selected in training lib)
# All other settings identical to rep2 (lr=1e-4, batch=256, 1M steps, G6, N=4096)
#
# Array index → (wd, exp_name):
#   1 → 1e-6  DiT_mini_parity_N4096_D36_G6_even_wd1e6
#   2 → 1e-5  DiT_mini_parity_N4096_D36_G6_even_wd1e5
#   3 → 1e-4  DiT_mini_parity_N4096_D36_G6_even_wd1e4
#   4 → 1e-3  DiT_mini_parity_N4096_D36_G6_even_wd1e3
#   5 → 1e-2  DiT_mini_parity_N4096_D36_G6_even_wd1e2
#   6 → 5e-2  DiT_mini_parity_N4096_D36_G6_even_wd5e2

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

param_list=\
'--weight_decay 1e-6  --exp_name DiT_mini_parity_N4096_D36_G6_even_wd1e6
--weight_decay 1e-5  --exp_name DiT_mini_parity_N4096_D36_G6_even_wd1e5
--weight_decay 1e-4  --exp_name DiT_mini_parity_N4096_D36_G6_even_wd1e4
--weight_decay 1e-3  --exp_name DiT_mini_parity_N4096_D36_G6_even_wd1e3
--weight_decay 1e-2  --exp_name DiT_mini_parity_N4096_D36_G6_even_wd1e2
--weight_decay 5e-2  --exp_name DiT_mini_parity_N4096_D36_G6_even_wd5e2
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

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/DiT_learn_parity_CLI.py \
    --sample_num 4096  --sample_len 36 \
    --group_size 6     --parity 0 \
    --patch_size 1     --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 \
    --nsteps 1000000   --batch_size 256  --lr 1e-4 \
    --eval_sample_size 2048  --eval_batch_size 512 \
    --eval_sampling_steps 35 --eval_fix_noise_seed \
    --record_frequency 0 \
    --save_ckpts --num_ckpts 40 \
    --use_tensorboard \
    $param_name
