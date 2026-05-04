#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH -c 16
#SBATCH --mem=75G
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o /n/home12/binxuwang/Github/DiffusionAttnConsistency/logs/basin_G3rep2_ep14251_119377_%j.out
#SBATCH -e /n/home12/binxuwang/Github/DiffusionAttnConsistency/logs/basin_G3rep2_ep14251_119377_%j.err
#SBATCH -J basin_G3rep2

module load python
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
mamba activate torch2 2>/dev/null || true
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

cd /n/home12/binxuwang/Github/DiffusionAttnConsistency

/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/measure_attractor_basin.py \
    --exp_name DiT_mini_parity_N4096_D36_G3_even_rep2 \
    --epochs 14251 119377 \
    --sigma 0.5 \
    --n_samples 30 \
    --n_points 150 \
    --device cuda \
    --verbose
