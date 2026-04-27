#!/bin/bash
#SBATCH -t 1:00:00           # 24 ckpts × ~70s = ~28 min
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o eval_sigma_loss_rowK2_early_%j.out
#SBATCH -e eval_sigma_loss_rowK2_early_%j.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "Job ID: $SLURM_JOB_ID"

module load python
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
mamba activate torch2 2>/dev/null || true
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/eval_sigma_loss_CLI.py \
    --exp_name DiT_mini_rowK2_n6_N4096 \
    --rule row_k --K 2 --n_size 6 \
    --ckpts 0 1 2 4 5 8 11 17 24 34 49 70 100 142 203 289 412 587 837 1701 2424 3455 4923 7017 \
    --n_sigma 50 --sigma_min 0.002 --sigma_max 80 \
    --noise_reps 1 --n_test 4096 \
    --batch_size 512 --device cuda \
    --splits train test random
