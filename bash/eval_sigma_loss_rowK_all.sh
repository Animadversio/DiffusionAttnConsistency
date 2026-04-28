#!/bin/bash
#SBATCH -t 2:00:00           # 39 ckpts × ~70s = ~45 min, 2h is safe
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=1-7%7
#SBATCH -o eval_sigma_loss_rowK_all_%A_%a.out
#SBATCH -e eval_sigma_loss_rowK_all_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "Job ID: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"

# load modules
module load python
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
mamba activate torch2 2>/dev/null || true
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
COMMON="--ckpts all --n_sigma 50 --sigma_min 0.002 --sigma_max 80 --noise_reps 1 --n_test 4096 --batch_size 512 --device cuda --splits train test random"

case $SLURM_ARRAY_TASK_ID in
1)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_rowK3_n6_N4096       --rule row_k          --K 3            --n_size 6 $COMMON ;;
2)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_rowVarK15_n6_N4096   --rule row_variable_k --K_list 1 5     --n_size 6 $COMMON ;;
3)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_rowVarK34_n6_N4096   --rule row_variable_k --K_list 3 4     --n_size 6 $COMMON ;;
4)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_rowVarK0246_n6_N4096 --rule row_variable_k --K_list 0 2 4 6 --n_size 6 $COMMON ;;
5)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_rowVarK3456_n6_N4096 --rule row_variable_k --K_list 3 4 5 6 --n_size 6 $COMMON ;;
6)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_globalK15_n6_N4096   --rule global_k       --K_list 1 5     --n_size 6 $COMMON ;;
7)  $PYTHON scripts/eval_sigma_loss_CLI.py --exp_name DiT_mini_globalK24_n6_N4096   --rule global_k       --K_list 2 4     --n_size 6 $COMMON ;;
esac
