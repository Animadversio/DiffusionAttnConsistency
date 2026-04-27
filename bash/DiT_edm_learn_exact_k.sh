#!/bin/bash
#SBATCH -t 15:00:00          # Runtime in D-HH:MM
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16                # Number of cores
#SBATCH --mem=75G            # Memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH --array=1-7%7
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o DiT_edm_learn_exact_k_%A_%a.out
#SBATCH -e DiT_edm_learn_exact_k_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--sample_num 4096  --sample_len 36 --k_ones 3   --exp_name DiT_mini_exactK_N4096_D36_K3    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --k_ones 4   --exp_name DiT_mini_exactK_N4096_D36_K4    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --k_ones 6   --exp_name DiT_mini_exactK_N4096_D36_K6    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --k_ones 8   --exp_name DiT_mini_exactK_N4096_D36_K8    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --k_ones 9   --exp_name DiT_mini_exactK_N4096_D36_K9    --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --k_ones 12  --exp_name DiT_mini_exactK_N4096_D36_K12   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --k_ones 18  --exp_name DiT_mini_exactK_N4096_D36_K18   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
unset CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX
mamba deactivate 2>/dev/null || true
mamba activate torch2 2>/dev/null || true
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

# run code
cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/DiT_learn_exact_k_CLI.py --record_frequency 0 --eval_sampling_steps 35 --eval_fix_noise_seed \
    --eval_sample_size 2048 --eval_batch_size 512  --lr 1e-4 --batch_size 256  \
    $param_name
