#!/bin/bash
#SBATCH -t 3-00:00:00        # Runtime in D-HH:MM (3 days for DiT-B jobs)
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16                # Number of cores
#SBATCH --mem=100G           # Memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH --array=11-14%4      # Jobs 1-4 = DiT-mini (done); 5-8 = DiT-B (running); 9-10 = encoding ablations; 11-14 = rule variants
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o DiT_edm_learn_latin_sq_%A_%a.out
#SBATCH -e DiT_edm_learn_latin_sq_%A_%a.err
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--sample_num 4096  --n_size 5  --encoding scalar  --exp_name DiT_mini_latinSq_n5_N4096_scalar  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 5  --encoding onehot  --exp_name DiT_mini_latinSq_n5_N4096_onehot  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --encoding scalar  --exp_name DiT_mini_latinSq_n6_N4096_scalar  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --encoding onehot  --exp_name DiT_mini_latinSq_n6_N4096_onehot  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 5  --encoding scalar  --exp_name DiT_B_latinSq_n5_N4096_scalar     --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 5  --encoding onehot  --exp_name DiT_B_latinSq_n5_N4096_onehot    --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --encoding scalar  --exp_name DiT_B_latinSq_n6_N4096_scalar     --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --encoding onehot  --exp_name DiT_B_latinSq_n6_N4096_onehot    --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --encoding onehot  --onehot_type zero_mean  --sigma_data auto  --exp_name DiT_mini_latinSq_n6_N4096_onehot_zeromean_autoSD  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --encoding onehot  --onehot_type zero_one   --sigma_data auto  --exp_name DiT_mini_latinSq_n6_N4096_onehot_zeroone_autoSD   --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --rule row_only  --encoding scalar  --exp_name DiT_mini_rowOnly_n6_N4096_scalar  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --rule row_only  --encoding onehot  --exp_name DiT_mini_rowOnly_n6_N4096_onehot  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --rule sudoku    --encoding scalar  --block_h 2  --block_w 3  --exp_name DiT_mini_sudoku6x6_N4096_scalar  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --n_size 6  --rule sudoku    --encoding onehot  --block_h 2  --block_w 3  --exp_name DiT_mini_sudoku6x6_N4096_onehot  --patch_size 1 --hidden_size 384 --depth 6  --num_heads 6  --mlp_ratio 4  --nsteps 1000000
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
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/DiT_learn_latin_sq_CLI.py \
    --record_frequency 0 --eval_sampling_steps 35 --eval_fix_noise_seed \
    --eval_sample_size 2048 --eval_batch_size 512  --lr 1e-4 --batch_size 256 \
    --snap_eps 0.15 \
    $param_name
