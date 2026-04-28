#!/bin/bash
#SBATCH -t 15:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array=87-94%8
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o DiT_edm_learn_parity_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e DiT_edm_learn_parity_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_mini_parity_N4096_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_mini_parity_N4096_D36_G18_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_mini_parity_N4096_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_S_parity_N4096_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_S_parity_N4096_D36_G18_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_S_parity_N4096_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_S_parity_N4096_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_S_parity_N4096_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_S_parity_N4096_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_S_parity_N4096_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_S_parity_N4096_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_nano_parity_N4096_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_nano_parity_N4096_D36_G18_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_nano_parity_N4096_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_nano_parity_N4096_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_nano_parity_N4096_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_nano_parity_N4096_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_nano_parity_N4096_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_nano_parity_N4096_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 3 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_B_parity_N4096_D36_G36_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_B_parity_N4096_D36_G18_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_B_parity_N4096_D36_G12_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_B_parity_N4096_D36_G9_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_B_parity_N4096_D36_G6_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_B_parity_N4096_D36_G4_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_B_parity_N4096_D36_G3_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_B_parity_N4096_D36_G2_even   --patch_size 1 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_mini_parity_N16384_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_mini_parity_N16384_D36_G18_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_mini_parity_N16384_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_mini_parity_N16384_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_mini_parity_N16384_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_mini_parity_N16384_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_mini_parity_N16384_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 16384  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_mini_parity_N16384_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_mini_parity_N65536_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_mini_parity_N65536_D36_G18_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_mini_parity_N65536_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_mini_parity_N65536_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_mini_parity_N65536_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_mini_parity_N65536_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_mini_parity_N65536_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 65536  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_mini_parity_N65536_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 4  --parity 0  --exp_name DiT_mini_parity_N8192_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 6  --parity 0  --exp_name DiT_mini_parity_N8192_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 12  --parity 0 --exp_name DiT_mini_parity_N8192_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 36  --parity 0 --exp_name DiT_mini_parity_N8192_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 4  --parity 0  --exp_name DiT_mini_parity_N2048_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 6  --parity 0  --exp_name DiT_mini_parity_N2048_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 12  --parity 0 --exp_name DiT_mini_parity_N2048_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 36  --parity 0 --exp_name DiT_mini_parity_N2048_D36_G36_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 4  --parity 0  --exp_name DiT_mini_parity_N1024_D36_G4_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 6  --parity 0  --exp_name DiT_mini_parity_N1024_D36_G6_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 12  --parity 0 --exp_name DiT_mini_parity_N1024_D36_G12_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 36  --parity 0 --exp_name DiT_mini_parity_N1024_D36_G36_even  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 2  --parity 0  --exp_name DiT_mini_parity_N8192_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 3  --parity 0  --exp_name DiT_mini_parity_N8192_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 9  --parity 0  --exp_name DiT_mini_parity_N8192_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 8192  --sample_len 36 --group_size 18  --parity 0 --exp_name DiT_mini_parity_N8192_D36_G18_even  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 2  --parity 0  --exp_name DiT_mini_parity_N2048_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 3  --parity 0  --exp_name DiT_mini_parity_N2048_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 9  --parity 0  --exp_name DiT_mini_parity_N2048_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 2048  --sample_len 36 --group_size 18  --parity 0 --exp_name DiT_mini_parity_N2048_D36_G18_even  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 2  --parity 0  --exp_name DiT_mini_parity_N1024_D36_G2_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 3  --parity 0  --exp_name DiT_mini_parity_N1024_D36_G3_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 
--sample_num 1024  --sample_len 36 --group_size 9  --parity 0  --exp_name DiT_mini_parity_N1024_D36_G9_even   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 1024  --sample_len 36 --group_size 18  --parity 0 --exp_name DiT_mini_parity_N1024_D36_G18_even  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000
--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G2_even_rep2   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G3_even_rep2   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G4_even_rep2   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G6_even_rep2   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name DiT_mini_parity_N4096_D36_G9_even_rep2   --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name DiT_mini_parity_N4096_D36_G12_even_rep2  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name DiT_mini_parity_N4096_D36_G18_even_rep2  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name DiT_mini_parity_N4096_D36_G36_even_rep2  --patch_size 1 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4  --nsteps 1000000 --save_ckpts --num_ckpts 40 --use_tensorboard
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
mamba deactivate
mamba activate torch2
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

# run code
cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
python scripts/DiT_learn_parity_CLI.py --record_frequency 0 --eval_sampling_steps 35 --eval_fix_noise_seed \
    --eval_sample_size 2048 --eval_batch_size 512  --lr 1e-4 --batch_size 256  \
    $param_name

