#!/bin/bash
#SBATCH -t 10:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to  
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array=17-32%12
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o GPT_learn_parity_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e GPT_learn_parity_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name GPT_mini_parity_N4096_D36_G2_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name GPT_mini_parity_N4096_D36_G3_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name GPT_mini_parity_N4096_D36_G4_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name GPT_mini_parity_N4096_D36_G6_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name GPT_mini_parity_N4096_D36_G9_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name GPT_mini_parity_N4096_D36_G12_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name GPT_mini_parity_N4096_D36_G18_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name GPT_mini_parity_N4096_D36_G36_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G2_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G3_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G4_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G6_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G9_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name GPT_nano_parity_N4096_D36_G12_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name GPT_nano_parity_N4096_D36_G18_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name GPT_nano_parity_N4096_D36_G36_even   --n_embd 384 --n_layer 3 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 2  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G2_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 3  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G3_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 4  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G4_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 6  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G6_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 9  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G9_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 12 --parity 0 --exp_name GPT_mini_parity_N8192_D36_G12_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 18 --parity 0 --exp_name GPT_mini_parity_N8192_D36_G18_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 8192  --sample_len 36 --group_size 36 --parity 0 --exp_name GPT_mini_parity_N8192_D36_G36_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 2  --parity 0 --exp_name GPT_mini_parity_N16384_D36_G2_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 3  --parity 0 --exp_name GPT_mini_parity_N16384_D36_G3_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 4  --parity 0 --exp_name GPT_mini_parity_N16384_D36_G4_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 6  --parity 0 --exp_name GPT_mini_parity_N16384_D36_G6_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 9  --parity 0 --exp_name GPT_mini_parity_N16384_D36_G9_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 12 --parity 0 --exp_name GPT_mini_parity_N16384_D36_G12_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 18 --parity 0 --exp_name GPT_mini_parity_N16384_D36_G18_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
--sample_num 16384 --sample_len 36 --group_size 36 --parity 0 --exp_name GPT_mini_parity_N16384_D36_G36_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
'
# --sample_num 8192  --sample_len 36 --group_size 2  --parity 0  --exp_name GPT_mini_parity_N8192_D36_G2_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 3  --parity 0  --exp_name GPT_mini_parity_N8192_D36_G3_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 4  --parity 0  --exp_name GPT_mini_parity_N8192_D36_G4_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 6  --parity 0  --exp_name GPT_mini_parity_N8192_D36_G6_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 9  --parity 0  --exp_name GPT_mini_parity_N8192_D36_G9_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 18  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G18_even  --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 12  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G12_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 8192  --sample_len 36 --group_size 36  --parity 0 --exp_name GPT_mini_parity_N8192_D36_G36_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 2  --parity 0  --exp_name GPT_mini_parity_N2048_D36_G2_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 3  --parity 0  --exp_name GPT_mini_parity_N2048_D36_G3_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 4  --parity 0  --exp_name GPT_mini_parity_N2048_D36_G4_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 6  --parity 0  --exp_name GPT_mini_parity_N2048_D36_G6_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 9  --parity 0  --exp_name GPT_mini_parity_N2048_D36_G9_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 18  --parity 0 --exp_name GPT_mini_parity_N2048_D36_G18_even  --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 12  --parity 0 --exp_name GPT_mini_parity_N2048_D36_G12_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 2048  --sample_len 36 --group_size 36  --parity 0 --exp_name GPT_mini_parity_N2048_D36_G36_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 2  --parity 0  --exp_name GPT_mini_parity_N1024_D36_G2_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 3  --parity 0  --exp_name GPT_mini_parity_N1024_D36_G3_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 4  --parity 0  --exp_name GPT_mini_parity_N1024_D36_G4_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 6  --parity 0  --exp_name GPT_mini_parity_N1024_D36_G6_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 9  --parity 0  --exp_name GPT_mini_parity_N1024_D36_G9_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 18  --parity 0 --exp_name GPT_mini_parity_N1024_D36_G18_even  --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 12  --parity 0 --exp_name GPT_mini_parity_N1024_D36_G12_even   --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 1024  --sample_len 36 --group_size 36  --parity 0 --exp_name GPT_mini_parity_N1024_D36_G36_even  --n_embd 384 --n_layer 6 --n_head 6 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name GPT_S_parity_N4096_D36_G36_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name GPT_S_parity_N4096_D36_G18_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name GPT_S_parity_N4096_D36_G12_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name GPT_S_parity_N4096_D36_G9_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name GPT_S_parity_N4096_D36_G6_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name GPT_S_parity_N4096_D36_G4_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name GPT_S_parity_N4096_D36_G3_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name GPT_S_parity_N4096_D36_G2_even   --n_embd 512 --n_layer 12 --n_head 8 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 36 --parity 0 --exp_name GPT_nano_parity_N4096_D36_G36_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 18 --parity 0 --exp_name GPT_nano_parity_N4096_D36_G18_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 12 --parity 0 --exp_name GPT_nano_parity_N4096_D36_G12_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 9  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G9_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 6  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G6_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 4  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G4_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 3  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G3_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 
# --sample_num 4096  --sample_len 36 --group_size 2  --parity 0 --exp_name GPT_nano_parity_N4096_D36_G2_even   --n_embd 256 --n_layer 3 --n_head 4 --nsteps 100000 

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
python scripts/GPT_learn_parity_CLI.py --record_frequency 0 \
    --lr 1e-4 --batch_size 256 --weight_decay 0.01 \
    --eval_sample_size 2048 --eval_batch_size 1024 --temperature 1.0 --use_tensorboard --tb_log_every 100 \
    $param_name