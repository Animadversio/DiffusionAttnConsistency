
module load python
conda deactivate
mamba activate torch2 
which python
which python3
# run code
cd /n/home12/binxuwang/Github/DiffusionAttnConsistency
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
python scripts/train_flower_latents.py --data-path ~/Datasets --image-size 16 \
    --global-seed 42 --num-workers 8 --log-every 100 \
    --global-batch-size 256 \
    --ckpt-every 10000 --save-samples-every 1000 \
    --num_eval_sample 256 --eval_sampler ddim100 \
    --model DiT-S/2 --num-classes 0  --epochs 2000 \
    --dataset flower_latents10k_pilots --expname flower_latents10k_pilots


torchrun --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
   scripts/train_flower_latents.py --data-path ~/Datasets --image-size 16 \
    --global-seed 42 --num-workers 8 --log-every 100 \
    --global-batch-size 256 \
    --ckpt-every 10000 --save-samples-every 1000 \
    --num_eval_sample 256 --eval_sampler ddim100 \
    --model DiT-S/1 --num-classes 0  --epochs 2000 \
    --dataset flower_latents10k_pilots --expname flower_latents10k_pilots


torchrun --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
   scripts/train_flower_latents.py --data-path ~/Datasets --image-size 16 \
    --global-seed 42 --num-workers 8 --log-every 100 \
    --global-batch-size 256 \
    --ckpt-every 10000 --save-samples-every 1000 \
    --num_eval_sample 256 --eval_sampler ddim100 \
    --model DiT-mini/1 --num-classes 0  --epochs 2000 \
    --dataset flower_latents10k_pilots --expname flower_latents10k_pilots

  
torchrun --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
   scripts/train_flower_latents.py --data-path ~/Datasets --image-size 16 \
    --global-seed 42 --num-workers 8 --log-every 100 \
    --global-batch-size 256 \
    --ckpt-every 10000 --save-samples-every 1000 \
    --num_eval_sample 256 --eval_sampler ddim100 \
    --model DiT-micro/1 --num-classes 0  --epochs 2000 \
    --dataset flower_latents10k_pilots --expname flower_latents10k_pilots


torchrun --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
   scripts/train_flower_latents.py --data-path ~/Datasets --image-size 16 \
    --global-seed 42 --num-workers 8 --log-every 100 \
    --global-batch-size 256 \
    --ckpt-every 10000 --save-samples-every 1000 \
    --num_eval_sample 256 --eval_sampler ddim100 \
    --model DiT-nano/1 --num-classes 0  --epochs 2000 \
    --dataset flower_latents10k_pilots --expname flower_latents10k_pilots

