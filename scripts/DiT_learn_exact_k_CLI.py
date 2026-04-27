
# %%
import sys
import os
from os.path import join
import json
import pickle as pkl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone_NoFirstNorm
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
from core.diffusion_esm_edm_lib import EDMDeltaGMMScoreLoss
from core.DiT_model_lib import *
from core.exact_k_lib import sample_exact_k_dataset, exact_k_check, ones_count, round_to_pos_neg_one
from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid
from torch.utils.tensorboard import SummaryWriter


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# %%
from pprint import pprint
import argparse
from typing import List, Tuple
def parse_range(range_str: List[str]) -> Tuple[int, int, int]:
    if len(range_str) != 3:
        raise argparse.ArgumentTypeError("Each range must have exactly three integers: start end step.")
    try:
        start, end, step = map(int, range_str)
    except ValueError:
        raise argparse.ArgumentTypeError("All range values must be integers.")
    if start >= end:
        raise argparse.ArgumentTypeError(f"Start ({start}) must be less than end ({end}).")
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Step ({step}) must be a positive integer.")
    return (start, end, step)


def generate_record_times(ranges: List[Tuple[int, int, int]]) -> List[int]:
    record_times = []
    for start, end, step in ranges:
        record_times.extend(range(start, end, step))
    return record_times


def generate_ckpt_step_list(max_steps, num_ckpts=100, sequence="geomspace") -> List[int]:
    if sequence == "geomspace":
        ckpt_step_list = np.geomspace(1, max_steps+1, num_ckpts).astype(int)
        ckpt_step_list = np.unique(ckpt_step_list)
        ckpt_step_list = ckpt_step_list[ckpt_step_list <= max_steps]
    elif sequence == "linspace":
        ckpt_step_list = np.linspace(1, max_steps, num_ckpts).astype(int)
        ckpt_step_list = np.unique(ckpt_step_list)
        ckpt_step_list = ckpt_step_list[ckpt_step_list <= max_steps]
    else:
        raise ValueError(f"Invalid sequence type: {sequence}")
    return ckpt_step_list


def parse_args():
    parser = argparse.ArgumentParser(description="DiT Exact-K Learning Experiment")
    parser.add_argument("--exp_name", type=str, default="DiT_mini_exactK_N4096_D36_K18", help="Experiment name")
    parser.add_argument("--loss_type", type=str, default="DSM", help="Loss type (DSM, ESM)")
    # training hyper-parameters
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--nsteps", type=int, default=5000, help="Number of steps")
    # model hyper-parameters
    parser.add_argument("--patch_size", type=int, default=1, help="Patch size")
    parser.add_argument("--hidden_size", type=int, default=384, help="Hidden size")
    parser.add_argument("--depth", type=int, default=6, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1, help="Class dropout probability")
    # evaluation hyper-parameters
    parser.add_argument("--eval_sample_size", type=int, default=1000, help="Evaluation sample size")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Evaluation batch size")
    parser.add_argument("--eval_sampling_steps", type=int, default=35, help="Evaluation sampling steps")
    parser.add_argument("--eval_fix_noise_seed", action="store_true", help="Evaluation fix noise seed")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    # dataset hyper-parameters
    parser.add_argument("--sample_num", type=int, default=4096, help="Number of training samples")
    parser.add_argument("--sample_len", type=int, default=36, help="Sample length (D)")
    parser.add_argument("--k_ones", type=int, default=18, help="Number of ones in each sample (K)")
    parser.add_argument(
        '-r', '--record_step_range',
        metavar=('START', 'END', 'STEP'),
        type=int,
        nargs=3,
        action='append',
        default=[],
        help="Define a range with start, end, and step. Can be used multiple times."
    )
    parser.add_argument("--save_ckpts", action="store_true", help="Save checkpoint trajectory")
    parser.add_argument("--num_ckpts", type=int, default=100, help="Number of checkpoints")
    return parser.parse_args()


args = parse_args()
sample_num = args.sample_num
sample_len = args.sample_len
k_ones = args.k_ones
exp_name = args.exp_name
batch_size = args.batch_size
nsteps = args.nsteps
lr = args.lr
eval_sample_size = args.eval_sample_size
eval_batch_size = args.eval_batch_size
eval_sampling_steps = args.eval_sampling_steps
eval_fix_noise_seed = args.eval_fix_noise_seed
record_frequency = args.record_frequency
record_step_range = args.record_step_range
save_ckpts = args.save_ckpts
num_ckpts = args.num_ckpts
ckpt_step_list = generate_ckpt_step_list(nsteps, num_ckpts=num_ckpts, sequence="geomspace")
if args.record_step_range is None or len(args.record_step_range) == 0:
    print("using default record step range")
    ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256), (50000, 100000, 512), (100000, 1000000, 1024)]
    record_step_range = ranges
else:
    record_step_range = args.record_step_range
    ranges = []
    for r in record_step_range:
        try:
            parsed_range = parse_range(r)
            ranges.append(parsed_range)
        except argparse.ArgumentTypeError as e:
            raise argparse.ArgumentTypeError(str(e))

record_times = generate_record_times(ranges)
print(f"record_frequency: {record_frequency}")
print(f"record_step_range: {record_step_range}")
print(f"record_times: {record_times}")

saveroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"
savedir = f"{saveroot}/{exp_name}"
sample_dir = f"{savedir}/samples"
ckpt_dir = f"{savedir}/ckpts"
os.makedirs(savedir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
writer = SummaryWriter(log_dir=f"{savedir}/tensorboard")

# %%
loss_store = {}
def sampling_eval_callback_fn(epoch, loss, model, grad_norm=None):
    loss_store[epoch] = loss
    x_out_batches = []
    if eval_fix_noise_seed:
        noise_init_all = torch.randn(eval_sample_size, *imgshape, generator=torch.Generator().manual_seed(0))
    else:
        noise_init_all = torch.randn(eval_sample_size, *imgshape)
    for i in range(0, eval_sample_size, eval_batch_size):
        batch_size_i = min(eval_batch_size, eval_sample_size - i)
        noise_init = noise_init_all[i:i+batch_size_i].to(device)
        x_out_i = edm_sampler(model, noise_init, num_steps=eval_sampling_steps,
                        sigma_min=0.002, sigma_max=80, rho=7, return_traj=False)
        x_out_batches.append(x_out_i)

    x_out = torch.cat(x_out_batches, dim=0)
    torch.save(x_out, f"{sample_dir}/samples_epoch_{epoch:06d}.pt")
    mtg = to_imgrid(((x_out.cpu()[:64] + 1) / 2).clamp(0, 1), nrow=8, padding=1)
    mtg.save(f"{sample_dir}/samples_epoch_{epoch:06d}.png")

    # Invalid sample ratios at two eps thresholds
    sample_int_1e1 = round_to_pos_neg_one(x_out, eps=1e-1)
    nan_mask_1e1 = th.isnan(sample_int_1e1.flatten(1)).any(dim=1)
    nan_ratio_1e1 = nan_mask_1e1.float().mean().item()

    sample_int_1e2 = round_to_pos_neg_one(x_out, eps=1e-2)
    nan_mask_1e2 = th.isnan(sample_int_1e2.flatten(1)).any(dim=1)
    nan_ratio_1e2 = nan_mask_1e2.float().mean().item()

    # Use eps=1e-1 for further evaluation
    sample_flat = sample_int_1e1.flatten(1).cpu().numpy()
    nan_mask = np.isnan(sample_flat).any(axis=1)
    nan_num = int(nan_mask.sum())

    # Evaluate exact-K correctness on valid samples
    valid_flat = sample_flat[~nan_mask]
    if len(valid_flat) > 0:
        k_correct = exact_k_check(valid_flat, k_ones)
        k_correct_num = int(k_correct.sum())
        k_correct_ratio = k_correct_num / len(valid_flat)
        ones_counts = ones_count(valid_flat)
        mean_ones = float(ones_counts.mean())
    else:
        k_correct_num = 0
        k_correct_ratio = 0.0
        ones_counts = np.array([0])
        mean_ones = float('nan')

    # Memorization: compare generated samples to precomputed train codes
    if len(valid_flat) > 0:
        gen_bits  = th.from_numpy((valid_flat > 0).astype(np.int64))
        gen_codes = (gen_bits * _mem_weights).sum(dim=1)
        mem_ratio = th.isin(gen_codes, _train_codes).float().mean().item()
    else:
        mem_ratio = 0.0

    print(f"epoch: {epoch:06d} | exact-K correct: {k_correct_num}/{eval_sample_size - nan_num} valid "
          f"({k_correct_ratio:.3f}) | mem: {mem_ratio:.3f} | mean ones: {mean_ones:.2f} "
          f"(target K={k_ones}) | nan: {nan_num}")

    # TensorBoard logging
    writer.add_scalar("train/loss",              loss,            epoch)
    writer.add_scalar("eval/k_correct_ratio",    k_correct_ratio, epoch)
    writer.add_scalar("eval/sample_mem_ratio",   mem_ratio,       epoch)
    writer.add_scalar("eval/nan_ratio_eps_1e-1", nan_ratio_1e1,   epoch)
    writer.add_scalar("eval/nan_ratio_eps_1e-2", nan_ratio_1e2,   epoch)
    writer.add_scalar("eval/mean_ones",          mean_ones,       epoch)
    writer.add_histogram("eval/ones_count_dist",
                         th.from_numpy(ones_counts.astype(np.int32)), epoch)


device = get_device()
imgsize = int(math.sqrt(sample_len))
imgchannels = 1
print(f"Generating exact-K dataset: N={sample_num}, D={sample_len}, K={k_ones}")
x = sample_exact_k_dataset(N=sample_num, D=sample_len, K=k_ones)
dataset_name = f"exactK_N{sample_num}_D{sample_len}_K{k_ones}"
print(dataset_name, "dataset")
Xtsr = torch.from_numpy(x).float()
Xtsr = Xtsr.view(sample_num, imgchannels, imgsize, imgsize)
th.save(Xtsr, f"{savedir}/training_data_tsr.pt")

# Precompute train sample codes once for fast memorization check in callback
_mem_weights = 1 << th.arange(sample_len, dtype=th.long)
_train_bits  = (Xtsr.flatten(1) > 0).long().cpu()
_train_codes = (_train_bits * _mem_weights).sum(dim=1).unique()

sigma_data = 1.0
pnts = Xtsr.view(Xtsr.shape[0], -1)
imgshape = Xtsr.shape[1:]
ndim = pnts.shape[1]
args.dataset_name = dataset_name
print(f"dataset {Xtsr.shape[0]} samples, {ndim} features")
config = edict(
    input_size=imgsize,
    in_channels=imgchannels,
    patch_size=args.patch_size,
    hidden_size=args.hidden_size,
    depth=args.depth,
    num_heads=args.num_heads,
    mlp_ratio=args.mlp_ratio,
    class_dropout_prob=args.class_dropout_prob,
    num_classes=0,
    learn_sigma=False,
)
pprint(config)

json.dump(config, open(f"{savedir}/config.json", "w"))
json.dump(args.__dict__, open(f"{savedir}/args.json", "w"))

DiT_model = DiT(**config)
model_precd = EDMDiTPrecondWrapper(DiT_model, sigma_data=sigma_data, sigma_min=0.002, sigma_max=80, rho=7.0)
if args.loss_type == "DSM":
    edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=sigma_data)
elif args.loss_type == "ESM":
    edm_loss_fn = EDMDeltaGMMScoreLoss(train_Xmat=Xtsr.to(device), P_mean=-1.2, P_std=1.2, sigma_data=sigma_data)
else:
    raise ValueError(f"Invalid loss type: {args.loss_type}")
model_precd, loss_traj = train_score_model_custom_loss(Xtsr, model_precd, edm_loss_fn,
                                    lr=lr, nepochs=nsteps, batch_size=batch_size, device=device,
                                    callback=sampling_eval_callback_fn, callback_freq=record_frequency, callback_step_list=record_times,
                                    save_ckpts=save_ckpts, ckpt_dir=ckpt_dir, save_ckpt_step_list=ckpt_step_list)

pkl.dump(loss_store, open(f"{savedir}/loss_store.pkl", "wb"))
pkl.dump(loss_traj, open(f"{savedir}/loss_traj.pkl", "wb"))
torch.save(model_precd.model.state_dict(), f"{savedir}/model_final.pth")

noise_init = torch.randn(64, *imgshape).to(device)
x_out, x_traj, x0hat_traj, t_steps = edm_sampler(model_precd, noise_init,
                num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
mtg = to_imgrid(((x_out.cpu()[:]+1)/2).clamp(0, 1), nrow=8, padding=1)
mtg.save(f"{savedir}/learned_samples_final.png")
writer.close()
# %%
