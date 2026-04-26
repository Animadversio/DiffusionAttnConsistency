
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
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from easydict import EasyDict as edict
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone_NoFirstNorm
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
from core.diffusion_esm_edm_lib import EDMDeltaGMMScoreLoss
from core.DiT_model_lib import *
from core.latin_square_lib import (
    sample_latin_square_dataset,
    encode_latin_square,
    int_to_onehot,
    snap_to_integer,
    evaluate_latin_square_samples,
    evaluate_latin_square_onehot_samples,
    compute_memorization,
    valid_set_size,
    expected_memorization_ratio,
)
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
    parser = argparse.ArgumentParser(description="DiT Latin Square Learning Experiment")
    parser.add_argument("--exp_name", type=str, default="DiT_mini_latinSq_n6_N4096", help="Experiment name")
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
    parser.add_argument("--encoding", type=str, default="scalar", choices=["scalar", "onehot"],
                        help="Encoding: 'scalar' (1-channel normalized int) or 'onehot' (n-channel binary {-1,+1})")
    parser.add_argument("--onehot_type", type=str, default="pm1",
                        choices=["pm1", "zero_one", "zero_mean"],
                        help="One-hot encoding variant: pm1={-1,+1}, zero_one={0,1}, zero_mean=zero-mean {-1/n,(n-1)/n}")
    parser.add_argument("--sigma_data", type=str, default="1.0",
                        help="EDM sigma_data; use 'auto' to compute from training data RMS")
    parser.add_argument("--snap_eps", type=float, default=0.15, help="Snapping tolerance for scalar decoding")
    parser.add_argument("--onehot_eps", type=float, default=0.3, help="Ambiguity threshold for one-hot decoding (max channel must exceed 1-eps)")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    # dataset hyper-parameters
    parser.add_argument("--sample_num", type=int, default=4096, help="Number of training samples")
    parser.add_argument("--n_size", type=int, default=6, help="Latin square size (n×n, symbols {0,...,n-1})")
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
n_size = args.n_size
sample_len = n_size * n_size
exp_name = args.exp_name
batch_size = args.batch_size
nsteps = args.nsteps
lr = args.lr
eval_sample_size = args.eval_sample_size
eval_batch_size = args.eval_batch_size
eval_sampling_steps = args.eval_sampling_steps
eval_fix_noise_seed = args.eval_fix_noise_seed
encoding = args.encoding
onehot_type = args.onehot_type
snap_eps = args.snap_eps
onehot_eps = args.onehot_eps
record_frequency = args.record_frequency
save_ckpts = args.save_ckpts
num_ckpts = args.num_ckpts
ckpt_step_list = generate_ckpt_step_list(nsteps, num_ckpts=num_ckpts, sequence="geomspace")

if args.record_step_range is None or len(args.record_step_range) == 0:
    print("using default record step range")
    ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16),
              (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256),
              (50000, 100000, 512), (100000, 1000000, 1024)]
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

def sampling_eval_callback_fn(epoch, loss, model):
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

    x_out = torch.cat(x_out_batches, dim=0)   # (eval_sample_size, C, n_size, n_size)
    torch.save(x_out, f"{sample_dir}/samples_epoch_{epoch:06d}.pt")

    # Save image grid — always visualize as grayscale (1 channel)
    x_cpu = x_out.cpu()
    if encoding == "onehot":
        # argmax over symbol channel → intensity in [0, 1]
        x_vis = (x_cpu.argmax(dim=1, keepdim=True).float() / (n_size - 1)).clamp(0, 1)
    else:
        x_vis = ((x_cpu[:64] + 1) / 2).clamp(0, 1)
    mtg = to_imgrid(x_vis[:64], nrow=8, padding=1)
    mtg.save(f"{sample_dir}/samples_epoch_{epoch:06d}.png")

    # --- Encoding-specific decoding & evaluation ---
    if encoding == "scalar":
        x_flat_cont = x_cpu.flatten(1).numpy()   # (N, n²) float
        metrics_perm = evaluate_latin_square_samples(x_flat_cont, n_size, eps=snap_eps)
        metrics_strict = evaluate_latin_square_samples(x_flat_cont, n_size, eps=snap_eps * 0.1)
        nan_ratio_perm   = metrics_perm["nan_ratio"]
        nan_ratio_strict = metrics_strict["nan_ratio"]
        valid_int = metrics_perm["valid_int"]
    else:  # onehot
        x_oh = x_cpu.numpy().reshape(eval_sample_size, n_size, n_size * n_size)  # (N, n, n²)
        metrics_perm   = evaluate_latin_square_onehot_samples(x_oh, n_size, eps=onehot_eps)
        metrics_strict = evaluate_latin_square_onehot_samples(x_oh, n_size, eps=onehot_eps * 0.33)
        nan_ratio_perm   = metrics_perm["nan_ratio"]
        nan_ratio_strict = metrics_strict["nan_ratio"]
        valid_int = metrics_perm["valid_int"]

    M = len(valid_int)
    if M > 0:
        full_valid_ratio = metrics_perm["full_valid_ratio"]
        row_valid_ratio  = metrics_perm["row_valid_ratio"]
        col_valid_ratio  = metrics_perm["col_valid_ratio"]
        mem_ratio = compute_memorization(_train_int_flat, valid_int)
    else:
        full_valid_ratio = row_valid_ratio = col_valid_ratio = mem_ratio = 0.0

    print(f"epoch: {epoch:06d} | [{encoding}] full_valid: {int(M * full_valid_ratio)}/{M} valid "
          f"({full_valid_ratio:.3f}) | row={row_valid_ratio:.3f} col={col_valid_ratio:.3f} "
          f"| mem={mem_ratio:.4f} | nan_perm={nan_ratio_perm:.3f} nan_strict={nan_ratio_strict:.3f}")

    # TensorBoard logging
    writer.add_scalar("train/loss",              loss,             epoch)
    writer.add_scalar("eval/full_valid_ratio",   full_valid_ratio, epoch)
    writer.add_scalar("eval/row_valid_ratio",    row_valid_ratio,  epoch)
    writer.add_scalar("eval/col_valid_ratio",    col_valid_ratio,  epoch)
    writer.add_scalar("eval/sample_mem_ratio",   mem_ratio,        epoch)
    writer.add_scalar("eval/nan_ratio_permissive", nan_ratio_perm,   epoch)
    writer.add_scalar("eval/nan_ratio_strict",     nan_ratio_strict, epoch)


device = get_device()
imgsize = n_size

print(f"Generating Latin square dataset: N={sample_num}, n={n_size}, encoding={encoding}")
x_int = sample_latin_square_dataset(N=sample_num, n=n_size)   # (N, n²), integer

dataset_name = f"latinSq_n{n_size}_N{sample_num}_{encoding}"
print(f"{dataset_name} dataset, valid_set_size={valid_set_size(n_size):,}, "
      f"mem_ratio={expected_memorization_ratio(sample_num, n_size):.3e}")

if encoding == "scalar":
    # (N, n²) float in [-1, +1], reshaped to (N, 1, n, n)
    x_encoded = encode_latin_square(x_int, n_size).astype(np.float32)
    imgchannels = 1
    Xtsr = torch.from_numpy(x_encoded).view(sample_num, imgchannels, imgsize, imgsize)
else:  # onehot
    if onehot_type == "pm1":
        active, inactive = 1.0, -1.0
    elif onehot_type == "zero_one":
        active, inactive = 1.0, 0.0
    elif onehot_type == "zero_mean":
        active  = (n_size - 1) / n_size
        inactive = -1.0 / n_size
    else:
        raise ValueError(f"Unknown onehot_type: {onehot_type}")
    x_encoded = int_to_onehot(x_int, n_size, active=active, inactive=inactive)
    imgchannels = n_size
    Xtsr = torch.from_numpy(x_encoded).view(sample_num, imgchannels, imgsize, imgsize)

th.save(Xtsr, f"{savedir}/training_data_tsr.pt")

# Precompute integer training set for memorization lookup (encoding-agnostic)
_train_int_flat = x_int   # (N, n²) integer

if args.sigma_data == "auto":
    sigma_data = float(Xtsr.pow(2).mean().sqrt())
    print(f"sigma_data (auto) = {sigma_data:.4f}")
else:
    sigma_data = float(args.sigma_data)
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

model_precd, loss_traj = train_score_model_custom_loss(
    Xtsr, model_precd, edm_loss_fn,
    lr=lr, nepochs=nsteps, batch_size=batch_size, device=device,
    callback=sampling_eval_callback_fn,
    callback_freq=record_frequency,
    callback_step_list=record_times,
    save_ckpts=save_ckpts, ckpt_dir=ckpt_dir, save_ckpt_step_list=ckpt_step_list,
)

pkl.dump(loss_store, open(f"{savedir}/loss_store.pkl", "wb"))
pkl.dump(loss_traj, open(f"{savedir}/loss_traj.pkl", "wb"))
torch.save(model_precd.model.state_dict(), f"{savedir}/model_final.pth")

noise_init = torch.randn(64, *imgshape).to(device)
x_out, x_traj, x0hat_traj, t_steps = edm_sampler(
    model_precd, noise_init,
    num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True,
)
if encoding == "onehot":
    x_vis = (x_out.cpu().argmax(dim=1, keepdim=True).float() / (n_size - 1)).clamp(0, 1)
else:
    x_vis = ((x_out.cpu() + 1) / 2).clamp(0, 1)
mtg = to_imgrid(x_vis, nrow=8, padding=1)
mtg.save(f"{savedir}/learned_samples_final.png")
writer.close()
# %%
