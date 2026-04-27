
# %%
import sys
import os
from os.path import join
import json
import pickle as pkl
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from tqdm.auto import trange, tqdm
from pprint import pprint
import argparse
from typing import List, Tuple
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
from core.DiT_model_lib import *
from core.row_k_lib import (
    sample_row_k_batch, sample_row_k_dataset,
    sample_row_variable_k_batch, sample_row_variable_k_dataset,
    sample_global_k_batch, sample_global_k_dataset,
    evaluate_row_k_samples, evaluate_row_variable_k_samples, evaluate_global_k_samples,
    valid_set_size_row_k, valid_set_size_row_variable_k, valid_set_size_global_k,
)
from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid
from torch.utils.tensorboard import SummaryWriter


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_ckpt_step_list(max_steps, num_ckpts=100, sequence="geomspace"):
    if sequence == "geomspace":
        steps = np.geomspace(1, max_steps + 1, num_ckpts).astype(int)
    else:
        steps = np.linspace(1, max_steps, num_ckpts).astype(int)
    return np.unique(steps[steps <= max_steps]).tolist()


def generate_record_times(ranges):
    times = []
    for start, end, step in ranges:
        times.extend(range(start, end, step))
    return times


def parse_args():
    parser = argparse.ArgumentParser(description="DiT Row-K Learning Experiment")
    parser.add_argument("--exp_name", type=str, default="DiT_mini_rowK3_n6_N4096")
    parser.add_argument("--loss_type", type=str, default="DSM", choices=["DSM", "ESM"])
    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nsteps", type=int, default=1000000)
    # model
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)
    # evaluation
    parser.add_argument("--eval_sample_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--eval_sampling_steps", type=int, default=35)
    parser.add_argument("--eval_fix_noise_seed", action="store_true")
    parser.add_argument("--record_frequency", type=int, default=0)
    parser.add_argument(
        "-r", "--record_step_range",
        metavar=("START", "END", "STEP"),
        type=int, nargs=3, action="append", default=[],
    )
    # dataset / rule
    parser.add_argument("--sample_num", type=int, default=4096)
    parser.add_argument("--n_size", type=int, default=6, help="Grid size (n×n)")
    parser.add_argument("--rule", type=str, default="row_k",
                        choices=["row_k", "row_variable_k", "global_k"])
    parser.add_argument("--K", type=int, default=3,
                        help="Active count per row (used for row_k)")
    parser.add_argument("--K_list", type=int, nargs="+", default=[1, 5],
                        help="Allowed active counts (used for row_variable_k and global_k)")
    # checkpoints
    parser.add_argument("--save_ckpts", action="store_true")
    parser.add_argument("--num_ckpts", type=int, default=100)
    return parser.parse_args()


args = parse_args()
rule = args.rule
n_size = args.n_size
K = args.K
K_list = args.K_list
sample_num = args.sample_num
exp_name = args.exp_name
batch_size = args.batch_size
nsteps = args.nsteps
lr = args.lr
eval_sample_size = args.eval_sample_size
eval_batch_size = args.eval_batch_size
eval_sampling_steps = args.eval_sampling_steps
eval_fix_noise_seed = args.eval_fix_noise_seed
record_frequency = args.record_frequency
ckpt_step_list = generate_ckpt_step_list(nsteps, num_ckpts=args.num_ckpts)

if args.record_step_range is None or len(args.record_step_range) == 0:
    ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16),
              (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256),
              (50000, 100000, 512), (100000, 1000000, 1024)]
else:
    ranges = [tuple(r) for r in args.record_step_range]
record_times = generate_record_times(ranges)
print(f"record_frequency: {record_frequency}, record_times: {len(record_times)} steps")

saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"
savedir = f"{saveroot}/{exp_name}"
sample_dir = f"{savedir}/samples"
ckpt_dir = f"{savedir}/ckpts"
os.makedirs(savedir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
writer = SummaryWriter(log_dir=f"{savedir}/tensorboard")

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
print(f"Generating dataset: rule={rule}, n={n_size}, N={sample_num}, "
      + (f"K={K}" if rule == "row_k" else f"K_list={K_list}"))

if rule == "row_k":
    vs = valid_set_size_row_k(n_size, K)
    if sample_num <= vs:
        x_int = sample_row_k_dataset(sample_num, n_size, K)
    else:
        x_int = sample_row_k_batch(sample_num, n_size, K)
    dataset_name = f"rowK{K}_n{n_size}_N{sample_num}"
elif rule == "row_variable_k":
    vs = valid_set_size_row_variable_k(n_size, K_list)
    if sample_num <= vs:
        x_int = sample_row_variable_k_dataset(sample_num, n_size, K_list)
    else:
        x_int = sample_row_variable_k_batch(sample_num, n_size, K_list)
    kstr = "".join(str(k) for k in sorted(K_list))
    dataset_name = f"rowVarK{kstr}_n{n_size}_N{sample_num}"
elif rule == "global_k":
    vs = valid_set_size_global_k(n_size, K_list)
    if sample_num <= vs:
        x_int, _ = sample_global_k_dataset(sample_num, n_size, K_list)
    else:
        x_int, _ = sample_global_k_batch(sample_num, n_size, K_list)
    kstr = "".join(str(k) for k in sorted(K_list))
    dataset_name = f"globalK{kstr}_n{n_size}_N{sample_num}"

print(f"Dataset: {dataset_name}, valid_set_size={vs:.3e}, mem_ratio={sample_num/vs:.3e}")

# Shape: (N, 1, n, n) — single channel binary grid
Xtsr = torch.from_numpy(x_int.reshape(sample_num, 1, n_size, n_size)).float()
th.save(Xtsr, f"{savedir}/training_data_tsr.pt")

# Precompute train codes for memorization check (bit-packing over n²=36 dims)
_n2 = n_size * n_size
_mem_weights = 1 << th.arange(_n2, dtype=th.long)
_train_bits  = (Xtsr.flatten(1) > 0).long().cpu()
_train_codes = (_train_bits * _mem_weights).sum(dim=1).unique()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
imgshape = Xtsr.shape[1:]   # (1, n_size, n_size)
sigma_data = 1.0

config = edict(
    input_size=n_size,
    in_channels=1,
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

args.dataset_name = dataset_name
json.dump(config, open(f"{savedir}/config.json", "w"))
json.dump(args.__dict__, open(f"{savedir}/args.json", "w"))

DiT_model = DiT(**config)
model_precd = EDMDiTPrecondWrapper(DiT_model, sigma_data=sigma_data, sigma_min=0.002, sigma_max=80, rho=7.0)

# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------
loss_store = {}

def sampling_eval_callback_fn(epoch, loss, model, grad_norm=None):
    loss_store[epoch] = loss

    if eval_fix_noise_seed:
        noise_init_all = torch.randn(eval_sample_size, *imgshape,
                                     generator=torch.Generator().manual_seed(0))
    else:
        noise_init_all = torch.randn(eval_sample_size, *imgshape)

    x_out_batches = []
    for i in range(0, eval_sample_size, eval_batch_size):
        bsz = min(eval_batch_size, eval_sample_size - i)
        x_out_i = edm_sampler(model, noise_init_all[i:i+bsz].to(device),
                               num_steps=eval_sampling_steps,
                               sigma_min=0.002, sigma_max=80, rho=7, return_traj=False)
        x_out_batches.append(x_out_i)
    x_out = torch.cat(x_out_batches, dim=0)   # (eval_sample_size, 1, n, n)
    torch.save(x_out, f"{sample_dir}/samples_epoch_{epoch:06d}.pt")

    mtg = to_imgrid(((x_out.cpu()[:64] + 1) / 2).clamp(0, 1), nrow=8, padding=1)
    mtg.save(f"{sample_dir}/samples_epoch_{epoch:06d}.png")

    # Flatten to (N, n²) float for eval
    x_flat = x_out.cpu().flatten(1).numpy()

    if rule == "row_k":
        m = evaluate_row_k_samples(x_flat, n_size, K, eps=0.15)
        valid_int = m["valid_int"]
        full_valid = m["full_valid_ratio"]
        per_row_valid = m["per_row_valid_ratio"]
        row_k_valid = m["row_k_valid"]
        rule_info = f"per_row={per_row_valid:.3f}  row_k_valid={row_k_valid:.3f}"
    elif rule == "row_variable_k":
        m = evaluate_row_variable_k_samples(x_flat, n_size, K_list, eps=0.15)
        valid_int = m["valid_int"]
        full_valid = m["full_valid_ratio"]
        per_row_valid = m["per_row_valid_ratio"]
        row_var_k_valid = m["row_var_k_valid"]
        rule_info = f"per_row={per_row_valid:.3f}  row_var_k_valid={row_var_k_valid:.3f}"
    elif rule == "global_k":
        m = evaluate_global_k_samples(x_flat, n_size, K_list, eps=0.15)
        valid_int = m["valid_int"]
        full_valid = m["full_valid_ratio"]
        per_row_valid = m["per_row_valid_ratio"]
        row_consistency = m["row_consistency"]
        global_k_valid = m["global_k_valid"]
        rule_info = (f"per_row={per_row_valid:.3f}  consistency={row_consistency:.3f}"
                     f"  global_k_valid={global_k_valid:.3f}  breakdown={m['per_k_breakdown']}")

    # Memorization check
    if len(valid_int) > 0:
        gen_bits  = th.from_numpy((valid_int > 0).astype(np.int64))
        gen_codes = (gen_bits * _mem_weights).sum(dim=1)
        mem_ratio = th.isin(gen_codes, _train_codes).float().mean().item()
    else:
        mem_ratio = 0.0

    print(f"epoch: {epoch:06d} | [{rule}] full_valid={full_valid:.3f} | {rule_info} "
          f"| nan={m['nan_ratio']:.3f} | mem={mem_ratio:.4f} | k_hist={m['per_row_k_hist']}")

    # TensorBoard
    writer.add_scalar("train/loss",              loss,           epoch)
    if grad_norm is not None:
        writer.add_scalar("train/grad_norm",     grad_norm,      epoch)
    writer.add_scalar("eval/full_valid_ratio",   full_valid,     epoch)
    writer.add_scalar("eval/per_row_valid_ratio", per_row_valid, epoch)
    writer.add_scalar("eval/nan_ratio",          m["nan_ratio"], epoch)
    writer.add_scalar("eval/sample_mem_ratio",   mem_ratio,      epoch)
    if rule == "row_k":
        writer.add_scalar("eval/row_k_valid",    m["row_k_valid"], epoch)
    elif rule == "row_variable_k":
        writer.add_scalar("eval/row_var_k_valid", m["row_var_k_valid"], epoch)
    elif rule == "global_k":
        writer.add_scalar("eval/row_consistency", m["row_consistency"], epoch)
        writer.add_scalar("eval/global_k_valid",  m["global_k_valid"],  epoch)
        for k_val, cnt in m["per_k_breakdown"].items():
            writer.add_scalar(f"eval/per_k_valid_K{k_val}",
                              cnt / max(len(valid_int), 1), epoch)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
device = get_device()
edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=sigma_data)

model_precd, loss_traj = train_score_model_custom_loss(
    Xtsr, model_precd, edm_loss_fn,
    lr=lr, nepochs=nsteps, batch_size=batch_size, device=device,
    callback=sampling_eval_callback_fn,
    callback_freq=record_frequency,
    callback_step_list=record_times,
    save_ckpts=args.save_ckpts,
    ckpt_dir=ckpt_dir,
    save_ckpt_step_list=ckpt_step_list,
)

pkl.dump(loss_store, open(f"{savedir}/loss_store.pkl", "wb"))
pkl.dump(loss_traj,  open(f"{savedir}/loss_traj.pkl",  "wb"))
torch.save(model_precd.model.state_dict(), f"{savedir}/model_final.pth")

noise_init = torch.randn(64, *imgshape).to(device)
x_out, *_ = edm_sampler(model_precd, noise_init,
                         num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
mtg = to_imgrid(((x_out.cpu() + 1) / 2).clamp(0, 1), nrow=8, padding=1)
mtg.save(f"{savedir}/learned_samples_final.png")
writer.close()
