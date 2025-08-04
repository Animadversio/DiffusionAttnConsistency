
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
from core.parity_lib import sample_group_parity_vec, sample_ensuring_uniqueness, parity_func, round_to_pos_neg_one
# from core.dataset_lib import load_dataset
from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# def load_raw_dataset(dataset_name):
#     import sys
#     sys.path.append("/n/home12/binxuwang/Github/edm")
#     from training.dataset import TensorDataset, ImageFolderDataset
#     ffhq256dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256"
#     edm_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
#     if dataset_name == "FFHQ":
#         edm_ffhq64_path = join(edm_dataset_root, "ffhq-64x64.zip")
#         dataset = ImageFolderDataset(edm_ffhq64_path)
#         imgsize = 64
#         imgchannels = 3
#         Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
#     elif dataset_name == "AFHQ":
#         edm_afhq_path = join(edm_dataset_root, "afhqv2-64x64.zip")
#         dataset = ImageFolderDataset(edm_afhq_path)
#         imgsize = 64
#         imgchannels = 3
#         Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
#     elif dataset_name == "CIFAR":
#         edm_cifar_path = join(edm_dataset_root, "cifar10-32x32.zip")
#         dataset = ImageFolderDataset(edm_cifar_path)
#         imgsize = 32
#         imgchannels = 3
#         Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
#     elif dataset_name == "words32x32_50k":
#         wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
#         image_tensor = torch.load(join(wordimg_root, "words32x32_50k.pt"))
#         text_list = pkl.load(open(join(wordimg_root, "words32x32_50k_words.pkl"), "rb"))
#         imgsize = 32
#         imgchannels = 1
#         Xtsr_raw = image_tensor
#     elif dataset_name == "words32x32_50k_BW":
#         wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
#         image_tensor = torch.load(join(wordimg_root, "words32x32_50k_BW.pt"))
#         text_list = pkl.load(open(join(wordimg_root, "words32x32_50k_BW_words.pkl"), "rb"))
#         imgsize = 32
#         imgchannels = 1
#         Xtsr_raw = image_tensor
#     elif dataset_name == "FFHQ_fix_words":
#         wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
#         save_path = join(wordimg_root, "ffhq-64x64-fixed_text.pt")
#         image_tensor = torch.load(save_path)
#         imgsize = 64
#         imgchannels = 3
#         Xtsr_raw = image_tensor
#     elif dataset_name == "FFHQ_random_words_jitter":
#         wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
#         save_path = join(wordimg_root, "ffhq-64x64-random_word_jitter2-8.pt")
#         image_tensor = torch.load(save_path)
#         imgsize = 64
#         imgchannels = 3
#         Xtsr_raw = image_tensor
#     elif dataset_name == "ffhq-32x32":
#         data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/ffhq-32x32.pt")
#         imgsize = 32
#         imgchannels = 3
#         Xtsr_raw = data_Xtsr
#     elif dataset_name == "ffhq-32x32-fix_words":
#         data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/ffhq-32x32-fixed_text.pt")
#         imgsize = 32
#         imgchannels = 3
#         Xtsr_raw = data_Xtsr
#     elif dataset_name == "ffhq-32x32-random_word_jitter":
#         data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/ffhq-32x32-random_word_jitter1-4.pt")
#         imgsize = 32
#         imgchannels = 3
#         Xtsr_raw = data_Xtsr
#     elif dataset_name == "afhq-32x32":
#         data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/afhq-32x32.pt")
#         imgsize = 32
#         imgchannels = 3
#         Xtsr_raw = data_Xtsr

#     assert Xtsr_raw.shape[1] == imgchannels
#     assert Xtsr_raw.shape[2] == imgsize
#     assert Xtsr_raw.shape[3] == imgsize
#     print(f"{dataset_name} dataset: {len(Xtsr_raw)}")
#     print(f"value range" , (Xtsr_raw.max()), (Xtsr_raw.min()))
#     return Xtsr_raw, imgsize, imgchannels

# %%
from pprint import pprint
import argparse
from typing import List, Tuple
def parse_range(range_str: List[str]) -> Tuple[int, int, int]:
    """
    Parses a list of strings into a tuple of three integers representing a range.

    Args:
        range_str (List[str]): List containing start, end, and step as strings.

    Returns:
        Tuple[int, int, int]: Parsed (start, end, step).

    Raises:
        argparse.ArgumentTypeError: If the input is invalid.
    """
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
    """
    Generates a list of record times based on the provided ranges.

    Args:
        ranges (List[Tuple[int, int, int]]): List of ranges defined by (start, end, step).

    Returns:
        List[int]: Generated record times.
    """
    record_times = []
    for start, end, step in ranges:
        record_times.extend(range(start, end, step))
    return record_times


def generate_ckpt_step_list(max_steps, num_ckpts=100, sequence="geomspace") -> List[int]:
    """
    Generates a list of record times based on the provided ranges.

    Args:
        ranges (List[Tuple[int, int, int]]): List of ranges defined by (start, end, step).

    Returns:
        List[int]: Generated record times.
    """
    if sequence == "geomspace":
        # ckpt_step_list = np.unique(np.logspace(np.log10(1), np.log10(max_steps+1), num_ckpts).astype(int))
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
    parser = argparse.ArgumentParser(description="DiT Learning Curve Experiment")
    # parser.add_argument("--dataset_name", type=str, default="ffhq-32x32", help="Dataset name")
    parser.add_argument("--exp_name", type=str, default="FFHQ32_DiT_CNN_EDM", help="Experiment name")
    parser.add_argument("--loss_type", type=str, default="DSM", help="Loss type (DSM, ESM)")
    # training hyper-parameters
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--nsteps", type=int, default=5000, help="Number of steps")
    # model hyper-parameters
    parser.add_argument("--patch_size", type=int, default=1, help="Patch size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1, help="Class dropout probability")
    # NOTE: `class_dropout_prob` need to be non zero, or the y embedding will not work be a (0, d) shaped embedding 
    # evaluation hyper-parameters
    parser.add_argument("--eval_sample_size", type=int, default=1000, help="Evaluation sample size")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Evaluation batch size")
    parser.add_argument("--eval_sampling_steps", type=int, default=35, help="Evaluation sampling steps")
    parser.add_argument("--eval_fix_noise_seed", action="store_true", help="Evaluation fix noise seed")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    parser.add_argument("--sample_num", type=int, default=4096, help="Sample number")
    parser.add_argument("--sample_len", type=int, default=36, help="Sample length")
    parser.add_argument("--group_size", type=int, default=36, help="Group size")
    parser.add_argument("--parity", type=int, default=0, help="Parity (0 for even, 1 for odd)")
    parser.add_argument(
        '-r', '--record_step_range',
        metavar=('START', 'END', 'STEP'),
        type=int,
        nargs=3,
        action='append',
        # default=[(0, 10, 2), (10, 50, 4), (50, 100, 8), (100, 500, 16), (500, 2500, 32), (2500, 5000, 64), (5000, 10000, 128), (10000, 50000, 256)],#
        # default=[(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256)],#
        default=[],
        help="Define a range with start, end, and step. Can be used multiple times. Evaluation sample frequency"
    )
    parser.add_argument("--save_ckpts", action="store_true", help="Save checkpoint trajectory")
    parser.add_argument("--num_ckpts", type=int, default=100, help="Number of checkpoints")
    return parser.parse_args()

# %%
# layers_per_block = 1
# decoder_init_attn = True
# attn_resolutions = [8, 16,]
# model_channels = 64
# channel_mult = [1, 2, 4, 4]
# batch_size = 512
# nsteps = 10000
# lr = 1e-4
# record_frequency = 100
# exp_name = "ffhq64_edm_model_pilot"
# eval_sample_size = 5000
# eval_batch_size = 1024 # Process in batches of 1000

# ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 64)]
# record_times = generate_record_times(ranges)
# import json 
# args = json.load(open("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/FFHQ32_DiT_P2_192D_3H_6L_EDM_pilot/args.json", "r"))
# args = edict(args)
# args.class_dropout_prob = 0.1


args = parse_args()
# dataset_name = args.dataset_name
sample_num = args.sample_num
sample_len = args.sample_len
group_size = args.group_size
parity = args.parity
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

#%%
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
        # x_out_i, x_traj_i, x0hat_traj_i, t_steps_i = edm_sampler(model, noise_init,
        #                 num_steps=eval_sampling_steps, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
        x_out_batches.append(x_out_i)
    
    x_out = torch.cat(x_out_batches, dim=0)
    # sample_store[epoch] = x_out.cpu(), # x_traj.cpu(), x0hat_traj.cpu(), t_steps.cpu()
    torch.save(x_out, f"{sample_dir}/samples_epoch_{epoch:06d}.pt")
    mtg = to_imgrid(((x_out.cpu()[:64] + 1) / 2).clamp(0, 1), nrow=8, padding=1)
    mtg.save(f"{sample_dir}/samples_epoch_{epoch:06d}.png")
        
    sample_tsr_int = round_to_pos_neg_one(x_out, eps=1e-1)
    sample_eval_parity = parity_func(sample_tsr_int.flatten(1), axis=1)
    sample_num = len(sample_eval_parity)
    even_parity_num = th.sum(sample_eval_parity == 0).item()
    odd_parity_num = th.sum(sample_eval_parity == 1).item()
    nan_num = th.sum(th.isnan(sample_eval_parity)).item()
    print(f"epoch: {epoch:06d} | Even parity: {even_parity_num}, Odd parity: {odd_parity_num} nan: {nan_num}| total: {sample_num}")
    


device = get_device()
# # Xtsr_raw, imgsize, imgchannels = load_raw_dataset(dataset_name)
# sample_num = 4096
# sample_len = 36
# group_size = 36
# parity = 0
imgsize = int(math.sqrt(sample_len))
imgchannels = 1
parity_str = "even" if parity == 0 else "odd"
x = sample_ensuring_uniqueness(N=sample_num, sample_func=lambda N: sample_group_parity_vec(N=N, sample_len=sample_len, group_size=group_size, parity=parity))
dataset_name = f"parity_N{sample_num}_D{sample_len}_G{group_size}_{parity_str}"
print(dataset_name, "dataset")
Xtsr = torch.from_numpy(x).float()
Xtsr = Xtsr.view(sample_num, imgchannels, imgsize, imgsize,)
th.save(Xtsr, f"{savedir}/training_data_tsr.pt")
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
    num_classes=0,  # No class conditioning
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
# %%



