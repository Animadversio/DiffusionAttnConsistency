
%load_ext autoreload
%autoreload 2

# %%
import re
import os
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from tqdm.auto import tqdm, trange
from einops import rearrange, repeat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("/n/home12/binxuwang/Github/DiT")
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from copy import deepcopy
from collections import defaultdict
from circuit_toolkit.layer_hook_utils import featureFetcher_module
from circuit_toolkit.plot_utils import to_imgrid, saveallforms
from core.attention_analysis_lib import *
# %% [markdown]
# ### Loading and sampling from trained models

def vae_decode_batch(vae, latents, batch_size=8):
    samples = []
    for i in trange(0, len(latents), batch_size):
        samples.append(vae.decode(latents[i:i+batch_size].to(vae.device) / 0.18215).sample)
    pred_x0_traj = torch.cat(samples, dim=0)
    pred_x0_traj = (0.5 * pred_x0_traj + 0.5).clamp(0, 1)
    return pred_x0_traj

import re
def extract_dit_segment(s: str) -> str:
    """
    Extracts the substring starting at 'DiT' and ending just before '2025'.
    Returns an empty string if no match is found.
    """
    m = re.search(r'(DiT.*?)(?=2025)', s)
    return m.group(1) if m else ''

def extract_dit_version(s: str) -> str:
    """
    Finds the first occurrence of 'DiT-X-Y' (two hyphen-separated fields after 'DiT')
    """
    m = re.search(r'(DiT-[^-]+-[^-]+)', s)
    return m.group(1) if m else ''


def replace_second_dash(s: str) -> str:
    # split into at most 3 parts
    parts = s.split('-', 2)
    # if there are exactly 3 parts, re-join with a slash for the second separator
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]}/{parts[2]}"
    return s  # fallback if unexpected format
# %%
device = "cuda"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
vae.eval()
vae.to(device)
vae.requires_grad_(False);
#%%
# %%
# exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionAttnConsistency/DiT/flowers_results/006-flower_latents10k_pilots-uncond-DiT-S-1-flower_latents10k_pilots_20250625-0351"
result_root = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionAttnConsistency/DiT/flowers_results"
expname = "010-flower_latents10k_pilots-uncond-DiT-S-1-flower_latents10k_pilots_seed43_20250625-1754"
expname = "007-flower_latents10k_pilots-uncond-DiT-mini-1-flower_latents10k_pilots_20250625-0355"
for expname in [
    # "007-flower_latents10k_pilots-uncond-DiT-mini-1-flower_latents10k_pilots_20250625-0355",
    "004-flower_latents10k_pilots-uncond-DiT-S-2-flower_latents10k_pilots_20250624-1454",
    "005-flower_latents10k_pilots-uncond-DiT-S-1-flower_latents10k_pilots_20250624-1512",
    "006-flower_latents10k_pilots-uncond-DiT-S-1-flower_latents10k_pilots_20250625-0351",
    "008-flower_latents10k_pilots-uncond-DiT-micro-1-flower_latents10k_pilots_20250625-0706",
    "009-flower_latents10k_pilots-uncond-DiT-nano-1-flower_latents10k_pilots_20250625-0847",
    "010-flower_latents10k_pilots-uncond-DiT-S-1-flower_latents10k_pilots_seed43_20250625-1754",
]:
    print(f"Processing {expname}")
    exproot = join(result_root, expname)
    ckptdir = join(exproot, "checkpoints")
    sample_dir = join(exproot, "samples")
    figdir = join(exproot, "figures")
    os.makedirs(figdir, exist_ok=True)
    dit_version = replace_second_dash(extract_dit_version(expname))
    expname_short = extract_dit_segment(expname)

    latent_size = 16
    num_classes = 0
    class_dropout_prob = 1.0
    model = DiT_models[dit_version](
            input_size=latent_size,
            in_channels=4,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob
        )

    model.load_state_dict(torch.load(join(ckptdir, "0070000.pt"))["ema"]);
    model.to(device)
    model.eval()
    model.requires_grad_(False);

    # %%
    toggle_fused_attn(model, fused_attn=False);
    fetcher = featureFetcher_module()
    for blocki, block in enumerate(model.blocks):
        fetcher.record_module(block.attn.attn_drop, f"attn_{blocki}", store_device="cpu")
    # %%
    n_steps = 20
    diffusion_eval = create_diffusion(timestep_respacing=f"ddim{n_steps}")
    final = None
    batch_size = 16
    z = torch.randn(batch_size, 4, 16, 16, device="cuda", 
                    generator=torch.Generator(device="cuda").manual_seed(0))
    model_kwargs = {"y": torch.zeros(batch_size, dtype=torch.int, device="cuda")}
    intermediate_Xt = []
    intermediate_pred_x0 = []
    intermediate_Xt.append(z.clone().cpu())
    attn_maps_organized = defaultdict(list)
    for sample in diffusion_eval.ddim_sample_loop_progressive(
        model.forward,
        z.shape,
        z,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=model_kwargs,
        device=device,
        progress=True,
        eta=0.0,
    ):
        final = sample
        intermediate_Xt.append(sample["sample"].cpu())
        intermediate_pred_x0.append(sample["pred_xstart"].cpu())
        # attn_maps.append(deepcopy(fetcher.activations))
        for key in fetcher.activations.keys():
            attn_maps_organized[key].append(fetcher.activations[key].clone().cpu())
    intermediate_Xt = torch.stack(intermediate_Xt)
    intermediate_pred_x0 = torch.stack(intermediate_pred_x0)
    attn_maps_organized = {key: torch.stack(attn_maps_organized[key]) for key in attn_maps_organized.keys()}
    images = vae_decode_batch(vae, final["sample"])
    to_imgrid(images)

    # %%
    for key in attn_maps_organized.keys():
        print(key, attn_maps_organized[key].shape) # (num_sampling_steps, batch_size, num_heads, num_tokens, num_tokens)
    attn_maps_stacked = torch.stack(list(attn_maps_organized.values()))
    print("attn_maps_stacked.shape:", attn_maps_stacked.shape) # (num_layers, num_sampling_steps, batch_size, num_heads, num_tokens, num_tokens)
    # make sure each attention map sums to 1
    assert torch.allclose(attn_maps_organized["attn_1"].sum(dim=-1), torch.ones(1))
    # make sure different time steps have different attn maps (not overwritten)
    assert not torch.allclose(attn_maps_organized["attn_2"][0], attn_maps_organized["attn_2"][-1])
    assert not torch.allclose(attn_maps_organized["attn_1"][0], attn_maps_organized["attn_2"][-1])

    
    # %% [markdown]
    # ## Compute some statistics of attention maps
    # %%
    attn_entropy = compute_entropy_last_dim(attn_maps_stacked, dim=-1)
    attn_entropy_token_mean = attn_entropy.mean(dim=-1)
    print(attn_entropy_token_mean.shape)
    fig = plot_attention_layer_head_heatmaps(attn_entropy_token_mean, title_str="Attention entropy | run seed 43", figsize=(12, 8), sample_idx=None, )
    saveallforms(figdir, "attn_entropy_token_sample_avg_seed0", fig)
    print("Computed attn entropy")
    # %%
    attn_top1_score = top_k_attention_score(attn_maps_stacked, k=1, dim=-1)
    attn_top1_score_token_mean = attn_top1_score.mean(dim=-1)
    fig = plot_attention_layer_head_heatmaps(attn_top1_score_token_mean, title_str="Attention top1 score | run seed 43", figsize=(12, 8), sample_idx=None, num_heads=6)
    saveallforms(figdir, "attn_top1_score_token_sample_avg_seed0", fig)
    print("Computed attn top1 score")
    # %%
    attn_top5_score = top_k_attention_score(attn_maps_stacked, k=5, dim=-1)
    attn_top5_score_token_mean = attn_top5_score.mean(dim=-1)
    fig = plot_attention_layer_head_heatmaps(attn_top5_score_token_mean, title_str="Attention top5 score | run seed 43", figsize=(12, 8), sample_idx=None, num_heads=6)
    saveallforms(figdir, "attn_top5_score_token_sample_avg_seed0", fig)
    print("Computed attn top5 score")
    # %%
    attn_local_rad2_score = local_attention_score(attn_maps_stacked, dist_type="L2", threshold=2.0)
    attn_local_rad2_score_token_mean = attn_local_rad2_score.mean(dim=-1)
    fig = plot_attention_layer_head_heatmaps(attn_local_rad2_score_token_mean, title_str="Attention local score (radius=2) | run seed 43", figsize=(12, 8), sample_idx=None, num_heads=6)
    saveallforms(figdir, "attn_local_rad2_score_token_sample_avg_seed0", fig)
    print("Computed attn local rad2 score")
    # %%
    attn_avg_dist = average_attention_distance(attn_maps_stacked, dist_type="L2")
    attn_avg_dist_token_mean = attn_avg_dist.mean(dim=-1)
    fig = plot_attention_layer_head_heatmaps(attn_avg_dist_token_mean, title_str="Attention L2 distance | run seed 43", figsize=(12, 8), sample_idx=None, num_heads=6)
    saveallforms(figdir, "attn_avg_dist_token_sample_avg_seed0", fig)
    print("Computed attn avg dist")
    # %%
    attn_weighted_var = attention_spatial_variance(attn_maps_stacked)
    attn_weighted_var_token_mean = attn_weighted_var.mean(dim=-1)
    attn_weighted_var_token_mean.shape
    fig = plot_attention_layer_head_heatmaps(attn_weighted_var_token_mean, title_str="Attention spatial variance | run seed 43", figsize=(12, 8), sample_idx=None, num_heads=6)
    saveallforms(figdir, "attn_weighted_var_token_sample_avg_seed0", fig)
    print("Computed attn weighted var")
    plt.close("all")

    # %% [markdown]
    # ### Visualize attention maps
    to_imgrid(images[6]).save(join(figdir, "attn_map_sample_6.png"))
    # %% [markdown]
    # #### Layer x Attention head
    n_tokens = attn_maps_stacked.shape[-1]
    map_shape = infer_spatial_shape(n_tokens)
    probe_token = 99 if n_tokens == 256 else int(n_tokens*0.4)
    plt.switch_backend("Agg")
    for step_idx in range(n_steps):
        figh = visualize_attn_maps(attn_maps_stacked, layer_idx=None, step_idx=step_idx, sample_idx=6, head_idx=None, token_idx=probe_token, row_dim="head", col_dim="layer", map_shape=map_shape);
        saveallforms(figdir, f"attn_map_sample_6_step{step_idx}_layer_x_head_all", figh)
        figh = visualize_attn_maps(attn_maps_stacked, layer_idx=None, step_idx=step_idx, sample_idx=6, head_idx=None, token_idx=probe_token, row_dim="layer", col_dim="head", map_shape=map_shape);
        saveallforms(figdir, f"attn_map_sample_6_step{step_idx}_head_x_layer_all", figh)
        print(f"exported attn map for step {step_idx} head x layer")
        plt.close(figh)
        plt.close("all")
    # %%
    for layer_idx in range(attn_maps_stacked.shape[0]):
        figh = visualize_attn_maps(attn_maps_stacked, layer_idx=layer_idx, step_idx=None, sample_idx=6, head_idx=None, token_idx=probe_token, row_dim="step", col_dim="head", map_shape=map_shape);
        saveallforms(figdir, f"attn_map_sample_6_layer{layer_idx}_step_x_head_all", figh)
        figh = visualize_attn_maps(attn_maps_stacked, layer_idx=layer_idx, step_idx=None, sample_idx=6, head_idx=None, token_idx=probe_token, row_dim="head", col_dim="step", map_shape=map_shape);
        saveallforms(figdir, f"attn_map_sample_6_layer{layer_idx}_head_x_step_all", figh)
        print(f"exported attn map for layer {layer_idx} head x step")
        plt.close(figh)
        plt.close("all")

    
