#!/usr/bin/env python3
"""
CLI version of parity sample memorization evaluation.
Refactored from notebooks/20250804_parity_sample_memorization_eval.ipynb

This script evaluates memorization and parity correctness for a given experiment.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm
from easydict import EasyDict as edict
from core.parity_lib import parity_func, round_to_pos_neg_one
from circuit_toolkit import saveallforms


def get_all_available_steps(sample_dir, excluded_steps=[]):
    """Get all available step numbers from sample files in the directory."""
    sample_files = [f for f in os.listdir(sample_dir) if f.startswith("samples_epoch_") and f.endswith(".pt")]
    steps = []
    for f in sample_files:
        step_str = f.replace("samples_epoch_", "").replace(".pt", "")
        steps.append(int(step_str))
    steps = sorted(steps)
    steps = [step for step in steps if step not in excluded_steps]
    return steps


def compute_membership_counts(train_int: th.Tensor,
                              gen_int: th.Tensor,
                              group_size: int):
    """
    Args:
        train_int:    Tensor of shape [N_train, D], values in {−1, +1}
        gen_int:      Tensor of shape [N_gen, D], values in {−1, +1}
        group_size:   int, width of each bit‐group (must divide D)

    Returns a dict with:
        sample_mem_num    – how many full samples appear in the train set
        sample_mem_ratio  – sample_mem_num / N_gen
        bitgroup_mem_num  – how many bit‐groups appear in the train set
        bitgroup_mem_ratio– bitgroup_mem_num / (N_gen * D / group_size)
    """
    # ---- full‐sample codes ----
    N_gen, D = gen_int.shape

    # map −1→0, +1→1
    train_bits = (train_int > 0).long()  # [N_train, D]
    gen_bits   = (gen_int   > 0).long()  # [N_gen,   D]

    # build powers‐of‐two vector [1,2,4,…,2^(D−1)]
    weights = 1 << th.arange(D, device=train_int.device, dtype=th.long)

    # pack each row into a single integer code
    train_codes = (train_bits * weights).sum(dim=1).unique()
    gen_codes   = (gen_bits   * weights).sum(dim=1)

    # vectorized membership test
    mask = th.isin(gen_codes, train_codes)
    sample_mem_num   = int(mask.sum().item())
    sample_mem_ratio = sample_mem_num / N_gen

    # ---- bit‐group codes ----
    # reshape into rows of width `group_size`
    BG_train = train_bits.view(-1, group_size)  # [N_train * (D/group_size), group_size]
    BG_gen   = gen_bits.view(-1,   group_size)  # [N_gen   * (D/group_size), group_size]

    # powers‐of‐two for each group
    gw = 1 << th.arange(group_size, device=train_int.device, dtype=th.long)

    train_group_codes = (BG_train * gw).sum(dim=1).unique()
    gen_group_codes   = (BG_gen   * gw).sum(dim=1)

    mask2 = th.isin(gen_group_codes, train_group_codes)
    bitgroup_mem_num   = int(mask2.sum().item())
    bitgroup_mem_ratio = bitgroup_mem_num / gen_group_codes.numel()

    return {
        "sample_mem_num":    sample_mem_num,
        "sample_mem_ratio":  sample_mem_ratio,
        "bitgroup_mem_num":  bitgroup_mem_num,
        "bitgroup_mem_ratio": bitgroup_mem_ratio,
    }


def plot_memorization_results(mem_eval_stats_df: pd.DataFrame,
                             exp_name: str,
                             train_sample_num: int,
                             sample_len: int,
                             group_size: int,
                             nlayer: int,
                             nhead: int,
                             synopsis_dir: str,
                             savedir: str):
    """
    Generate and save memorization evaluation plots.
    
    Args:
        mem_eval_stats_df: DataFrame with evaluation statistics
        exp_name: Name of the experiment
        train_sample_num: Number of training samples
        sample_len: Length of each sample in bits
        group_size: Size of each bit group
        nlayer: Number of transformer layers
        nhead: Number of attention heads
        synopsis_dir: Directory to save summary plots
        savedir: Experiment directory
    """
    if len(mem_eval_stats_df) == 0:
        print("Warning: No data to plot")
        return
    
    num_groups = sample_len // group_size
    
    plt.figure(figsize=(6, 5))
    
    # Plot parity correctness
    sns.lineplot(mem_eval_stats_df, x="step", y="pergroup_parity_acc", 
                label="Per group parity correctness", alpha=0.7)
    sns.lineplot(mem_eval_stats_df, x="step", y="sample_corr_acc", 
                label="Sample correctness", alpha=0.7)
    
    # Plot baselines
    plt.axhline(y=0.5, linestyle=":", label="Per group accuracy baseline", 
               color="C0", alpha=0.7)
    plt.axhline(y=0.5 ** num_groups, linestyle=":", label="Sample accuracy baseline", 
               color="C1", alpha=0.7)
    
    # Plot NaN ratios
    sns.lineplot(mem_eval_stats_df, x="step", y="nan_ratio_eps_1e-1", 
                label="Exceed EPS ratio (eps=1e-1)", alpha=0.5, linestyle="--")
    sns.lineplot(mem_eval_stats_df, x="step", y="nan_ratio_eps_1e-2", 
                label="Exceed EPS ratio (eps=1e-2)", alpha=0.5, linestyle="--")
    
    # Plot memorization ratios
    sns.lineplot(mem_eval_stats_df, x="step", y="sample_mem_ratio", 
                label="Sample memorization ratio", alpha=0.7, linestyle=":", color="C5")
    sns.lineplot(mem_eval_stats_df, x="step", y="bitgroup_mem_ratio", 
                label="Bit group memorization ratio", alpha=0.7, linestyle=":", color="C6")
    
    plt.title(f"Sample evaluation\n{exp_name}\nDiT {nlayer}L{nhead}H {train_sample_num} samples, "
             f"{sample_len} bits per sample, {group_size} bits per group")
    plt.xlabel("Training Step")
    plt.ylabel("Ratio")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    
    # Save plot
    os.makedirs(synopsis_dir, exist_ok=True)
    saveallforms([synopsis_dir, savedir], f"{exp_name}_sample_parity_memorization_eval")
    plt.show()
    plt.close()
    
    print(f"Saved plot to {synopsis_dir}/{exp_name}_sample_parity_memorization_eval.png")# /pdf
    print(f"Saved plot to {synopsis_dir}/{exp_name}_sample_parity_memorization_eval.pdf")# /pdf


def evaluate_experiment_memorization(exp_name: str, 
                                     saveroot: str = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                                     synopsis_dir: str = "/n/home12/binxuwang/Github/DiffusionAttnConsistency/figures",
                                     excluded_steps: list = [],
                                     save_plot: bool = True,
                                     skip_completed: bool = True,
                                     incremental: bool = False):
    """
    Evaluate memorization and parity correctness for a single experiment.
    
    Args:
        exp_name: Name of the experiment (e.g., 'DiT_mini_parity_N4096_D36_G12_even')
        saveroot: Root directory containing experiment results
        synopsis_dir: Directory to save summary plots
        excluded_steps: List of steps to exclude from evaluation
        save_plot: Whether to generate and save plots
        skip_completed: If True, skip steps already present in existing CSV
        incremental: If True, only process new steps and append to existing CSV
        
    Returns:
        pandas.DataFrame: Evaluation statistics for all steps
    """
    print(f"Evaluating experiment: {exp_name}")
    
    # Set up directories
    savedir = f"{saveroot}/{exp_name}"
    sample_dir = f"{savedir}/samples"
    ckpt_dir = f"{savedir}/ckpts"
    
    # Check if directories exist
    if not os.path.exists(savedir):
        raise FileNotFoundError(f"Experiment directory not found: {savedir}")
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Samples directory not found: {sample_dir}")
    
    # Load experiment metadata
    args = edict(json.load(open(f"{savedir}/args.json")))
    config = json.load(open(f"{savedir}/config.json"))
    
    # Extract experiment parameters
    train_sample_num = args.sample_num
    sample_len = args.sample_len
    group_size = args.group_size
    parity = args.parity
    num_groups = sample_len // group_size
    nlayer = config["depth"]
    nhead = config["num_heads"]
    
    print(f"Parameters: {train_sample_num} samples, {sample_len} bits per sample, {group_size} bits per group")
    
    # Load training data
    Xtsr = th.load(f"{savedir}/training_data_tsr.pt", weights_only=False)
    
    # Get all available evaluation steps
    all_steps = get_all_available_steps(sample_dir, excluded_steps=excluded_steps)
    if not all_steps:
        raise RuntimeError(f"No evaluation steps found in {sample_dir}")
    
    print(f"Found {len(all_steps)} evaluation steps: {min(all_steps)} to {max(all_steps)}")
    
    # Check for existing evaluation results
    csv_path = f"{savedir}/mem_eval_stats.csv"
    existing_df = None
    completed_steps = set()
    
    if skip_completed and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        completed_steps = set(existing_df['step'].values)
        print(f"Found existing results for {len(completed_steps)} steps")
        
        if incremental:
            # Only process steps not in existing CSV
            steps_to_process = [step for step in all_steps if step not in completed_steps]
            print(f"Incremental mode: processing {len(steps_to_process)} new steps")
        else:
            # Skip completed steps but recompute final results
            steps_to_process = [step for step in all_steps if step not in completed_steps]
            print(f"Skip mode: processing {len(steps_to_process)} missing steps")
    else:
        steps_to_process = all_steps
        print(f"Processing all {len(steps_to_process)} steps")
    
    # Initialize results list
    if incremental and existing_df is not None:
        mem_eval_stats = existing_df.to_dict('records')
        print(f"Starting with {len(mem_eval_stats)} existing results")
    else:
        mem_eval_stats = []
    
    # Evaluate each step
    if steps_to_process:
        pbar = tqdm(steps_to_process, desc=f"Evaluating {exp_name}")
    else:
        pbar = []
        print("No new steps to process")
    
    for step in pbar:
        try:
            # Load generated samples
            sample_tsr = th.load(f"{sample_dir}/samples_epoch_{step:06d}.pt", weights_only=False)
            eval_sample_num = sample_tsr.shape[0]
            
            # Process samples with different epsilon values
            eps = 1e-2
            sample_tsr_int = round_to_pos_neg_one(sample_tsr, eps=eps)
            exist_nan = th.sum(sample_tsr_int.flatten(start_dim=1), dim=1)
            nan_num_eps_1e_2 = th.sum(th.isnan(exist_nan)).item()
            nan_ratio_eps_1e_2 = nan_num_eps_1e_2 / eval_sample_num
            
            eps = 1e-1
            sample_tsr_int = round_to_pos_neg_one(sample_tsr, eps=eps)
            exist_nan = th.sum(sample_tsr_int.flatten(start_dim=1), dim=1)
            nan_num_eps_1e_1 = th.sum(th.isnan(exist_nan)).item()
            nan_ratio_eps_1e_1 = nan_num_eps_1e_1 / eval_sample_num
            
            # Evaluate parity correctness
            sample_pergroup_int = sample_tsr_int.reshape(eval_sample_num, num_groups, group_size)
            sample_eval_parity = parity_func(sample_pergroup_int, axis=-1)
            
            # Count parity correctness for each group
            pergroup_parity_correctness = th.sum(sample_eval_parity == parity).sum()
            pergroup_parity_correctness_ratio = pergroup_parity_correctness / (eval_sample_num * num_groups)
            
            # Count sample correct only when all groups are correct
            sample_correctness = th.all(sample_eval_parity == parity, dim=1).sum()
            sample_correctness_ratio = sample_correctness / eval_sample_num
            
            # Compute memorization statistics
            gen_samples_mat = sample_tsr_int.to(int).flatten(start_dim=1)
            sample_mem_stats = compute_membership_counts(Xtsr.flatten(start_dim=1).cpu(), 
                                                       gen_samples_mat.cpu(), group_size)
            
            # Update progress bar
            pbar.set_description(
                f"Step {step}: mem_ratio {sample_mem_stats['sample_mem_ratio']:.3f}, "
                f"bitgroup_mem_ratio {sample_mem_stats['bitgroup_mem_ratio']:.3f}, "
                f"parity_acc {pergroup_parity_correctness_ratio:.3f}, "
                f"sample_acc {sample_correctness_ratio:.3f}"
            )
            
            # Store statistics
            mem_eval_stats.append({
                "step": step,
                **sample_mem_stats,
                "sample_num": eval_sample_num,
                "group_size": group_size,
                "nan_num_eps_1e-1": nan_num_eps_1e_1,
                "nan_ratio_eps_1e-1": nan_ratio_eps_1e_1,
                "nan_num_eps_1e-2": nan_num_eps_1e_2,
                "nan_ratio_eps_1e-2": nan_ratio_eps_1e_2,
                "pergroup_parity_num": pergroup_parity_correctness.item(),
                "pergroup_parity_acc": pergroup_parity_correctness_ratio.item(),
                "sample_corr_num": sample_correctness.item(),
                "sample_corr_acc": sample_correctness_ratio.item(),
            })
            
        except Exception as e:
            print(f"Warning: Failed to process step {step}: {e}")
            continue
    
    # Create DataFrame and save results
    mem_eval_stats_df = pd.DataFrame(mem_eval_stats)
    mem_eval_stats_df.to_csv(f"{savedir}/mem_eval_stats.csv", index=False)
    
    print(f"Saved evaluation statistics to {savedir}/mem_eval_stats.csv")
    
    # Generate and save plots if requested
    if save_plot:
        plot_memorization_results(
            mem_eval_stats_df=mem_eval_stats_df,
            exp_name=exp_name,
            train_sample_num=train_sample_num,
            sample_len=sample_len,
            group_size=group_size,
            nlayer=nlayer,
            nhead=nhead,
            synopsis_dir=synopsis_dir,
            savedir=savedir
        )
    
    return mem_eval_stats_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate parity sample memorization for an experiment')
    parser.add_argument('exp_name', type=str, help='Name of the experiment to evaluate')
    parser.add_argument('--saveroot', type=str, 
                       default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                       help='Root directory containing experiment results')
    parser.add_argument('--synopsis_dir', type=str,
                       default="/n/home12/binxuwang/Github/DiffusionAttnConsistency/figures",
                       help='Directory to save summary plots')
    parser.add_argument('--excluded_steps', type=int, nargs='*', default=[],
                       help='List of steps to exclude from evaluation')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--output_csv', type=str,
                       help='Path to save evaluation statistics CSV (default: <exp_dir>/mem_eval_stats.csv)')
    parser.add_argument('--no_skip_completed', action='store_true',
                       help='Disable skipping of already computed steps')
    parser.add_argument('--incremental', action='store_true',
                       help='Run in incremental mode - only process new steps and append to existing CSV')
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        df = evaluate_experiment_memorization(
            exp_name=args.exp_name,
            saveroot=args.saveroot,
            synopsis_dir=args.synopsis_dir,
            excluded_steps=args.excluded_steps,
            save_plot=not args.no_plot,
            skip_completed=not args.no_skip_completed,
            incremental=args.incremental
        )
        
        # Optionally save to custom location
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Evaluation statistics also saved to {args.output_csv}")
        
        print(f"Successfully evaluated experiment: {args.exp_name}")
        print(f"Processed {len(df)} evaluation steps")
        
    except Exception as e:
        print(f"Error evaluating experiment {args.exp_name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()