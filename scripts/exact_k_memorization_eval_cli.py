#!/usr/bin/env python3
"""
CLI version of exact-K sample memorization evaluation.
Adapted from scripts/parity_memorization_eval_cli.py

Evaluates memorization and exact-K correctness for a given experiment,
including a histogram of the ones count distribution across generated samples.
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch as th
from math import comb
from tqdm import tqdm
from easydict import EasyDict as edict
from core.exact_k_lib import exact_k_check, ones_count, round_to_pos_neg_one, valid_set_size, expected_memorization_ratio
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


def compute_exact_k_stats(train_int: th.Tensor,
                           gen_int: th.Tensor,
                           K: int):
    """
    Compute memorization and exact-K correctness statistics.

    Args:
        train_int : Tensor of shape [N_train, D], values in {-1, +1}
        gen_int   : Tensor of shape [N_gen, D], values in {-1, +1}
        K         : target number of ones

    Returns dict with:
        sample_mem_num, sample_mem_ratio  -- exact-match memorization
        k_correct_num, k_correct_ratio   -- fraction with exactly K ones
        mean_ones, std_ones              -- ones count distribution summary
        ones_histogram                   -- np.ndarray of length D+1
    """
    N_gen, D = gen_int.shape

    # Map -1 -> 0, +1 -> 1
    train_bits = (train_int > 0).long()   # [N_train, D]
    gen_bits   = (gen_int   > 0).long()   # [N_gen,   D]

    # --- Sample-level memorization (hashing trick) ---
    weights = 1 << th.arange(D, device=train_int.device, dtype=th.long)
    train_codes = (train_bits * weights).sum(dim=1).unique()
    gen_codes   = (gen_bits   * weights).sum(dim=1)
    mask = th.isin(gen_codes, train_codes)
    sample_mem_num   = int(mask.sum().item())
    sample_mem_ratio = sample_mem_num / N_gen

    # --- Exact-K correctness ---
    gen_np = gen_int.cpu().numpy()
    ones_counts = ones_count(gen_np)                  # shape (N_gen,)
    k_correct_mask = (ones_counts == K)
    k_correct_num   = int(k_correct_mask.sum())
    k_correct_ratio = k_correct_num / N_gen

    # --- Ones count distribution ---
    mean_ones = float(ones_counts.mean())
    std_ones  = float(ones_counts.std())
    ones_histogram = np.bincount(ones_counts, minlength=D + 1)  # length D+1

    return {
        "sample_mem_num":    sample_mem_num,
        "sample_mem_ratio":  sample_mem_ratio,
        "k_correct_num":     k_correct_num,
        "k_correct_ratio":   k_correct_ratio,
        "mean_ones":         mean_ones,
        "std_ones":          std_ones,
        "ones_histogram":    ones_histogram,   # np.ndarray, stored as separate columns in CSV
    }


def plot_eval_results(eval_stats_df: pd.DataFrame,
                      exp_name: str,
                      train_sample_num: int,
                      sample_len: int,
                      k_ones: int,
                      nlayer: int,
                      nhead: int,
                      synopsis_dir: str,
                      savedir: str):
    """Generate and save accuracy + memorization plots."""
    if len(eval_stats_df) == 0:
        print("Warning: No data to plot")
        return

    chance_acc = 1.0 / valid_set_size(sample_len, k_ones)
    exp_mem    = expected_memorization_ratio(train_sample_num, sample_len, k_ones)

    plt.figure(figsize=(7, 5))

    sns.lineplot(eval_stats_df, x="step", y="k_correct_ratio",
                 label=f"Exact-K accuracy (K={k_ones})", alpha=0.8)
    sns.lineplot(eval_stats_df, x="step", y="sample_mem_ratio",
                 label="Sample memorization ratio", alpha=0.7, linestyle=":", color="C5")
    sns.lineplot(eval_stats_df, x="step", y="nan_ratio_eps_1e-1",
                 label="Invalid ratio (eps=1e-1)", alpha=0.5, linestyle="--")
    sns.lineplot(eval_stats_df, x="step", y="nan_ratio_eps_1e-2",
                 label="Invalid ratio (eps=1e-2)", alpha=0.5, linestyle="--")

    plt.axhline(y=chance_acc, linestyle=":", color="C0", alpha=0.7,
                label=f"Chance accuracy (1/C({sample_len},{k_ones}))")
    plt.axhline(y=exp_mem, linestyle=":", color="C5", alpha=0.7,
                label=f"Expected mem ratio (N/C({sample_len},{k_ones}))")

    plt.title(f"Exact-K Evaluation\n{exp_name}\n"
              f"DiT {nlayer}L{nhead}H | N={train_sample_num} | D={sample_len} | K={k_ones}")
    plt.xlabel("Training Step")
    plt.ylabel("Ratio")
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(synopsis_dir, exist_ok=True)
    saveallforms([synopsis_dir, savedir], f"{exp_name}_sample_exactK_memorization_eval")
    plt.close()
    print(f"Saved plot to {synopsis_dir}/{exp_name}_sample_exactK_memorization_eval.png/pdf")


def evaluate_experiment(exp_name: str,
                        saveroot: str = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                        synopsis_dir: str = "/n/home12/binxuwang/Github/DiffusionAttnConsistency/figures",
                        excluded_steps: list = [],
                        save_plot: bool = True,
                        skip_completed: bool = True,
                        incremental: bool = False):
    """
    Evaluate exact-K memorization and accuracy for a single experiment.

    Returns:
        pandas.DataFrame with evaluation statistics for all steps
    """
    print(f"Evaluating experiment: {exp_name}")

    savedir    = f"{saveroot}/{exp_name}"
    sample_dir = f"{savedir}/samples"

    if not os.path.exists(savedir):
        raise FileNotFoundError(f"Experiment directory not found: {savedir}")
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Samples directory not found: {sample_dir}")

    args   = edict(json.load(open(f"{savedir}/args.json")))
    config = json.load(open(f"{savedir}/config.json"))

    train_sample_num = args.sample_num
    sample_len       = args.sample_len
    k_ones           = args.k_ones
    nlayer           = config["depth"]
    nhead            = config["num_heads"]

    print(f"Parameters: N={train_sample_num}, D={sample_len}, K={k_ones}")
    print(f"Valid set size: C({sample_len},{k_ones}) = {valid_set_size(sample_len, k_ones):,}")

    Xtsr = th.load(f"{savedir}/training_data_tsr.pt", weights_only=False)

    all_steps = get_all_available_steps(sample_dir, excluded_steps=excluded_steps)
    if not all_steps:
        raise RuntimeError(f"No evaluation steps found in {sample_dir}")
    print(f"Found {len(all_steps)} evaluation steps: {min(all_steps)} to {max(all_steps)}")

    csv_path = f"{savedir}/exactK_eval_stats.csv"
    existing_df = None
    completed_steps = set()

    if skip_completed and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        completed_steps = set(existing_df['step'].values)
        print(f"Found existing results for {len(completed_steps)} steps")
        steps_to_process = [step for step in all_steps if step not in completed_steps]
        if incremental:
            print(f"Incremental mode: processing {len(steps_to_process)} new steps")
        else:
            print(f"Skip mode: processing {len(steps_to_process)} missing steps")
    else:
        steps_to_process = all_steps
        print(f"Processing all {len(steps_to_process)} steps")

    if incremental and existing_df is not None:
        eval_stats = existing_df.to_dict('records')
        print(f"Starting with {len(eval_stats)} existing results")
    else:
        eval_stats = []

    pbar = tqdm(steps_to_process, desc=f"Evaluating {exp_name}") if steps_to_process else []

    for step in pbar:
        try:
            sample_tsr = th.load(f"{sample_dir}/samples_epoch_{step:06d}.pt", weights_only=False)
            eval_sample_num = sample_tsr.shape[0]

            # Invalid sample ratios at two eps thresholds
            eps = 1e-2
            sample_int_1e2 = round_to_pos_neg_one(sample_tsr, eps=eps)
            exist_nan_1e2  = th.sum(sample_int_1e2.flatten(start_dim=1), dim=1)
            nan_num_eps_1e_2   = int(th.sum(th.isnan(exist_nan_1e2)).item())
            nan_ratio_eps_1e_2 = nan_num_eps_1e_2 / eval_sample_num

            eps = 1e-1
            sample_int_1e1 = round_to_pos_neg_one(sample_tsr, eps=eps)
            exist_nan_1e1  = th.sum(sample_int_1e1.flatten(start_dim=1), dim=1)
            nan_num_eps_1e_1   = int(th.sum(th.isnan(exist_nan_1e1)).item())
            nan_ratio_eps_1e_1 = nan_num_eps_1e_1 / eval_sample_num

            # Use eps=1e-1 binarization for all further evaluation
            gen_int = sample_int_1e1.flatten(start_dim=1)

            # Replace NaN entries with 0 for counting (they'll be excluded via nan_mask)
            nan_rows = th.isnan(gen_int).any(dim=1)
            gen_int_filled = gen_int.clone()
            gen_int_filled[nan_rows] = 0

            train_int = Xtsr.flatten(start_dim=1).cpu()

            stats = compute_exact_k_stats(train_int, gen_int_filled.cpu(), k_ones)

            # Flatten histogram into per-column entries
            hist = stats.pop("ones_histogram")   # np.ndarray length D+1
            hist_dict = {f"ones_hist_{j}": int(hist[j]) for j in range(len(hist))}

            row = {
                "step":              step,
                "sample_num":        eval_sample_num,
                "nan_num_eps_1e-1":  nan_num_eps_1e_1,
                "nan_ratio_eps_1e-1": nan_ratio_eps_1e_1,
                "nan_num_eps_1e-2":  nan_num_eps_1e_2,
                "nan_ratio_eps_1e-2": nan_ratio_eps_1e_2,
                **stats,
                **hist_dict,
            }
            eval_stats.append(row)

            if hasattr(pbar, 'set_description'):
                pbar.set_description(
                    f"Step {step}: k_acc={stats['k_correct_ratio']:.3f} "
                    f"mem={stats['sample_mem_ratio']:.3f} "
                    f"mean_ones={stats['mean_ones']:.1f}"
                )

        except Exception as e:
            print(f"Warning: Failed to process step {step}: {e}")
            continue

    eval_stats_df = pd.DataFrame(eval_stats)
    eval_stats_df.to_csv(csv_path, index=False)
    print(f"Saved evaluation statistics to {csv_path}")

    if save_plot:
        plot_eval_results(
            eval_stats_df=eval_stats_df,
            exp_name=exp_name,
            train_sample_num=train_sample_num,
            sample_len=sample_len,
            k_ones=k_ones,
            nlayer=nlayer,
            nhead=nhead,
            synopsis_dir=synopsis_dir,
            savedir=savedir,
        )

    return eval_stats_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate exact-K sample memorization for an experiment')
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
    parser.add_argument('--no_skip_completed', action='store_true',
                        help='Disable skipping of already computed steps')
    parser.add_argument('--incremental', action='store_true',
                        help='Only process new steps and append to existing CSV')

    args = parser.parse_args()

    try:
        df = evaluate_experiment(
            exp_name=args.exp_name,
            saveroot=args.saveroot,
            synopsis_dir=args.synopsis_dir,
            excluded_steps=args.excluded_steps,
            save_plot=not args.no_plot,
            skip_completed=not args.no_skip_completed,
            incremental=args.incremental,
        )
        print(f"Successfully evaluated experiment: {args.exp_name}")
        print(f"Processed {len(df)} evaluation steps")

    except Exception as e:
        print(f"Error evaluating experiment {args.exp_name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
