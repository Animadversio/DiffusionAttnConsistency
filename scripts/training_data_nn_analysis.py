#!/usr/bin/env python3
"""
Training Data Nearest Neighbor Analysis

This script loads training data for different experiments and computes
the average distance between nearest neighbors to understand data density
and distribution characteristics.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from easydict import EasyDict as edict


def compute_nn_distances(data: th.Tensor, k: int = 1, metric: str = 'euclidean'):
    """
    Compute nearest neighbor distances for the training data.
    
    Args:
        data: Tensor of shape [N, D] containing training samples
        k: Number of nearest neighbors to consider (default: 1)
        metric: Distance metric ('euclidean', 'hamming', 'manhattan')
        
    Returns:
        dict: Statistics about nearest neighbor distances
    """
    N, D = data.shape
    print(f"Computing {k}-NN distances for {N} samples of dimension {D}")
    
    # Convert to numpy for sklearn
    data_np = data.cpu().numpy()
    
    # For binary data, use hamming distance
    if metric == 'hamming' or th.all((data == -1) | (data == 1)):
        # Convert {-1, 1} to {0, 1} for hamming distance
        if th.all((data == -1) | (data == 1)):
            data_np = (data_np + 1) / 2
        metric = 'hamming'
        print("Using Hamming distance for binary data")
    
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nbrs.fit(data_np)
    
    # Find nearest neighbors (k+1 because first neighbor is the point itself)
    distances, indices = nbrs.kneighbors(data_np)
    
    # Remove self-distances (first column)
    nn_distances = distances[:, 1:k+1]
    
    # Compute statistics
    mean_nn_dist = np.mean(nn_distances)
    std_nn_dist = np.std(nn_distances)
    min_nn_dist = np.min(nn_distances)
    max_nn_dist = np.max(nn_distances)
    median_nn_dist = np.median(nn_distances)
    
    # For binary data with Hamming distance, convert back to bit differences
    if metric == 'hamming':
        mean_nn_dist *= D
        std_nn_dist *= D
        min_nn_dist *= D
        max_nn_dist *= D
        median_nn_dist *= D
        print(f"Converted Hamming distances to bit differences (multiply by {D})")
    
    return {
        'mean_nn_distance': mean_nn_dist,
        'std_nn_distance': std_nn_dist,
        'min_nn_distance': min_nn_dist,
        'max_nn_distance': max_nn_dist,
        'median_nn_distance': median_nn_dist,
        'num_samples': N,
        'dimension': D,
        'k_neighbors': k,
        'metric': metric
    }


def analyze_experiment_training_data(exp_name: str,
                                   saveroot: str = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                                   k: int = 1,
                                   metric: str = 'auto'):
    """
    Analyze nearest neighbor distances for a single experiment's training data.
    
    Args:
        exp_name: Name of the experiment
        saveroot: Root directory containing experiment results
        k: Number of nearest neighbors to consider
        metric: Distance metric ('auto', 'euclidean', 'hamming', 'manhattan')
        
    Returns:
        dict: Analysis results including NN distance statistics
    """
    print(f"Analyzing training data for experiment: {exp_name}")
    
    # Set up directories
    savedir = f"{saveroot}/{exp_name}"
    
    # Check if experiment directory exists
    if not os.path.exists(savedir):
        raise FileNotFoundError(f"Experiment directory not found: {savedir}")
    
    # Load experiment metadata
    args = edict(json.load(open(f"{savedir}/args.json")))
    
    # Extract experiment parameters
    train_sample_num = args.sample_num
    sample_len = args.sample_len
    group_size = args.group_size
    parity = args.parity
    
    print(f"Parameters: {train_sample_num} samples, {sample_len} bits per sample, {group_size} bits per group")
    
    # Load training data
    training_data_path = f"{savedir}/training_data_tsr.pt"
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data not found: {training_data_path}")
    
    Xtsr = th.load(training_data_path, weights_only=False)
    print(f"Loaded training data shape: {Xtsr.shape}")
    
    # Flatten data for NN analysis
    data_flat = Xtsr.flatten(start_dim=1)
    
    # Auto-select metric based on data type
    if metric == 'auto':
        if th.all((Xtsr == -1) | (Xtsr == 1)):
            metric = 'hamming'
        else:
            metric = 'euclidean'
    
    # Compute nearest neighbor distances
    nn_stats = compute_nn_distances(data_flat, k=k, metric=metric)
    
    # Add experiment metadata
    result = {
        'exp_name': exp_name,
        'train_sample_num': train_sample_num,
        'sample_len': sample_len,
        'group_size': group_size,
        'parity': parity,
        **nn_stats
    }
    
    return result


def analyze_multiple_experiments(exp_names: list,
                                saveroot: str = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                                k: int = 1,
                                metric: str = 'auto',
                                output_csv: str = None):
    """
    Analyze nearest neighbor distances for multiple experiments.
    
    Args:
        exp_names: List of experiment names to analyze
        saveroot: Root directory containing experiment results
        k: Number of nearest neighbors to consider
        metric: Distance metric
        output_csv: Path to save results CSV
        
    Returns:
        pandas.DataFrame: Analysis results for all experiments
    """
    results = []
    
    for exp_name in tqdm(exp_names, desc="Analyzing experiments"):
        try:
            result = analyze_experiment_training_data(
                exp_name=exp_name,
                saveroot=saveroot,
                k=k,
                metric=metric
            )
            results.append(result)
            print(f"✓ {exp_name}: mean NN distance = {result['mean_nn_distance']:.3f}")
            
        except Exception as e:
            print(f"✗ Failed to analyze {exp_name}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
    
    return df


def find_all_experiments(saveroot: str):
    """Find all experiment directories in the saveroot."""
    if not os.path.exists(saveroot):
        return []
    
    experiments = []
    for item in os.listdir(saveroot):
        exp_dir = os.path.join(saveroot, item)
        if os.path.isdir(exp_dir):
            # Check if it has training data and args
            if (os.path.exists(f"{exp_dir}/training_data_tsr.pt") and 
                os.path.exists(f"{exp_dir}/args.json")):
                experiments.append(item)
    
    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(description='Analyze nearest neighbor distances in training data')
    parser.add_argument('exp_names', nargs='*', help='Experiment names to analyze (leave empty to analyze all)')
    parser.add_argument('--saveroot', type=str,
                       default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning",
                       help='Root directory containing experiment results')
    parser.add_argument('--k', type=int, default=1,
                       help='Number of nearest neighbors to consider')
    parser.add_argument('--metric', type=str, default='auto',
                       choices=['auto', 'euclidean', 'hamming', 'manhattan'],
                       help='Distance metric to use')
    parser.add_argument('--output_csv', type=str,
                       help='Path to save results CSV')
    parser.add_argument('--list_experiments', action='store_true',
                       help='List all available experiments and exit')
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list_experiments:
        experiments = find_all_experiments(args.saveroot)
        print(f"Found {len(experiments)} experiments in {args.saveroot}:")
        for exp in experiments:
            print(f"  {exp}")
        return
    
    # Determine which experiments to analyze
    if args.exp_names:
        exp_names = args.exp_names
    else:
        exp_names = find_all_experiments(args.saveroot)
        print(f"No experiments specified, analyzing all {len(exp_names)} experiments")
    
    if not exp_names:
        print("No experiments to analyze")
        return
    
    try:
        # Analyze experiments
        df = analyze_multiple_experiments(
            exp_names=exp_names,
            saveroot=args.saveroot,
            k=args.k,
            metric=args.metric,
            output_csv=args.output_csv
        )
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Analyzed {len(df)} experiments")
        print(f"\nNearest neighbor distance statistics:")
        print(f"  Mean NN distance: {df['mean_nn_distance'].mean():.3f} ± {df['mean_nn_distance'].std():.3f}")
        print(f"  Min NN distance:  {df['min_nn_distance'].mean():.3f} ± {df['min_nn_distance'].std():.3f}")
        print(f"  Max NN distance:  {df['max_nn_distance'].mean():.3f} ± {df['max_nn_distance'].std():.3f}")
        
        # Group by key parameters if multiple experiments
        if len(df) > 1:
            print(f"\nGrouped by sample size:")
            grouped = df.groupby('train_sample_num')['mean_nn_distance'].agg(['mean', 'std', 'count'])
            print(grouped)
            
            print(f"\nGrouped by dimension:")
            grouped = df.groupby('sample_len')['mean_nn_distance'].agg(['mean', 'std', 'count'])
            print(grouped)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()