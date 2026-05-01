#!/usr/bin/env python3
"""
Attractor basin measurement CLI.

Measures the denoiser basin width along three directions from training samples:
  1. Nearest Hamming-1 invalid neighbor (breaks rule)
  2. Nearest Hamming-2 valid-novel neighbor (preserves rule, not in train set)
  3. Nearest other training sample (by Hamming distance)

Results are saved as NPZ files per checkpoint and can be plotted for comparison.

Usage
-----
  python scripts/measure_attractor_basin.py \\
    --exp_name DiT_mini_G3_rep2 \\
    --epochs 58780 492388 \\
    --sigma 1.0 \\
    --n_samples 50 \\
    --device cuda

  # Multiple sigmas:
  python scripts/measure_attractor_basin.py \\
    --exp_name DiT_mini_G3_rep2 \\
    --epochs 58780 492388 \\
    --sigma 0.5 1.0 2.0 \\
    --n_samples 50

Output
------
  {exp_dir}/basin_analysis/basin_ep{epoch:06d}_sig{sigma:.4f}_N{n}.npz
    Contains: per_sample results, mean_profiles, summary, t_vals

  {exp_dir}/basin_analysis/basin_ep{epoch:06d}_sig{sigma:.4f}_N{n}_summary.json
    Scalar summary statistics for quick inspection.
"""

import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, "/n/home12/binxuwang/Github/DiffusionAttnConsistency")

from core.vector_field_lib import load_model, load_training_data, DEFAULT_SAVEROOT
from core.basin_lib import (
    load_rule_params,
    measure_basin_batch,
)


def parse_args():
    p = argparse.ArgumentParser(description="Measure attractor basin widths along three directions.")
    p.add_argument("--exp_name", type=str, required=True,
                   help="Experiment name (subfolder of saveroot)")
    p.add_argument("--epochs", type=int, nargs="+", required=True,
                   help="Checkpoint epoch(s) to evaluate")
    p.add_argument("--sigma", type=float, nargs="+", default=[1.0],
                   help="Noise level(s) σ to evaluate at (default: 1.0)")
    p.add_argument("--n_samples", type=int, default=50,
                   help="Number of training samples to average over (default: 50)")
    p.add_argument("--n_points", type=int, default=150,
                   help="Number of t-values along each line (default: 150)")
    p.add_argument("--t_min", type=float, default=-0.5,
                   help="Minimum t value (default: -0.5)")
    p.add_argument("--t_max", type=float, default=2.0,
                   help="Maximum t value (default: 2.0)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device (default: cpu)")
    p.add_argument("--saveroot", type=str, default=DEFAULT_SAVEROOT,
                   help="Root directory for experiments")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable per-line caching (recompute everything)")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Print progress (default: True)")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = os.path.join(args.saveroot, args.exp_name)
    if not os.path.isdir(exp_dir):
        print(f"ERROR: exp_dir not found: {exp_dir}")
        sys.exit(1)

    # Load training data and rule params
    print(f"Loading training data from {exp_dir} ...")
    x_train_t = load_training_data(args.exp_name, saveroot=args.saveroot)  # (N, D) torch float32
    x_train = x_train_t.numpy()  # (N, D) numpy float32 {-1,+1}
    rule_params = load_rule_params(exp_dir)
    print(f"  x_train shape: {x_train.shape}  rule_params: {rule_params}")

    # Build train code set for novelty check
    train_codes = set()
    for row in (x_train > 0).astype(np.int8):
        n = len(row)
        train_codes.add(int(sum(int(row[i]) << i for i in range(n))))

    output_dir = os.path.join(exp_dir, "basin_analysis")
    os.makedirs(output_dir, exist_ok=True)

    t_range = (args.t_min, args.t_max)

    for epoch in args.epochs:
        print(f"\n{'='*60}")
        print(f"Checkpoint epoch {epoch}")
        model, _, _ = load_model(args.exp_name, epoch, device=args.device,
                                  saveroot=args.saveroot)
        model.eval()

        for sigma in args.sigma:
            tag = f"ep{epoch:06d}_sig{sigma:.4f}_N{args.n_samples}"
            out_npz  = os.path.join(output_dir, f"basin_{tag}.npz")
            out_json = os.path.join(output_dir, f"basin_{tag}_summary.json")

            if os.path.exists(out_npz) and not args.no_cache:
                print(f"  [CACHED] σ={sigma:.4f} → {out_npz}")
                continue

            print(f"  σ={sigma:.4f} — measuring {args.n_samples} samples ...")

            # Per-line cache dir (individual line NPZs)
            line_cache_dir = os.path.join(output_dir, "line_cache") if not args.no_cache else None
            cache_prefix   = f"ep{epoch:06d}_sig{sigma:.4f}" if line_cache_dir else None

            result = measure_basin_batch(
                model=model,
                sigma=sigma,
                x_train=x_train,
                train_codes=train_codes,
                rule_params=rule_params,
                n_samples=args.n_samples,
                n_points=args.n_points,
                t_range=t_range,
                device=args.device,
                cache_dir=line_cache_dir,
                cache_prefix=cache_prefix,
                verbose=args.verbose,
            )

            # Save aggregated NPZ (exclude per_sample to keep file small)
            save_dict = dict(
                t_vals=result['t_vals'],
                sigma=np.float32(result['sigma']),
                n_samples=np.int32(result['n_samples']),
            )
            for direction in ('invalid', 'valid_novel', 'other_train'):
                for metric in ('exact_match', 'bit_agreement', 'dist_from_start', 'proj_pull'):
                    save_dict[f"{direction}_{metric}_mean"] = result['mean_profiles'][direction][metric]
                    save_dict[f"{direction}_{metric}_std"]  = result['mean_profiles'][direction][f"{metric}_std"]
            np.savez_compressed(out_npz, **save_dict)

            # Save scalar summary JSON
            summary_json = {}
            for direction, stats in result['summary'].items():
                summary_json[direction] = stats
            with open(out_json, 'w') as f:
                json.dump(summary_json, f, indent=2)

            print(f"  Saved → {out_npz}")
            print(f"  Summary:")
            for direction, stats in result['summary'].items():
                bw = stats['basin_width_l2_mean']
                bw_std = stats['basin_width_l2_std']
                print(f"    {direction:12s}: basin_width_l2 = {bw:.3f} ± {bw_std:.3f} "
                      f"  (t: {stats['basin_width_t_mean']:.3f} ± {stats['basin_width_t_std']:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
