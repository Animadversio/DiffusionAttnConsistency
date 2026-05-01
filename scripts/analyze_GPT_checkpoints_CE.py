#!/usr/bin/env python3
"""
Analyze GPT parity-learning checkpoints: cross-entropy loss vs training step.

For each checkpoint, computes:
  - Mean CE loss on: training set, valid-novel set, boolean cube
  - Per-position CE loss (36 positions) on the same three splits

Results are saved as a single NPZ file per experiment for easy replotting.

Save format
-----------
  epochs                : (C,)      — training step at each checkpoint
  loss_train            : (C,)      — mean CE, training set
  loss_valid_novel      : (C,)      — mean CE, valid-novel test set
  loss_boolean_cube     : (C,)      — mean CE, boolean cube
  pos_loss_train        : (C, 36)   — per-position CE, training set
  pos_loss_valid_novel  : (C, 36)
  pos_loss_boolean_cube : (C, 36)
  args                  : JSON string of experiment config

Usage
-----
  python scripts/analyze_GPT_checkpoints_CE.py \
      --exp_names GPT_mini_parity_N4096_D36_G6_even_lr1e4 \
                  GPT_mini_parity_N4096_D36_G6_even_wd1e2 \
      --saveroot /n/holylfs06/LABS/.../DiffusionParityLearning \
      --n_eval 4096 --batch_size 512 --device cuda
"""

import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gpt_eval_lib import (
    load_gpt_checkpoint, list_checkpoints,
    compute_ce_loss, compute_per_position_loss,
    build_all_test_sets,
)

SAVEROOT_DEFAULT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_names", nargs="+", required=True)
    p.add_argument("--saveroot", default=SAVEROOT_DEFAULT)
    p.add_argument("--n_eval",   type=int, default=4096,
                   help="Number of samples per eval split")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device",   default="cuda")
    p.add_argument("--out_suffix", default="",
                   help="Optional suffix appended to output filename")
    return p.parse_args()


def analyze_experiment(exp_dir, n_eval, batch_size, device, out_suffix=""):
    print(f"\n{'='*60}")
    print(f"Experiment: {os.path.basename(exp_dir)}")
    print(f"{'='*60}")

    ckpts = list_checkpoints(exp_dir)
    if not ckpts:
        print("  No checkpoints found — skipping.")
        return

    print(f"  Found {len(ckpts)} checkpoints.")

    # Build eval sets once
    print("  Building evaluation sets…")
    sets = build_all_test_sets(exp_dir, n_samples=n_eval)
    split_names = ["train", "valid_novel", "boolean_cube"]

    # Result arrays
    epochs = np.array([step for step, _ in ckpts])
    C      = len(ckpts)
    seq_len = sets["train"].shape[1]   # 37 for D=36
    n_pos   = seq_len - 1              # 36

    loss_arrays     = {s: np.full(C, np.nan) for s in split_names}
    pos_loss_arrays = {s: np.full((C, n_pos), np.nan) for s in split_names}

    with open(os.path.join(exp_dir, "args.json")) as f:
        args_json = json.load(f)

    for idx, (step, ckpt_path) in enumerate(tqdm(ckpts, desc="  Checkpoints")):
        model = load_gpt_checkpoint(ckpt_path, exp_dir, device=device)

        for sname in split_names:
            tokens = sets[sname]
            loss_arrays[sname][idx] = compute_ce_loss(
                model, tokens, device, batch_size=batch_size
            )
            pos_loss_arrays[sname][idx] = compute_per_position_loss(
                model, tokens, device, batch_size=batch_size
            )

        del model
        if device == "cuda":
            import torch; torch.cuda.empty_cache()

    # Save
    out_dir  = os.path.join(exp_dir, "ce_analysis")
    os.makedirs(out_dir, exist_ok=True)
    tag      = f"_n{n_eval}{out_suffix}" if out_suffix else f"_n{n_eval}"
    out_path = os.path.join(out_dir, f"ce_vs_step{tag}.npz")

    np.savez(
        out_path,
        epochs               = epochs,
        loss_train           = loss_arrays["train"],
        loss_valid_novel     = loss_arrays["valid_novel"],
        loss_boolean_cube    = loss_arrays["boolean_cube"],
        pos_loss_train       = pos_loss_arrays["train"],
        pos_loss_valid_novel = pos_loss_arrays["valid_novel"],
        pos_loss_boolean_cube= pos_loss_arrays["boolean_cube"],
        args_json            = json.dumps(args_json),
    )
    print(f"  Saved → {out_path}")
    return out_path


def main():
    args = parse_args()
    for exp_name in args.exp_names:
        exp_dir = os.path.join(args.saveroot, exp_name)
        if not os.path.isdir(exp_dir):
            print(f"WARNING: {exp_dir} not found — skipping.")
            continue
        analyze_experiment(
            exp_dir    = exp_dir,
            n_eval     = args.n_eval,
            batch_size = args.batch_size,
            device     = args.device,
            out_suffix = args.out_suffix,
        )
    print("\nAll done.")


if __name__ == "__main__":
    main()
