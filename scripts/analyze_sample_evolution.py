"""
Analyze per-sample evolution across training checkpoints.

For each experiment, streams through all .pt checkpoint files and computes:
  - change_rate[t]        : fraction of samples that flipped >=1 bit (T-1,)
  - bit_change_rate[t,d]  : per-position flip freq (T-1, 36)
  - n_bits_changed[t,i]   : bits flipped per sample per transition (T-1, 2048) int16
  - is_valid[t,i]         : rule validity at each checkpoint (T, 2048) bool
  - is_mem[t,i]           : exact match to training set (T, 2048) bool
  - mean_confidence[t,i]  : mean |x| per sample (T, 2048) float32 -- how decided the model is
  - has_ambiguous[t,i]    : any bit with |x| < AMBIG_THRESH (T, 2048) bool -- quantization error flag
  - ambig_count[t,i]      : number of ambiguous bits (T, 2048) int16

Outputs saved to {exp_dir}/evolution_analysis/evolution_metrics.npz

Usage:
  python scripts/analyze_sample_evolution.py --exp_name DiT_mini_rowK2_n6_N4096
  python scripts/analyze_sample_evolution.py --exp_name DiT_mini_rowK2_n6_N4096 \\
      --saveroot /path/to/DiffusionParityLearning --outdir /path/to/output
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.row_k_lib import (
    check_row_k_batch,
    check_row_variable_k_batch,
    check_global_k_batch,
    per_row_counts,
)
from core.parity_lib import parity_func
from core.exact_k_lib import exact_k_check

# Bits with |x| < this are considered ambiguous (uncertain after quantization)
AMBIG_THRESH = 0.1

DEFAULT_SAVEROOT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sorted_pt_files(samples_dir: str):
    """Return sorted list of (epoch_int, filepath) for all .pt files."""
    entries = []
    for fn in os.listdir(samples_dir):
        if not fn.endswith('.pt'):
            continue
        ep_str = fn.replace('samples_epoch_', '').replace('.pt', '')
        try:
            entries.append((int(ep_str), os.path.join(samples_dir, fn)))
        except ValueError:
            pass
    entries.sort(key=lambda x: x[0])
    return entries


def load_flat(filepath: str) -> np.ndarray:
    """Load .pt checkpoint -> (N, 36) float32 numpy."""
    t = torch.load(filepath, map_location='cpu', weights_only=False)
    return t.reshape(t.shape[0], -1).float().numpy()  # (N, 36)


def quantize(x: np.ndarray) -> np.ndarray:
    """sign(x): values near 0 map to -1 (|x|=0 edge case). Returns int8."""
    return np.where(x > 0, np.int8(1), np.int8(-1))


def load_training_set(exp_dir: str):
    """
    Load training data. Returns:
      x_train : (4096, 36) int8, values in {-1, +1}
      train_codes : set of int64 for O(1) membership test
    """
    path = os.path.join(exp_dir, 'training_data_tsr.pt')
    t = torch.load(path, map_location='cpu', weights_only=False)
    x_train = t.reshape(t.shape[0], -1).numpy().astype(np.int8)
    train_codes = bitpack_set(x_train)
    return x_train, train_codes


def bitpack_set(x_int8: np.ndarray) -> set:
    """Pack each (36,) row into a single int64. Returns set for O(1) lookup."""
    weights = (np.int64(1) << np.arange(36, dtype=np.int64))
    bits = (x_int8 > 0).astype(np.int64)
    codes = (bits * weights).sum(axis=1)
    return set(codes.tolist())


def bitpack_array(x_int8: np.ndarray) -> np.ndarray:
    """Pack (N, 36) int8 -> (N,) int64."""
    weights = (np.int64(1) << np.arange(36, dtype=np.int64))
    bits = (x_int8 > 0).astype(np.int64)
    return (bits * weights).sum(axis=1)


def check_memorized(x_quant: np.ndarray, train_codes: set) -> np.ndarray:
    """(N, 36) int8 -> (N,) bool."""
    codes = bitpack_array(x_quant)
    return np.array([c in train_codes for c in codes.tolist()], dtype=bool)


def load_rule_params(exp_dir: str) -> dict:
    with open(os.path.join(exp_dir, 'args.json')) as f:
        args = json.load(f)
    # ExactK experiments — detect by 'k_ones'
    if 'k_ones' in args:
        return {
            'rule':   'exact_k',
            'k_ones': args['k_ones'],
            'K':      None,
            'K_list': None,
            'n':      None,
        }
    # Parity experiments don't have a 'rule' field — detect by 'group_size'
    if 'group_size' in args:
        return {
            'rule':       'parity',
            'group_size': args['group_size'],
            'parity':     args.get('parity', 0),
            'K':          None,
            'K_list':     None,
            'n':          None,
        }
    return {
        'rule':   args['rule'],
        'K':      args.get('K', None),
        'K_list': args.get('K_list', None),
        'n':      args.get('n_size', 6),
    }


def check_parity_group_batch(x_quant: np.ndarray, group_size: int, parity: int) -> np.ndarray:
    """(N, D) int8 {-1,+1} -> (N,) bool: each group of group_size must have correct parity."""
    N, D = x_quant.shape
    num_groups = D // group_size
    groups = x_quant[:, :num_groups * group_size].reshape(N, num_groups, group_size)
    group_prod = groups.prod(axis=2)          # (N, num_groups), values in {-1,+1}
    target = (-1) ** parity                   # +1 (even) or -1 (odd)
    return (group_prod == target).all(axis=1)


def check_validity(x_quant: np.ndarray, rule_params: dict) -> np.ndarray:
    """(N, 36) int8 -> (N,) bool."""
    rule = rule_params['rule']
    if rule == 'exact_k':
        return exact_k_check(x_quant, rule_params['k_ones'])
    if rule == 'parity':
        return check_parity_group_batch(x_quant, rule_params['group_size'], rule_params['parity'])
    n    = rule_params['n']
    K    = rule_params['K']
    Kl   = rule_params['K_list']
    if rule == 'row_k':
        return check_row_k_batch(x_quant, n, K)
    elif rule == 'row_variable_k':
        return check_row_variable_k_batch(x_quant, n, Kl)
    elif rule == 'global_k':
        valid, _ = check_global_k_batch(x_quant, n, Kl)
        return valid
    else:
        raise ValueError(f"Unknown rule: {rule}")


# ── Main computation ──────────────────────────────────────────────────────────

def run_analysis(exp_name: str, saveroot: str, outdir: str = None):
    exp_dir     = os.path.join(saveroot, exp_name)
    samples_dir = os.path.join(exp_dir, 'samples')

    if outdir is None:
        outdir = os.path.join(exp_dir, 'evolution_analysis')
    os.makedirs(outdir, exist_ok=True)

    out_path = os.path.join(outdir, 'evolution_metrics.npz')
    if os.path.exists(out_path):
        print(f"[skip] {exp_name} — already exists: {out_path}")
        return

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")

    # Load training set and rule params
    x_train, train_codes = load_training_set(exp_dir)
    rule_params = load_rule_params(exp_dir)
    if rule_params['rule'] == 'exact_k':
        print(f"  Rule: exact_k, k_ones={rule_params['k_ones']}")
    elif rule_params['rule'] == 'parity':
        print(f"  Rule: parity, group_size={rule_params['group_size']}, parity={rule_params['parity']}")
    else:
        print(f"  Rule: {rule_params['rule']}, K={rule_params['K']}, K_list={rule_params['K_list']}, n={rule_params['n']}")
    print(f"  Training set: {x_train.shape[0]} samples")

    # Discover checkpoints
    entries = get_sorted_pt_files(samples_dir)
    T = len(entries)
    epochs_arr = np.array([e for e, _ in entries], dtype=np.int64)
    print(f"  Checkpoints: {T}  (ep {epochs_arr[0]} .. {epochs_arr[-1]})")

    # Load first file to get N
    x0 = load_flat(entries[0][1])
    N, D = x0.shape
    print(f"  Samples per checkpoint: N={N}, D={D}")

    # Pre-allocate output arrays
    change_rate      = np.zeros(T - 1, dtype=np.float32)
    bit_change_rate  = np.zeros((T - 1, D), dtype=np.float32)
    n_bits_changed   = np.zeros((T - 1, N), dtype=np.int16)
    is_valid         = np.zeros((T, N), dtype=bool)
    is_mem           = np.zeros((T, N), dtype=bool)
    mean_confidence  = np.zeros((T, N), dtype=np.float32)
    has_ambiguous    = np.zeros((T, N), dtype=bool)
    ambig_count      = np.zeros((T, N), dtype=np.int16)

    prev_quant = None

    for t, (epoch, fpath) in enumerate(tqdm(entries, desc=exp_name)):
        x_cont = load_flat(fpath)                     # (N, 36) float32
        q      = quantize(x_cont)                     # (N, 36) int8
        ambig  = np.abs(x_cont) < AMBIG_THRESH        # (N, 36) bool

        # Per-sample metrics at this checkpoint
        mean_confidence[t] = np.abs(x_cont).mean(axis=1)
        has_ambiguous[t]   = ambig.any(axis=1)
        ambig_count[t]     = ambig.sum(axis=1).astype(np.int16)
        is_valid[t]        = check_validity(q, rule_params)
        is_mem[t]          = check_memorized(q, train_codes)

        # Change metrics vs previous checkpoint
        if prev_quant is not None:
            diff              = (q != prev_quant)                  # (N, 36) bool
            n_bits_changed[t-1] = diff.sum(axis=1).astype(np.int16)
            change_rate[t-1]    = (n_bits_changed[t-1] > 0).mean()
            bit_change_rate[t-1] = diff.mean(axis=0)

        prev_quant = q

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\n  Final checkpoint (ep {epochs_arr[-1]}):")
    print(f"    valid:      {is_valid[-1].mean():.3f}")
    print(f"    memorized:  {is_mem[-1].mean():.3f}")
    print(f"    ambiguous:  {has_ambiguous[-1].mean():.4f}  (>{AMBIG_THRESH})")
    print(f"    confidence: {mean_confidence[-1].mean():.4f}")

    # Ambiguous fraction across ALL checkpoints
    total_ambig = has_ambiguous.mean()
    print(f"\n  Ambiguous fraction across all checkpoints: {total_ambig:.4f}")
    if total_ambig > 0.05:
        print(f"  [WARN] High ambiguous rate — quantization results unreliable for many samples")

    # Report memorization arrival time for each training pattern
    # For each of the N_train patterns, find first epoch where any sample matches it
    print("\n  Computing memorization arrival times ...")
    # sample_codes[t] is an array of (N,) int64 codes
    # We stream again to get this (or derive from is_mem -- but we need which training pattern)
    # Efficient approach: at each checkpoint load quantized codes and intersect with train set

    # Actually we have is_mem[t, i] but not which train pattern sample i matched.
    # We need to re-stream with code comparison per train pattern.
    # Since this is expensive, build a dict: train_code -> first epoch it appears
    train_codes_list = list(train_codes)
    train_first_epoch = {c: -1 for c in train_codes_list}   # -1 = never seen

    print("  Streaming checkpoints for arrival times ...")
    for t, (epoch, fpath) in enumerate(tqdm(entries, desc='arrival')):
        x_cont = load_flat(fpath)
        q      = quantize(x_cont)
        codes  = bitpack_array(q)                 # (N,) int64
        seen   = set(codes.tolist())
        for c in train_codes_list:
            if train_first_epoch[c] == -1 and c in seen:
                train_first_epoch[c] = int(epoch)

    arrival_epochs = np.array([train_first_epoch[c] for c in train_codes_list], dtype=np.int64)
    n_never_seen = (arrival_epochs == -1).sum()
    print(f"  Training patterns seen in samples: {(arrival_epochs >= 0).sum()} / {len(train_codes_list)}")
    print(f"  Patterns never seen: {n_never_seen}")

    # Save all results
    np.savez_compressed(
        out_path,
        # metadata
        epochs          = epochs_arr,          # (T,) int64
        exp_name        = np.array([exp_name]),
        ambig_thresh    = np.float32(AMBIG_THRESH),
        # per-transition metrics (T-1,)
        change_rate     = change_rate,         # (T-1,) float32
        bit_change_rate = bit_change_rate,     # (T-1, 36) float32
        n_bits_changed  = n_bits_changed,      # (T-1, 2048) int16
        # per-checkpoint per-sample (T, 2048)
        is_valid        = is_valid,
        is_mem          = is_mem,
        mean_confidence = mean_confidence,
        has_ambiguous   = has_ambiguous,
        ambig_count     = ambig_count,
        # memorization arrival (N_train,) int64, -1 = never seen
        arrival_epochs  = arrival_epochs,
    )
    print(f"\n  Saved -> {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1e6:.1f} MB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--saveroot', default=DEFAULT_SAVEROOT)
    parser.add_argument('--outdir',   default=None,
                        help='Output dir. Default: {exp_dir}/evolution_analysis/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args.exp_name, args.saveroot, args.outdir)
