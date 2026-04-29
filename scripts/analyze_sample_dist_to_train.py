"""
Analyze Hamming / L2 distance of generated samples to training set across
training checkpoints, to understand why some noise seeds get memorized while
others stay erroneous or flickering.

For each checkpoint (strided subset), computes for every sample i:
  nearest_hamming[t, i]   : min Hamming distance to any training sample
  nearest_train_idx[t, i] : index of nearest training sample
  nearest_l2[t, i]        : min L2 distance (continuous pre-quantized)

Also classifies each sample slot by its final trajectory:
  0 = stuck invalid   (valid_rate in tail < 0.2)
  1 = flickering      (valid some of the time but < 0.8 in tail, not memorized)
  2 = valid novel     (valid_rate in tail >= 0.8, not memorized)
  3 = memorized       (is_mem in final checkpoint)

Outputs saved to {exp_dir}/evolution_analysis/dist_to_train.npz

Usage:
  python scripts/analyze_sample_dist_to_train.py --exp_name DiT_mini_parity_N4096_D36_G2_even_rep2
  python scripts/analyze_sample_dist_to_train.py --exp_name ... --stride 5
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_SAVEROOT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_sorted_pt_files(samples_dir):
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


def load_flat_cont(filepath):
    """Load .pt -> (N, D) float32, continuous values."""
    t = torch.load(filepath, map_location='cpu', weights_only=False)
    return t.reshape(t.shape[0], -1).float().numpy()


def quantize(x):
    """sign(x) -> {-1, +1} int8."""
    return np.where(x > 0, np.int8(1), np.int8(-1))


def hamming_to_set(q_samples, x_train_q):
    """
    Min Hamming distance from each sample to the training set.
    Uses dot-product trick for {-1,+1}: Hamming(u,v) = (D - u·v) / 2

    q_samples   : (N, D) int8 {-1, +1}
    x_train_q   : (M, D) int8 {-1, +1}
    Returns: min_hamming (N,) int16, nearest_idx (N,) int32
    """
    N, D = q_samples.shape
    dots = q_samples.astype(np.float32) @ x_train_q.astype(np.float32).T  # (N, M)
    hamming = ((D - dots) / 2).astype(np.int16)   # (N, M)
    nearest_idx = hamming.argmin(axis=1).astype(np.int32)
    min_hamming = hamming[np.arange(N), nearest_idx]
    return min_hamming, nearest_idx


def l2_to_set(x_cont, x_train_cont):
    """
    Min L2 distance (continuous) from each sample to training set.
    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b for efficiency.

    x_cont       : (N, D) float32
    x_train_cont : (M, D) float32
    Returns: min_l2 (N,) float32, nearest_idx (N,) int32
    """
    sq_a  = (x_cont ** 2).sum(axis=1, keepdims=True)        # (N, 1)
    sq_b  = (x_train_cont ** 2).sum(axis=1, keepdims=True)  # (M, 1)
    dots  = x_cont @ x_train_cont.T                          # (N, M)
    dist2 = np.clip(sq_a + sq_b.T - 2 * dots, 0, None)      # (N, M)
    nearest_idx = dist2.argmin(axis=1).astype(np.int32)
    min_l2 = np.sqrt(dist2[np.arange(len(x_cont)), nearest_idx])
    return min_l2.astype(np.float32), nearest_idx


def classify_trajectories(is_valid, is_mem, tail_frac=0.1):
    """
    0 = stuck invalid   (valid_rate in tail < 0.2)
    1 = flickering      (0.2 <= valid_rate in tail < 0.8, not memorized)
    2 = valid novel     (valid_rate in tail >= 0.8, not memorized)
    3 = memorized       (is_mem at final checkpoint)
    """
    T, N = is_valid.shape
    tail = max(1, int(T * tail_frac))
    valid_rate_tail = is_valid[-tail:].mean(axis=0)   # (N,)
    mem_final       = is_mem[-1]                       # (N,)

    state = np.zeros(N, dtype=np.uint8)
    # 0 already = stuck
    state[valid_rate_tail >= 0.2] = 1   # flickering
    state[valid_rate_tail >= 0.8] = 2   # valid novel
    state[mem_final]              = 3   # memorized (highest priority)
    return state


def first_epoch_where(mask, epochs):
    """For each sample, return first epoch where mask[t,i] is True. -1 if never."""
    T, N = mask.shape
    result = np.full(N, -1, dtype=np.int64)
    for t in range(T):
        newly = mask[t] & (result == -1)
        result[newly] = epochs[t]
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def run(exp_name, saveroot, stride=5, overwrite=False):
    exp_dir     = os.path.join(saveroot, exp_name)
    samples_dir = os.path.join(exp_dir, 'samples')
    outdir      = os.path.join(exp_dir, 'evolution_analysis')
    out_path    = os.path.join(outdir, 'dist_to_train.npz')

    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {exp_name} — already exists. Use --overwrite to recompute.")
        return

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")

    # Load training data
    train_pt_path = os.path.join(exp_dir, 'training_data_tsr.pt')
    x_train_tsr  = torch.load(train_pt_path, map_location='cpu', weights_only=False)
    x_train_cont = x_train_tsr.reshape(x_train_tsr.shape[0], -1).float().numpy()  # (M, D)
    x_train_q    = quantize(x_train_cont)                                           # (M, D) int8
    M, D = x_train_cont.shape
    print(f"  Training set: {M} samples, D={D}")

    # Load evolution metrics (for state classification)
    evo_path = os.path.join(outdir, 'evolution_metrics.npz')
    evo      = np.load(evo_path, allow_pickle=True)
    epochs   = evo['epochs']    # (T,)
    is_valid = evo['is_valid']  # (T, N)
    is_mem   = evo['is_mem']    # (T, N)
    T, N     = is_valid.shape
    print(f"  Evolution: T={T} checkpoints, N={N} samples/checkpoint")

    # Per-sample trajectory classification
    traj_state     = classify_trajectories(is_valid, is_mem)
    first_mem_ep   = first_epoch_where(is_mem,   epochs)
    first_valid_ep = first_epoch_where(is_valid, epochs)

    # Flickering score 1: std of is_valid in the LAST 1/5 of training
    tail_start = T - max(1, T // 5)
    flicker_score_tail = is_valid[tail_start:].astype(np.float32).std(axis=0)  # (N,)

    # Flickering score 2: std of is_valid AFTER memorization ratio rises above threshold
    mem_ratio   = is_mem.mean(axis=1)       # (T,) global mem ratio at each checkpoint
    onset_mask  = mem_ratio >= 0.1
    if onset_mask.any():
        t_onset       = int(np.argmax(onset_mask))   # first t where ratio >= 0.1
        mem_onset_epoch = int(epochs[t_onset])
    else:
        t_onset         = 0
        mem_onset_epoch = -1   # never reached 0.1
    flicker_score_post_mem = is_valid[t_onset:].astype(np.float32).std(axis=0)  # (N,)

    print(f"  Traj states: stuck={( traj_state==0).sum()}  flicker={(traj_state==1).sum()}  "
          f"novel={(traj_state==2).sum()}  mem={(traj_state==3).sum()}")
    print(f"  Mem onset (>=10%): epoch={mem_onset_epoch}  (t_idx={t_onset}/{T})")

    # Discover and subsample checkpoint files
    entries     = get_sorted_pt_files(samples_dir)
    entries_sub = entries[::stride]
    epochs_sub  = np.array([e for e, _ in entries_sub], dtype=np.int64)
    T_sub       = len(entries_sub)
    print(f"  Processing {T_sub} checkpoints (stride={stride}, total={len(entries)})")

    # Allocate
    nearest_hamming = np.zeros((T_sub, N), dtype=np.int16)
    nearest_l2      = np.zeros((T_sub, N), dtype=np.float32)
    nearest_idx_ham = np.zeros((T_sub, N), dtype=np.int32)
    nearest_idx_l2  = np.zeros((T_sub, N), dtype=np.int32)

    for t_sub, (epoch, fpath) in enumerate(tqdm(entries_sub, desc=exp_name)):
        x_cont = load_flat_cont(fpath)    # (N, D) float32
        q      = quantize(x_cont)         # (N, D) int8

        ham, idx_h = hamming_to_set(q, x_train_q)
        l2,  idx_l = l2_to_set(x_cont, x_train_cont)

        nearest_hamming[t_sub] = ham
        nearest_l2[t_sub]      = l2
        nearest_idx_ham[t_sub] = idx_h
        nearest_idx_l2[t_sub]  = idx_l

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        out_path,
        exp_name        = np.array([exp_name]),
        epochs_sub      = epochs_sub,          # (T_sub,)
        stride          = np.int32(stride),
        D               = np.int32(D),
        N_train         = np.int32(M),
        # per-checkpoint per-sample  (T_sub, N)
        nearest_hamming = nearest_hamming,
        nearest_l2      = nearest_l2,
        nearest_idx_ham = nearest_idx_ham,
        nearest_idx_l2  = nearest_idx_l2,
        # per-sample trajectory  (N,)
        traj_state             = traj_state,             # 0=stuck 1=flicker 2=novel 3=mem
        first_mem_ep           = first_mem_ep,           # first epoch memorized, -1=never
        first_valid_ep         = first_valid_ep,         # first epoch valid, -1=never
        flicker_score_tail     = flicker_score_tail,     # std(is_valid) in last 1/5
        flicker_score_post_mem = flicker_score_post_mem, # std(is_valid) after mem onset
        # memorization onset (scalar)
        mem_onset_epoch        = np.int64(mem_onset_epoch),  # epoch when global mem >= 10%
        mem_onset_t_idx        = np.int64(t_onset),          # checkpoint index of onset
    )
    print(f"  Saved -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp_name',  required=True)
    p.add_argument('--saveroot',  default=DEFAULT_SAVEROOT)
    p.add_argument('--stride',    type=int, default=1,
                   help='Process every Nth checkpoint (default 1 = all checkpoints)')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.exp_name, args.saveroot, args.stride, args.overwrite)
