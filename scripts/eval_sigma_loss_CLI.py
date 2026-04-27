"""
eval_sigma_loss_CLI.py

Evaluate the raw (unweighted) DSM loss as a function of noise scale σ for
saved DiT-mini row-K checkpoints.

For each checkpoint and each data split, computes:
    L(σ) = E_x E_ε[ ||D_θ(x + σε, σ) − x||² ]
where D_θ is the EDM preconditioned denoiser, ε ~ N(0, I).

Data splits:
  train   — original training data (loaded from training_data_tsr.pt)
  test    — freshly sampled valid samples, non-overlapping with train
  random  — uniform ±1 on the same grid (no rule, boolean hypercube)

Usage:
  python scripts/eval_sigma_loss_CLI.py \
      --exp_name DiT_mini_rowK2_n6_N4096 \
      --rule row_k --K 2 --n_size 6 \
      --ckpts 0 1000 10000 100000 999999 \
      --outdir /tmp/sigma_loss

  # Evaluate all 39 saved checkpoints:
  python scripts/eval_sigma_loss_CLI.py \
      --exp_name DiT_mini_rowK2_n6_N4096 \
      --rule row_k --K 2 \
      --ckpts all

  # globalK run:
  python scripts/eval_sigma_loss_CLI.py \
      --exp_name DiT_mini_globalK15_n6_N4096 \
      --rule global_k --K_list 1 5 \
      --ckpts all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAVEROOT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Experiment
    p.add_argument("--exp_name", type=str, required=True,
                   help="Experiment directory name (under SAVEROOT)")
    # Rule (needed to generate test data)
    p.add_argument("--rule", type=str, default="row_k",
                   choices=["row_k", "row_variable_k", "global_k"])
    p.add_argument("--n_size", type=int, default=6)
    p.add_argument("--K",     type=int, default=2,
                   help="Active count per row (row_k only)")
    p.add_argument("--K_list", type=int, nargs="+", default=[1, 5],
                   help="Allowed active counts (row_variable_k / global_k)")
    # Checkpoints
    p.add_argument("--ckpts", type=str, nargs="+", default=["all"],
                   help="Epoch numbers to evaluate, or 'all' for every saved ckpt")
    # σ grid
    p.add_argument("--n_sigma",    type=int,   default=50)
    p.add_argument("--sigma_min",  type=float, default=0.002)
    p.add_argument("--sigma_max",  type=float, default=80.0)
    # Monte Carlo
    p.add_argument("--noise_reps", type=int,   default=1,
                   help="Number of noise samples per data point per σ")
    p.add_argument("--n_test",     type=int,   default=None,
                   help="Test set size (default: same as training set)")
    # Compute
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--device",     type=str,   default="cuda")
    # Output
    p.add_argument("--outdir",     type=str,   default=None,
                   help="Output directory (default: <savedir>/sigma_loss/)")
    p.add_argument("--splits",     type=str,   nargs="+",
                   default=["train", "test", "random"],
                   choices=["train", "test", "random"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model reconstruction from saved config
# ---------------------------------------------------------------------------

def load_model(savedir, ckpt_epoch, device):
    from easydict import EasyDict as edict
    from core.DiT_model_lib import DiT
    from core.diffusion_edm_lib import EDMDiTPrecondWrapper

    config_path = os.path.join(savedir, "config.json")
    args_path   = os.path.join(savedir, "args.json")

    with open(config_path) as f:
        config = edict(json.load(f))
    with open(args_path) as f:
        train_args = edict(json.load(f))

    sigma_data = getattr(train_args, "sigma_data", 1.0)
    # sigma_data may have been saved as "auto" string in older runs
    if not isinstance(sigma_data, (int, float)):
        sigma_data = 1.0

    DiT_model  = DiT(**config)
    model_precd = EDMDiTPrecondWrapper(DiT_model, sigma_data=sigma_data,
                                       sigma_min=0.002, sigma_max=80, rho=7.0)

    ckpt_path = os.path.join(savedir, "ckpts", f"model_epoch_{ckpt_epoch:06d}.pth")
    state = torch.load(ckpt_path, map_location="cpu")
    model_precd.load_state_dict(state)
    model_precd.eval().to(device)
    return model_precd, sigma_data


# ---------------------------------------------------------------------------
# DSM loss at a fixed σ, no grad
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sigma_loss(model, X, sigma, noise_reps, batch_size, device):
    """
    Compute unweighted MSE  E_x E_ε[ ||D_θ(x + σε, σ) − x||² ]

    Parameters
    ----------
    model      : EDMDiTPrecondWrapper
    X          : torch.Tensor (N, C, H, W) — clean data (CPU or GPU)
    sigma      : float
    noise_reps : int — noise samples per data point
    batch_size : int — GPU batch size
    device     : str

    Returns
    -------
    mean_loss  : float
    std_loss   : float   (std over the N×noise_reps estimates)
    """
    N = len(X)
    sigma_t = torch.tensor(sigma, device=device, dtype=torch.float32)
    all_losses = []

    # Tile X by noise_reps so each clean sample appears noise_reps times
    X_rep = X.repeat_interleave(noise_reps, dim=0)  # (N*noise_reps, C, H, W)
    total = len(X_rep)

    for start in range(0, total, batch_size):
        x0 = X_rep[start:start + batch_size].to(device)
        eps = torch.randn_like(x0)
        x_noisy = x0 + sigma * eps
        # D_θ(x_noisy, σ) — returns denoised x0 estimate
        x_denoised = model(x_noisy, sigma_t.expand(len(x0)))
        # Per-sample MSE (flatten spatial dims)
        mse = ((x_denoised - x0) ** 2).flatten(1).mean(dim=1)  # (bsz,)
        all_losses.append(mse.cpu())

    all_losses = torch.cat(all_losses)  # (N*noise_reps,)
    return float(all_losses.mean()), float(all_losses.std())


# ---------------------------------------------------------------------------
# Test data generation (non-overlapping with train)
# ---------------------------------------------------------------------------

def sample_test_data(rule, n_size, K, K_list, n_test, train_codes):
    """
    Sample n_test valid samples not overlapping with training data.
    train_codes : 1-D torch.Tensor of packed integer codes of training samples.
    Returns torch.Tensor (n_test, 1, n_size, n_size) float.
    """
    from core.row_k_lib import (
        sample_row_k_batch, sample_row_variable_k_batch, sample_global_k_batch,
        valid_set_size_row_k, valid_set_size_row_variable_k, valid_set_size_global_k,
        sample_row_k_dataset, sample_row_variable_k_dataset, sample_global_k_dataset,
    )

    n2 = n_size * n_size
    mem_weights = 1 << torch.arange(n2, dtype=torch.long)
    train_set   = set(train_codes.tolist())

    collected = []
    n_collected = 0
    max_tries = 20

    for _ in range(max_tries):
        needed = (n_test - n_collected) * 4 + 256  # oversample

        if rule == "row_k":
            x = sample_row_k_batch(needed, n_size, K)
        elif rule == "row_variable_k":
            x = sample_row_variable_k_batch(needed, n_size, K_list)
        else:  # global_k
            x, _ = sample_global_k_batch(needed, n_size, K_list)

        # Pack to integer codes for fast membership test
        # x shape: (needed, n_size, n_size) — already ±1
        x_flat  = torch.from_numpy(x).long().reshape(len(x), -1)  # (needed, n2)
        x_pm1   = x_flat                            # already {-1,+1}
        codes   = (x_flat > 0).long()
        codes   = (codes * mem_weights).sum(dim=1)  # hash

        # Filter out overlaps
        mask  = torch.tensor([int(c) not in train_set for c in codes.tolist()])
        x_new = x_pm1[mask]
        collected.append(x_new)
        n_collected += len(x_new)
        if n_collected >= n_test:
            break

    x_test = torch.cat(collected, dim=0)[:n_test].float()           # (n_test, n2)
    return x_test.reshape(n_test, 1, n_size, n_size)                # (n_test,1,H,W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    savedir = os.path.join(SAVEROOT, args.exp_name)
    outdir  = args.outdir or os.path.join(savedir, "sigma_loss")
    os.makedirs(outdir, exist_ok=True)

    # ── Load training data ──────────────────────────────────────────────────
    train_path = os.path.join(savedir, "training_data_tsr.pt")
    X_train = torch.load(train_path, map_location="cpu")   # (N,1,n,n) float ±1
    N_train = len(X_train)
    n_size  = X_train.shape[-1]
    n_test  = args.n_test or N_train
    print(f"Training data: {X_train.shape}, n_test={n_test}")

    # Precompute train codes for overlap detection
    n2          = n_size * n_size
    mem_weights = 1 << torch.arange(n2, dtype=torch.long)
    _bits       = (X_train.flatten(1) > 0).long()
    train_codes = (_bits * mem_weights).sum(dim=1)

    # ── Build data splits ───────────────────────────────────────────────────
    splits = {}
    if "train" in args.splits:
        splits["train"] = X_train
        print(f"  train  : {len(X_train)} samples")

    if "test" in args.splits:
        print(f"  Sampling {n_test} test samples (non-overlapping)...")
        X_test = sample_test_data(
            args.rule, n_size, args.K, args.K_list, n_test, train_codes)
        splits["test"] = X_test
        print(f"  test   : {len(X_test)} samples")

    if "random" in args.splits:
        X_rand = torch.randint(0, 2, (n_test, 1, n_size, n_size)).float() * 2 - 1
        splits["random"] = X_rand
        print(f"  random : {len(X_rand)} samples (uniform ±1)")

    # ── σ grid ──────────────────────────────────────────────────────────────
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.n_sigma)
    print(f"σ grid: {args.n_sigma} points [{args.sigma_min:.4f}, {args.sigma_max:.1f}]")

    # ── Checkpoint list ──────────────────────────────────────────────────────
    ckpt_dir = os.path.join(savedir, "ckpts")
    if args.ckpts == ["all"]:
        ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pth"))
        ckpt_epochs = [int(f.split("_")[-1].replace(".pth", "")) for f in ckpt_files]
    else:
        ckpt_epochs = [int(c) for c in args.ckpts]
    print(f"Checkpoints to evaluate: {len(ckpt_epochs)} — {ckpt_epochs[:5]}...{ckpt_epochs[-3:]}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    for ckpt_epoch in ckpt_epochs:
        out_path = os.path.join(outdir, f"sigma_loss_epoch_{ckpt_epoch:06d}.npz")
        if os.path.exists(out_path):
            print(f"  [skip] epoch {ckpt_epoch} already done → {out_path}")
            continue

        print(f"\n── Epoch {ckpt_epoch} ──")
        model, sigma_data = load_model(savedir, ckpt_epoch, device)

        results = {"sigma_grid": sigma_grid}

        for split_name, X in splits.items():
            means = np.zeros(args.n_sigma)
            stds  = np.zeros(args.n_sigma)
            for si, sigma in enumerate(sigma_grid):
                m, s = compute_sigma_loss(
                    model, X, sigma, args.noise_reps, args.batch_size, device)
                means[si] = m
                stds[si]  = s
            results[f"loss_{split_name}"] = means
            results[f"std_{split_name}"]  = stds
            print(f"    {split_name}: loss@σ=0.002={means[0]:.4f}  loss@σ=1.0={means[args.n_sigma//2]:.4f}  loss@σ=80={means[-1]:.4f}")

        # Save metadata
        results["meta_exp_name"]   = args.exp_name
        results["meta_ckpt_epoch"] = ckpt_epoch
        results["meta_N_train"]    = N_train
        results["meta_N_test"]     = n_test
        results["meta_noise_reps"] = args.noise_reps
        results["meta_sigma_data"] = sigma_data
        results["meta_rule"]       = args.rule
        results["meta_K"]          = args.K
        results["meta_K_list"]     = np.array(args.K_list)

        np.savez_compressed(out_path, **results)
        print(f"  Saved → {out_path}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
