"""
eval_sigma_loss_CLI.py

Evaluate the raw (unweighted) DSM loss as a function of noise scale σ for
saved DiT-mini checkpoints.

For each checkpoint and each data split, computes:
    L(σ) = E_x E_ε[ ||D_θ(x + σε, σ) − x||² ]
where D_θ is the EDM preconditioned denoiser, ε ~ N(0, I).

Data splits:
  train   — original training data (loaded from training_data_tsr.pt)
  test    — freshly sampled valid samples, non-overlapping with train
  random  — uniform ±1 on the same grid (no rule, boolean hypercube)

Supported rules:
  row_k          --rule row_k --K 2 --n_size 6
  row_variable_k --rule row_variable_k --K_list 1 5 --n_size 6
  global_k       --rule global_k --K_list 1 5 --n_size 6
  parity         --rule parity --group_size 2 --parity_val 0

Usage:
  python scripts/eval_sigma_loss_CLI.py \
      --exp_name DiT_mini_rowK2_n6_N4096 \
      --rule row_k --K 2 --n_size 6 \
      --ckpts all

  python scripts/eval_sigma_loss_CLI.py \
      --exp_name DiT_mini_parity_N4096_D36_G2_even_rep2 \
      --rule parity --group_size 2 --parity_val 0 \
      --ckpts all

  # Auto-detect rule params from saved args.json:
  python scripts/eval_sigma_loss_CLI.py \
      --exp_name DiT_mini_parity_N4096_D36_G2_even_rep2 \
      --rule auto \
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
    # Rule
    p.add_argument("--rule", type=str, default="auto",
                   choices=["auto", "row_k", "row_variable_k", "global_k", "parity"],
                   help="Rule type; 'auto' reads from saved args.json")
    p.add_argument("--n_size",     type=int, default=6,
                   help="Grid size for row_k/row_variable_k/global_k rules")
    p.add_argument("--K",          type=int, default=2,
                   help="Active count per row (row_k only)")
    p.add_argument("--K_list",     type=int, nargs="+", default=[1, 5],
                   help="Allowed active counts (row_variable_k / global_k)")
    p.add_argument("--group_size", type=int, default=2,
                   help="Bits per group for parity rule")
    p.add_argument("--parity_val", type=int, default=0,
                   help="Parity target: 0=even, 1=odd")
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
# Auto-detect rule from saved args.json
# ---------------------------------------------------------------------------

def detect_rule_from_args(savedir):
    """Read args.json and return (rule, rule_kwargs) dict."""
    args_path = os.path.join(savedir, "args.json")
    with open(args_path) as f:
        a = json.load(f)
    if "group_size" in a:
        return "parity", {"group_size": a["group_size"], "parity_val": a.get("parity", 0)}
    if "k_ones" in a:
        raise NotImplementedError("exact_k rule not yet supported in sigma eval")
    # row_k / row_variable_k / global_k detected by exp_name heuristic
    name = a.get("exp_name", "")
    if "globalK" in name:
        K_list = [int(x) for x in a.get("K_list", [1, 5])]
        return "global_k", {"K_list": K_list, "n_size": a.get("n_size", 6)}
    if "rowVarK" in name or "varK" in name.lower():
        K_list = [int(x) for x in a.get("K_list", [1, 5])]
        return "row_variable_k", {"K_list": K_list, "n_size": a.get("n_size", 6)}
    K = a.get("K", 2)
    n_size = a.get("n_size", 6)
    return "row_k", {"K": K, "n_size": n_size}


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
    if not isinstance(sigma_data, (int, float)):
        sigma_data = 1.0

    DiT_model   = DiT(**config)
    model_precd = EDMDiTPrecondWrapper(DiT_model, sigma_data=sigma_data,
                                       sigma_min=0.002, sigma_max=80, rho=7.0)

    ckpt_path = os.path.join(savedir, "ckpts", f"model_epoch_{ckpt_epoch:06d}.pth")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    Returns (mean_loss, std_loss).
    """
    sigma_t  = torch.tensor(sigma, device=device, dtype=torch.float32)
    X_rep    = X.repeat_interleave(noise_reps, dim=0)
    all_mse  = []

    for start in range(0, len(X_rep), batch_size):
        x0        = X_rep[start:start + batch_size].to(device)
        x_noisy   = x0 + sigma * torch.randn_like(x0)
        x_denoise = model(x_noisy, sigma_t.expand(len(x0)))
        mse       = ((x_denoise - x0) ** 2).flatten(1).mean(dim=1)
        all_mse.append(mse.cpu())

    all_mse = torch.cat(all_mse)
    return float(all_mse.mean()), float(all_mse.std())


# ---------------------------------------------------------------------------
# Test data generation — per rule
# ---------------------------------------------------------------------------

def sample_test_data_parity(group_size, parity_val, sample_len, n_test, train_codes):
    """
    Generate n_test parity-valid samples not overlapping with training set.
    Returns (n_test, 1, H, W) float tensor where H*W == sample_len.
    """
    from core.parity_lib import sample_group_parity_vec

    n2          = sample_len
    mem_weights = 1 << torch.arange(n2, dtype=torch.long)
    train_set   = set(train_codes.tolist())
    collected   = []
    n_collected = 0

    for _ in range(30):
        needed  = (n_test - n_collected) * 4 + 256
        x_np    = sample_group_parity_vec(needed, sample_len, group_size, parity_val)  # (needed, D) int
        x_flat  = torch.from_numpy(x_np).long()          # {-1,+1}
        bits    = (x_flat > 0).long()
        codes   = (bits * mem_weights).sum(dim=1)
        mask    = torch.tensor([int(c) not in train_set for c in codes.tolist()])
        x_new   = x_flat[mask].float()
        collected.append(x_new)
        n_collected += len(x_new)
        if n_collected >= n_test:
            break

    x_test  = torch.cat(collected, dim=0)[:n_test]       # (n_test, D)
    # Reshape to (n_test, 1, H, W) — recover spatial dims from training data shape
    side    = int(round(sample_len ** 0.5))
    H, W    = (side, side) if side * side == sample_len else (1, sample_len)
    return x_test.reshape(n_test, 1, H, W)


def sample_test_data_rowk(rule, n_size, K, K_list, n_test, train_codes):
    """Generate n_test row-K / global-K valid samples, non-overlapping with train."""
    from core.row_k_lib import (
        sample_row_k_batch, sample_row_variable_k_batch, sample_global_k_batch,
    )

    n2          = n_size * n_size
    mem_weights = 1 << torch.arange(n2, dtype=torch.long)
    train_set   = set(train_codes.tolist())
    collected   = []
    n_collected = 0

    for _ in range(20):
        needed = (n_test - n_collected) * 4 + 256
        if rule == "row_k":
            x = sample_row_k_batch(needed, n_size, K)
        elif rule == "row_variable_k":
            x = sample_row_variable_k_batch(needed, n_size, K_list)
        else:
            x, _ = sample_global_k_batch(needed, n_size, K_list)

        x_flat  = torch.from_numpy(x).long().reshape(len(x), -1)
        codes   = ((x_flat > 0).long() * mem_weights).sum(dim=1)
        mask    = torch.tensor([int(c) not in train_set for c in codes.tolist()])
        x_new   = x_flat[mask]
        collected.append(x_new)
        n_collected += len(x_new)
        if n_collected >= n_test:
            break

    x_test = torch.cat(collected, dim=0)[:n_test].float()
    return x_test.reshape(n_test, 1, n_size, n_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    savedir = os.path.join(SAVEROOT, args.exp_name)
    outdir  = args.outdir or os.path.join(savedir, "sigma_loss")
    os.makedirs(outdir, exist_ok=True)

    # ── Auto-detect rule ────────────────────────────────────────────────────
    rule = args.rule
    rule_kwargs = {}
    if rule == "auto":
        rule, rule_kwargs = detect_rule_from_args(savedir)
        print(f"Auto-detected rule: {rule}  kwargs: {rule_kwargs}")
    elif rule == "parity":
        rule_kwargs = {"group_size": args.group_size, "parity_val": args.parity_val}
    else:
        rule_kwargs = {"K": args.K, "K_list": args.K_list, "n_size": args.n_size}

    # ── Load training data ──────────────────────────────────────────────────
    train_path = os.path.join(savedir, "training_data_tsr.pt")
    X_train    = torch.load(train_path, map_location="cpu", weights_only=False)
    N_train    = len(X_train)
    sample_len = X_train[0].numel()         # total bits per sample
    n_test     = args.n_test or N_train
    print(f"Training data: {X_train.shape}  sample_len={sample_len}  n_test={n_test}")

    # Precompute train codes for overlap detection
    mem_weights = 1 << torch.arange(sample_len, dtype=torch.long)
    _bits       = (X_train.flatten(1) > 0).long()
    train_codes = (_bits * mem_weights).sum(dim=1)

    # ── Build data splits ───────────────────────────────────────────────────
    splits = {}
    if "train" in args.splits:
        splits["train"] = X_train
        print(f"  train  : {N_train} samples")

    if "test" in args.splits:
        print(f"  Sampling {n_test} test samples (non-overlapping)...")
        if rule == "parity":
            X_test = sample_test_data_parity(
                rule_kwargs["group_size"], rule_kwargs["parity_val"],
                sample_len, n_test, train_codes)
        else:
            n_size = rule_kwargs.get("n_size", X_train.shape[-1])
            X_test = sample_test_data_rowk(
                rule, n_size, rule_kwargs.get("K", 2),
                rule_kwargs.get("K_list", [1, 5]), n_test, train_codes)
        splits["test"] = X_test
        print(f"  test   : {len(X_test)} samples")

    if "random" in args.splits:
        X_rand = torch.randint(0, 2, X_train.shape).float() * 2 - 1
        splits["random"] = X_rand
        print(f"  random : {len(X_rand)} samples (uniform ±1)")

    # ── σ grid ──────────────────────────────────────────────────────────────
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.n_sigma)
    print(f"σ grid: {args.n_sigma} points [{args.sigma_min:.4f}, {args.sigma_max:.1f}]")

    # ── Checkpoint list ──────────────────────────────────────────────────────
    ckpt_dir = os.path.join(savedir, "ckpts")
    if args.ckpts == ["all"]:
        ckpt_files  = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pth"))
        ckpt_epochs = [int(f.split("_")[-1].replace(".pth", "")) for f in ckpt_files]
    else:
        ckpt_epochs = [int(c) for c in args.ckpts]
    print(f"Checkpoints: {len(ckpt_epochs)} — {ckpt_epochs[:3]}...{ckpt_epochs[-2:]}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    for ckpt_epoch in ckpt_epochs:
        out_path = os.path.join(outdir, f"sigma_loss_epoch_{ckpt_epoch:06d}.npz")
        if os.path.exists(out_path):
            print(f"  [skip] epoch {ckpt_epoch}")
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
            print(f"    {split_name}: σ=0.002→{means[0]:.4f}  σ≈1→{means[args.n_sigma//2]:.4f}  σ=80→{means[-1]:.4f}")

        # Save metadata
        results.update({
            "meta_exp_name":   args.exp_name,
            "meta_ckpt_epoch": ckpt_epoch,
            "meta_N_train":    N_train,
            "meta_N_test":     n_test,
            "meta_noise_reps": args.noise_reps,
            "meta_sigma_data": sigma_data,
            "meta_rule":       rule,
            "meta_K":          rule_kwargs.get("K", -1),
            "meta_K_list":     np.array(rule_kwargs.get("K_list", [])),
            "meta_group_size": rule_kwargs.get("group_size", -1),
            "meta_parity_val": rule_kwargs.get("parity_val", -1),
        })

        np.savez_compressed(out_path, **results)
        print(f"  Saved → {out_path}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
