"""
GPT checkpoint evaluation library.

Computes cross-entropy loss on multiple data splits (train, valid-novel,
boolean cube) at each position in the sequence, enabling analysis of how
a GPT model's next-token prediction quality evolves across training.

Token format: [SOS(2), b1, b2, ..., b36]  (37 tokens, {0,1,2})
Logit at position i predicts token at position i+1.
Per-position loss p (1-indexed) = CE(predict b_p | SOS, b1...b_{p-1}).
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.parity_lib import sample_group_parity_vec, sample_ensuring_uniqueness


# ── Model loading ─────────────────────────────────────────────────────────────

def load_gpt_config(exp_dir):
    """Load GPT2Config from experiment directory (config.json + args.json)."""
    config_path = os.path.join(exp_dir, "config.json")
    args_path   = os.path.join(exp_dir, "args.json")
    if os.path.exists(config_path):
        return GPT2Config.from_pretrained(exp_dir)
    # Fall back to args.json
    with open(args_path) as f:
        args = json.load(f)
    return GPT2Config(
        vocab_size   = args.get("vocab_size",   3),
        n_positions  = args.get("n_positions",  40),
        n_embd       = args.get("n_embd",       384),
        n_layer      = args.get("n_layer",      6),
        n_head       = args.get("n_head",       6),
        resid_pdrop  = args.get("resid_pdrop",  0.0),
        embd_pdrop   = args.get("embd_pdrop",   0.0),
        attn_pdrop   = args.get("attn_pdrop",   0.0),
    )


def load_gpt_checkpoint(ckpt_path, exp_dir, device="cuda"):
    """Load GPT2LMHeadModel from a .pth checkpoint file."""
    config = load_gpt_config(exp_dir)
    model  = GPT2LMHeadModel(config).to(device)
    state  = torch.load(ckpt_path, map_location=device)
    # handle raw state_dict or wrapped dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def list_checkpoints(exp_dir):
    """Return sorted (step, path) pairs for all saved checkpoints."""
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    paths = sorted(
        (f for f in os.listdir(ckpt_dir) if f.endswith(".pth")),
        key=lambda x: int(x.split("_")[-1].replace(".pth", ""))
    )
    result = []
    for fname in paths:
        step = int(fname.split("_")[-1].replace(".pth", ""))
        result.append((step, os.path.join(ckpt_dir, fname)))
    return result


# ── CE loss computation ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_ce_loss(model, tokens, device, batch_size=512):
    """
    Mean cross-entropy loss over all positions (standard GPT LM loss).

    Parameters
    ----------
    model   : GPT2LMHeadModel (eval mode)
    tokens  : np.ndarray (N, seq_len) int — token sequences incl. SOS
    device  : torch device

    Returns
    -------
    float — mean CE loss
    """
    total_loss = 0.0
    n_batches  = 0
    N = len(tokens)
    for i in range(0, N, batch_size):
        batch = torch.tensor(tokens[i:i+batch_size], dtype=torch.long, device=device)
        out   = model(input_ids=batch, labels=batch)
        total_loss += out.loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def compute_per_position_loss(model, tokens, device, batch_size=512):
    """
    Cross-entropy loss at each bit position (predicting token p from tokens 0..p-1).

    Position index p in returned array (0-indexed) corresponds to predicting
    the (p+1)-th token, i.e. b_{p+1} in 1-indexed bit notation.
    The array length is seq_len - 1  (= 36 for standard 37-token sequences).

    Parameters
    ----------
    model   : GPT2LMHeadModel (eval mode)
    tokens  : np.ndarray (N, seq_len)
    device  : torch device

    Returns
    -------
    np.ndarray (seq_len - 1,) — mean CE at each prediction position
    """
    N, seq_len  = tokens.shape
    n_positions = seq_len - 1          # number of next-token predictions
    pos_loss_sum = np.zeros(n_positions, dtype=np.float64)
    n_samples    = 0

    for i in range(0, N, batch_size):
        batch = torch.tensor(tokens[i:i+batch_size], dtype=torch.long, device=device)
        logits = model(input_ids=batch).logits   # (B, seq_len, vocab)
        B = batch.size(0)
        # logits[:, p, :] predicts batch[:, p+1]
        for p in range(n_positions):
            loss_p = F.cross_entropy(
                logits[:, p, :],     # (B, vocab)
                batch[:, p + 1],     # (B,)
                reduction='sum'
            ).item()
            pos_loss_sum[p] += loss_p
        n_samples += B

    return pos_loss_sum / n_samples


# ── Dataset construction ──────────────────────────────────────────────────────

def _to_sos_tokens(x_pm1, sos_token=2):
    """
    Convert {-1,+1} parity samples → {0,1} + prepend SOS token.

    Parameters
    ----------
    x_pm1 : np.ndarray (N, D) in {-1, +1}

    Returns
    -------
    np.ndarray (N, D+1) int in {0, 1, sos_token}
    """
    x_01 = ((x_pm1 + 1) // 2).astype(int)
    sos  = sos_token * np.ones((len(x_01), 1), dtype=int)
    return np.concatenate([sos, x_01], axis=1)


def build_valid_novel_set(train_data_pm1, group_size, sample_len, parity=0,
                          n_samples=4096, sos_token=2, seed=42, max_tries=20):
    """
    Sample parity-valid sequences not in the training set.

    Parameters
    ----------
    train_data_pm1 : (N_train, D) in {-1, +1}

    Returns
    -------
    np.ndarray (n_samples, D+1) — tokenised with SOS, in {0,1,sos_token}
    """
    rng = np.random.default_rng(seed)
    train_set = set(map(tuple, train_data_pm1.tolist()))
    collected = []
    for _ in range(max_tries):
        candidates = sample_group_parity_vec(
            N=n_samples * 4, sample_len=sample_len,
            group_size=group_size, parity=parity
        )
        novel = [c for c in candidates if tuple(c.tolist()) not in train_set]
        collected.extend(novel)
        if len(collected) >= n_samples:
            break
    novel_pm1 = np.array(collected[:n_samples])
    return _to_sos_tokens(novel_pm1, sos_token)


def build_boolean_cube_set(sample_len, n_samples=4096, sos_token=2, seed=42):
    """
    Uniformly random {0,1}^sample_len sequences + SOS.
    No parity filtering — represents the full boolean cube prior.

    Returns
    -------
    np.ndarray (n_samples, sample_len+1)
    """
    rng   = np.random.default_rng(seed)
    x_01  = rng.integers(0, 2, size=(n_samples, sample_len))
    sos   = sos_token * np.ones((n_samples, 1), dtype=int)
    return np.concatenate([sos, x_01], axis=1)


def build_all_test_sets(exp_dir, n_samples=4096, sos_token=2, seed=42):
    """
    Build all three evaluation sets from a saved experiment directory.

    Returns
    -------
    dict with keys 'train', 'valid_novel', 'boolean_cube'
    Each value: np.ndarray (n_samples, seq_len) of token indices
    """
    with open(os.path.join(exp_dir, "args.json")) as f:
        args = json.load(f)

    # Training set (already tokenized with SOS in training_data.npy)
    train_tokens = np.load(os.path.join(exp_dir, "training_data.npy"))

    # Recover {-1,+1} from the stored {0,1,SOS} tokens (drop SOS, rescale)
    train_01  = train_tokens[:, 1:]   # drop SOS column
    train_pm1 = train_01 * 2 - 1

    valid_novel = build_valid_novel_set(
        train_pm1, group_size=args["group_size"],
        sample_len=args["sample_len"], parity=args.get("parity", 0),
        n_samples=n_samples, sos_token=sos_token, seed=seed,
    )
    boolean_cube = build_boolean_cube_set(
        sample_len=args["sample_len"], n_samples=n_samples,
        sos_token=sos_token, seed=seed,
    )
    return {
        "train":        train_tokens,
        "valid_novel":  valid_novel,
        "boolean_cube": boolean_cube,
    }
