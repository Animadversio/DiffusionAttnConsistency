#!/usr/bin/env python3

import sys
import os
from os.path import join
import json
import pickle as pkl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import argparse
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
sys.path.append("/n/home12/binxuwang/Github/DiffusionAttnConsistency")
from core.parity_lib import sample_group_parity_vec, sample_ensuring_uniqueness, parity_func
from scripts.parity_memorization_eval_cli import compute_membership_counts
from transformers import GPT2Config, GPT2LMHeadModel


def parse_range(range_str: List[str]) -> Tuple[int, int, int]:
    """
    Parses a list of strings into a tuple of three integers representing a range.

    Args:
        range_str (List[str]): List containing start, end, and step as strings.

    Returns:
        Tuple[int, int, int]: Parsed (start, end, step).

    Raises:
        argparse.ArgumentTypeError: If the input is invalid.
    """
    if len(range_str) != 3:
        raise argparse.ArgumentTypeError("Each range must have exactly three integers: start end step.")
    try:
        start, end, step = map(int, range_str)
    except ValueError:
        raise argparse.ArgumentTypeError("All range values must be integers.")
    if start >= end:
        raise argparse.ArgumentTypeError(f"Start ({start}) must be less than end ({end}).")
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Step ({step}) must be a positive integer.")
    return (start, end, step)


def generate_record_times(ranges: List[Tuple[int, int, int]]) -> List[int]:
    """
    Generates a list of record times based on the provided ranges.

    Args:
        ranges (List[Tuple[int, int, int]]): List of ranges defined by (start, end, step).

    Returns:
        List[int]: Generated record times.
    """
    record_times = []
    for start, end, step in ranges:
        record_times.extend(range(start, end, step))
    return record_times


def generate_ckpt_step_list(max_steps, num_ckpts=100, sequence="geomspace") -> List[int]:
    """
    Generates a list of checkpoint steps based on geometric or linear spacing.

    Args:
        max_steps (int): Maximum number of training steps.
        num_ckpts (int): Number of checkpoints to generate.
        sequence (str): Type of sequence - "geomspace" or "linspace".

    Returns:
        List[int]: Generated checkpoint step list.
    """
    if sequence == "geomspace":
        ckpt_step_list = np.geomspace(1, max_steps+1, num_ckpts).astype(int)
        ckpt_step_list = np.unique(ckpt_step_list)
        ckpt_step_list = ckpt_step_list[ckpt_step_list <= max_steps]
    elif sequence == "linspace":
        ckpt_step_list = np.linspace(1, max_steps, num_ckpts).astype(int)
        ckpt_step_list = np.unique(ckpt_step_list)
        ckpt_step_list = ckpt_step_list[ckpt_step_list <= max_steps]
    else:
        raise ValueError(f"Invalid sequence type: {sequence}")
    return ckpt_step_list


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="GPT Parity Learning Experiment")
    parser.add_argument("--exp_name", type=str, default="GPT_parity_pilot", help="Experiment name")
    
    # Training hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--nsteps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Model hyper-parameters
    parser.add_argument("--vocab_size", type=int, default=3, help="Vocabulary size (0, 1, SOS)")
    parser.add_argument("--n_positions", type=int, default=40, help="Maximum sequence length")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--resid_pdrop", type=float, default=0.1, help="Residual dropout")
    parser.add_argument("--embd_pdrop", type=float, default=0.1, help="Embedding dropout")
    parser.add_argument("--attn_pdrop", type=float, default=0.1, help="Attention dropout")
    
    # Data hyper-parameters
    parser.add_argument("--sample_num", type=int, default=4096, help="Number of training samples")
    parser.add_argument("--sample_len", type=int, default=36, help="Sequence length")
    parser.add_argument("--group_size", type=int, default=9, help="Parity group size")
    parser.add_argument("--parity", type=int, default=0, help="Parity constraint (0=even, 1=odd)")
    parser.add_argument("--sos_token", type=int, default=2, help="Start-of-sequence token")
    
    # Evaluation hyper-parameters
    parser.add_argument("--eval_sample_size", type=int, default=1024, help="Number of samples to generate for evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for evaluation sampling")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        '-r', '--record_step_range',
        metavar=('START', 'END', 'STEP'),
        type=int,
        nargs=3,
        action='append',
        default=[],
        help="Define a range with start, end, and step. Can be used multiple times. Evaluation sample frequency"
    )
    
    # Output settings
    parser.add_argument("--save_ckpts", action="store_true", help="Save model checkpoints")
    parser.add_argument("--num_ckpts", type=int, default=100, help="Number of checkpoints")
    parser.add_argument("--print_every", type=int, default=50, help="Print loss every N steps")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use TensorBoard logging")
    parser.add_argument("--tb_log_every", type=int, default=10, help="Log to TensorBoard every N steps")
    
    return parser.parse_args()


def sample_sequences_parallel(model, num_samples=10, max_length=36, sos_token=2, temperature=1.0, device='cuda', batch_size=None):
    """Sample sequences from the trained GPT model in parallel with optional batching"""
    model.eval()
    
    # Use single batch if batch_size not specified or large enough
    if batch_size is None or batch_size >= num_samples:
        batch_size = num_samples
    
    all_sequences = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_size_i = min(batch_size, num_samples - i)
            
            # Generate batch
            input_ids = torch.full((batch_size_i, 1), sos_token, dtype=torch.long).to(device)
            
            for step in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            all_sequences.append(input_ids[:, 1:].cpu())  # Remove SOS token
    
    return torch.cat(all_sequences, dim=0)


@torch.no_grad()
def evaluate_parity_satisfaction(generated_samples_parity, group_size, expected_parity=0):
    """Evaluate how well generated samples satisfy parity constraints - vectorized version"""
    # Convert to {-1, +1} format for parity checking
    num_samples, sample_len = generated_samples_parity.shape
    num_groups = sample_len // group_size
    # Vectorized computation - much faster than looping
    # Reshape to (num_samples, num_groups, group_size)
    sample_pergroup = generated_samples_parity.reshape(num_samples, num_groups, group_size)
    # Compute parity for all groups at once
    sample_eval_parity = parity_func(sample_pergroup, axis=-1)  # Shape: (num_samples, num_groups)
    # Count per-group parity correctness
    pergroup_correct = torch.sum(sample_eval_parity == expected_parity).item()
    pergroup_accuracy = pergroup_correct / (num_samples * num_groups)
    # Count samples where ALL groups are correct
    sample_correct = torch.all(sample_eval_parity == expected_parity, dim=1).sum().item()
    sample_accuracy = sample_correct / num_samples
    
    total_groups = num_samples * num_groups
    # violations = total_groups - pergroup_correct
    
    return {
        'pergroup_accuracy': pergroup_accuracy,
        'sample_accuracy': sample_accuracy,
        'pergroup_correct': pergroup_correct,
        'sample_correct': sample_correct,
        'total_groups': total_groups,
        # 'violations': violations
    }


def sampling_eval_callback_fn(step, model, train_X_tsr, args, device, sample_dir, tb_writer=None):
    """Evaluation callback function - optimized version with TensorBoard logging and sample saving like DiT"""
    
    # Generate samples (keep as tensor for efficiency)
    generated_samples = sample_sequences_parallel(
        model, 
        num_samples=args.eval_sample_size,
        max_length=args.sample_len,
        sos_token=args.sos_token,
        temperature=args.temperature,
        device=device,
        batch_size=args.eval_batch_size
    )
    generated_samples_parity = generated_samples * 2 - 1
    # Save samples to disk like DiT (following DiT naming convention)
    torch.save(generated_samples_parity, f"{sample_dir}/samples_step_{step:06d}.pt")
    
    # Evaluate parity satisfaction (vectorized)
    eval_results = evaluate_parity_satisfaction(
        generated_samples_parity, args.group_size, args.parity
    )
    sample_mem_stats = compute_membership_counts(train_X_tsr.flatten(start_dim=1).cpu(), 
                                             generated_samples_parity.cpu(), args.group_size)
    # Print results in DiT-like format
    print(f"step: {step:06d} | " +
          f"PerGroup correct: {eval_results['pergroup_accuracy']:.3f} [{eval_results['pergroup_correct']}/{eval_results['total_groups']}], " +
          f"Sample correct: {eval_results['sample_accuracy']:.3f} [{eval_results['sample_correct']}/{args.eval_sample_size}]  |  " +
          f"BitGroup mem: {sample_mem_stats['bitgroup_mem_ratio']:.3f} [{sample_mem_stats['bitgroup_mem_num']}/{eval_results['total_groups']}]"+ 
          f"Sample mem: {sample_mem_stats['sample_mem_ratio']:.3f} [{sample_mem_stats['sample_mem_num']}/{args.eval_sample_size}]")
        #   f"violations: {eval_results['violations']} | " +
    
    # Log to TensorBoard
    if tb_writer is not None:
        tb_writer.add_scalar('Eval/PerGroup_Accuracy', eval_results['pergroup_accuracy'], step)
        tb_writer.add_scalar('Eval/Sample_Accuracy', eval_results['sample_accuracy'], step)
        tb_writer.add_scalar('Eval/PerGroup_Correct', eval_results['pergroup_correct'], step)
        tb_writer.add_scalar('Eval/Sample_Correct', eval_results['sample_correct'], step)
        tb_writer.add_scalar('Eval/Sample_Mem_Ratio', sample_mem_stats['sample_mem_ratio'], step)
        tb_writer.add_scalar('Eval/BitGroup_Mem_Ratio', sample_mem_stats['bitgroup_mem_ratio'], step)
        tb_writer.add_scalar('Eval/Sample_Mem_Num', sample_mem_stats['sample_mem_num'], step)
        tb_writer.add_scalar('Eval/BitGroup_Mem_Num', sample_mem_stats['bitgroup_mem_num'], step)
        # tb_writer.add_scalar('Eval/Violations', eval_results['violations'], step)
        
        # Log some example generated sequences as text
        # if step % (args.record_frequency * 5) == 0 or step in [100, 500, 1000]:  # Log samples occasionally
        sample_text = "\n".join([f"Sample {i+1}: {generated_samples[i].tolist()}" 
                                for i in range(min(5, len(generated_samples)))])
        tb_writer.add_text('Eval/Generated_Samples', sample_text, step)
    
    return {
        'step': step,
        **eval_results,
        **sample_mem_stats,
    }


def main():
    args = parse_args()
    device = get_device()
    
    # Setup evaluation schedule (following DiT pattern)
    num_ckpts = args.num_ckpts
    ckpt_step_list = generate_ckpt_step_list(args.nsteps, num_ckpts=num_ckpts, sequence="geomspace")
    
    if args.record_step_range is None or len(args.record_step_range) == 0:
        print("Using default record step range")
        ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), 
                  (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256), 
                  (50000, 100000, 512), (100000, 1000000, 1024)]
        record_step_range = ranges
    else:
        record_step_range = args.record_step_range
        ranges = []
        for r in record_step_range:
            try:
                parsed_range = parse_range(r)
                ranges.append(parsed_range)
            except argparse.ArgumentTypeError as e:
                raise argparse.ArgumentTypeError(str(e))
    
    record_times = generate_record_times(ranges)
    print(f"Record frequency: {args.record_frequency}")
    print(f"Record step range: {record_step_range}")
    print(f"Record times: {record_times[:10]}{'...' if len(record_times) > 10 else ''} (total: {len(record_times)})")
    print(f"Checkpoint steps: {ckpt_step_list[:5]}{'...' if len(ckpt_step_list) > 5 else ''} (total: {len(ckpt_step_list)})")
    
    # Setup experiment directory
    saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionParityLearning"
    savedir = f"{saveroot}/{args.exp_name}"
    ckpt_dir = f"{savedir}/ckpts"
    sample_dir = f"{savedir}/samples"
    tb_dir = f"{savedir}/tensorboard"
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    # Setup TensorBoard
    tb_writer = None
    if args.use_tensorboard:
        tb_writer = SummaryWriter(tb_dir)
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir {tb_dir}")
    
    # Generate training data
    print("Generating training data...")
    parity_str = "even" if args.parity == 0 else "odd"
    x = sample_ensuring_uniqueness(
        N=args.sample_num, 
        sample_func=lambda N: sample_group_parity_vec(
            N=N, sample_len=args.sample_len, 
            group_size=args.group_size, parity=args.parity
        )
    )
    train_X_tsr = th.from_numpy(x).to(th.long)
    # Convert to {0, 1} format for GPT and add SOS token
    train_samples_01 = (x + 1) // 2
    train_samples_01 = train_samples_01.astype(int)
    
    # Add SOS token at the beginning (following notebook implementation)
    train_samples_01_sos = np.concatenate([
        args.sos_token * np.ones((train_samples_01.shape[0], 1), dtype=int),
        train_samples_01
    ], axis=1)
    
    dataset_name = f"parity_N{args.sample_num}_D{args.sample_len}_G{args.group_size}_{parity_str}"
    print(f"Dataset: {dataset_name}")
    print(f"Training data shape: {train_samples_01_sos.shape}")
    
    # Save training data
    np.save(f"{savedir}/training_data.npy", train_samples_01_sos)
    
    # Setup model
    print("Initializing GPT model...")
    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=args.resid_pdrop,
        embd_pdrop=args.embd_pdrop,
        attn_pdrop=args.attn_pdrop,
        loss_type="ForCausalLMLoss"
    )
    
    model = GPT2LMHeadModel(config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training (use data with SOS tokens)
    train_tensor = torch.tensor(train_samples_01_sos, dtype=torch.long).to(device)
    dataset = TensorDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Save configuration
    config_dict = {
        'vocab_size': args.vocab_size,
        'n_positions': args.n_positions,
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'resid_pdrop': args.resid_pdrop,
        'embd_pdrop': args.embd_pdrop,
        'attn_pdrop': args.attn_pdrop,
    }
    json.dump(config_dict, open(f"{savedir}/config.json", "w"))
    json.dump(vars(args), open(f"{savedir}/args.json", "w"))
    
    # Training loop
    print(f"Starting training for {args.nsteps} steps...")
    model.train()
    total_loss = 0
    step = 0
    eval_results = []
    
    # Calculate epochs needed
    steps_per_epoch = len(dataloader)
    num_epochs = (args.nsteps + steps_per_epoch - 1) // steps_per_epoch
    
    for epoch in range(num_epochs):
        for batch_idx, (batch_data,) in enumerate(dataloader):
            if step >= args.nsteps:
                break
                
            optimizer.zero_grad()
            
            # Prepare input and labels for next token prediction
            input_ids = batch_data
            # Use same tensor for input and labels (GPT2LMHeadModel handles the shifting)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            # TensorBoard logging
            if tb_writer is not None and step % args.tb_log_every == 0:
                avg_loss = total_loss / step
                tb_writer.add_scalar('Training/Loss_Step', loss.item(), step)
                tb_writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], step)
                tb_writer.add_scalar('Training/Loss_Avg', avg_loss, step)
                # Log gradient norms occasionally
                # if step % (args.tb_log_every * 10) == 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)
                tb_writer.add_scalar('Training/Gradient_Norm', total_grad_norm, step)
            
            # Print progress
            if step % args.print_every == 0:
                avg_loss = total_loss / step
                print(f"Step {step}/{args.nsteps}, Avg Loss: {avg_loss:.4f}, Current Loss: {loss.item():.4f}")
            
            # Evaluation (sophisticated scheduling like DiT)
            should_eval = False
            if args.record_frequency > 0 and step % args.record_frequency == 0:
                should_eval = True
            elif step in record_times:
                should_eval = True
            
            if should_eval:
                model.eval()
                eval_result = sampling_eval_callback_fn(step, model, train_X_tsr, args, device, sample_dir, tb_writer)
                eval_results.append(eval_result)
                model.train()  # Return to training mode
            
            # Save checkpoint (geometric spacing)
            if args.save_ckpts and step in ckpt_step_list:
                torch.save(model.state_dict(), f"{ckpt_dir}/model_step_{step:06d}.pth")
        
        if step >= args.nsteps:
            break
    
    print("Training completed!")
    print(f"Final average loss: {total_loss / step:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    final_eval = sampling_eval_callback_fn(step, total_loss / step, model, args, device, sample_dir, tb_writer)
    eval_results.append(final_eval)
    
    # Log final model info to TensorBoard
    if tb_writer is not None:
        # Log model architecture
        tb_writer.add_text('Model/Architecture', f"""
        Model Size: {sum(p.numel() for p in model.parameters()):,} parameters
        Embedding Dim: {args.n_embd}
        Layers: {args.n_layer}
        Heads: {args.n_head}
        Vocab Size: {args.vocab_size}
        """, 0)
        
        # Log experiment config
        config_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
        tb_writer.add_text('Experiment/Config', config_text, 0)
        
        # Close TensorBoard writer
        tb_writer.close()
        print(f"TensorBoard logs saved to: {tb_dir}")
    
    # Save final model and results
    torch.save(model.state_dict(), f"{savedir}/model_final.pth")
    pkl.dump(eval_results, open(f"{savedir}/eval_results.pkl", "wb"))
    
    print(f"\nExperiment completed! Results saved to: {savedir}")


if __name__ == "__main__":
    main()