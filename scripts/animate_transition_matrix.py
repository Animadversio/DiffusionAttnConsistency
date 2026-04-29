"""
Animate the 4×4 state transition heatmap (counts + row-normalized prob) over
training time, using a sliding window of `--window` consecutive transitions.

Each frame shows:
  Left  — raw transition counts  (summed over window)
  Right — row-normalized transition probabilities

The suptitle displays the current epoch range.

Output: GIF (default) or MP4 saved to {exp_dir}/evolution_analysis/transition_matrix_anim.{gif|mp4}

Usage:
  python scripts/animate_transition_matrix.py \\
      --exp_name DiT_mini_parity_N4096_D36_G3_even_rep2

  python scripts/animate_transition_matrix.py \\
      --exp_name DiT_mini_parity_N4096_D36_G3_even_rep2 \\
      --window 10 --frame_stride 3 --fps 12
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scripts.plot_sample_evolution import (
    load_data,
    compute_transition_matrix,
    STATE_COLORS_4,
)

DEFAULT_SAVEROOT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)

SHORT_LABELS = ['Inv-ambig\n(0)', 'Inv-rule\n(1)', 'Valid-novel\n(2)', 'Memorized\n(3)']


def draw_heatmap(ax, mat_show, fmt, vmin, vmax, cmap='Blues'):
    """Draw a single heatmap into ax, clearing it first."""
    ax.cla()
    sns.heatmap(
        mat_show, annot=True, fmt=fmt, cmap=cmap,
        vmin=vmin, vmax=vmax,
        xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
        linewidths=0.5, ax=ax, cbar=False,
    )
    ax.set_xlabel('TO →', fontsize=9)
    ax.set_ylabel('FROM ↓', fontsize=9)
    for tick, col in zip(ax.get_xticklabels(), STATE_COLORS_4):
        tick.set_color(col)
    for tick, col in zip(ax.get_yticklabels(), STATE_COLORS_4):
        tick.set_color(col)


def run(exp_name, saveroot, window=5, frame_stride=5, fps=10, dpi=120, fmt='gif'):
    exp_dir  = os.path.join(saveroot, exp_name)
    outdir   = os.path.join(exp_dir, 'evolution_analysis')
    out_path = os.path.join(outdir, f'transition_matrix_anim.{fmt}')

    print(f"Loading data for {exp_name} ...")
    d = load_data(exp_name, saveroot)
    T_count, T_prob, epochs, ep_mid = compute_transition_matrix(d)
    # T_count: (T-1, 4, 4)

    n_transitions = T_count.shape[0]
    # Frame indices: start of each window, strided
    frame_starts = list(range(0, n_transitions - window + 1, frame_stride))
    n_frames = len(frame_starts)
    print(f"  Transitions: {n_transitions}, window={window}, stride={frame_stride} → {n_frames} frames")

    # Precompute max count for stable color scale
    vmax_count = max(
        T_count[t:t+window].sum(axis=0).max()
        for t in frame_starts[::max(1, n_frames//20)]   # sample a few frames
    )
    vmax_count = float(vmax_count)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.85)
    suptitle = fig.suptitle('', fontsize=12, fontweight='bold')

    def update(frame_i):
        t0 = frame_starts[frame_i]
        t1 = t0 + window
        mat = T_count[t0:t1].sum(axis=0).astype(np.float64)

        # Left: raw counts
        draw_heatmap(axes[0], mat, fmt='.0f', vmin=0, vmax=vmax_count)
        axes[0].set_title('Counts', fontsize=10)

        # Right: row-normalized prob
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_prob = np.where(row_sums > 0, mat / row_sums, 0.0)
        draw_heatmap(axes[1], mat_prob, fmt='.2f', vmin=0.0, vmax=1.0)
        axes[1].set_title('Prob (row-normalized)', fontsize=10)

        ep_start = int(ep_mid[t0])
        ep_end   = int(ep_mid[min(t1 - 1, n_transitions - 1)])
        suptitle.set_text(
            f'{exp_name}\n'
            f'Transitions [{t0}:{t1}]  —  epoch {ep_start:,} … {ep_end:,}'
        )
        return []

    print(f"  Rendering {n_frames} frames ...")
    fig.canvas.draw()
    frames = []
    for fi in range(n_frames):
        update(fi)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3])
        if (fi + 1) % 50 == 0:
            print(f"    {fi+1}/{n_frames} frames done")

    plt.close(fig)
    os.makedirs(outdir, exist_ok=True)
    import imageio
    if fmt == 'gif':
        imageio.mimwrite(out_path, frames, fps=fps, loop=0)
    else:
        imageio.mimwrite(out_path, frames, fps=fps, codec='mpeg4', quality=7)
    print(f"  Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp_name',     required=True)
    p.add_argument('--saveroot',     default=DEFAULT_SAVEROOT)
    p.add_argument('--window',       type=int, default=5,
                   help='Number of consecutive transitions to sum per frame (default 5)')
    p.add_argument('--frame_stride', type=int, default=5,
                   help='Step between frame windows (default 5)')
    p.add_argument('--fps',          type=int, default=10)
    p.add_argument('--dpi',          type=int, default=120)
    p.add_argument('--format',       default='gif', choices=['gif', 'mp4'],
                   help='Output format: gif (default, uses pillow) or mp4 (needs libx264)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.exp_name, args.saveroot, args.window, args.frame_stride, args.fps, args.dpi, args.format)
