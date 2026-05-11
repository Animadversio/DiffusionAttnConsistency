"""
plot_rule_sample_montage.py

Creates a figure showing 5×5 training-set samples for multiple rule types
(parity, exactK, rowK, rowVarK, globalK).  Three rule variants per group.

Saves to figures/rule_sample_montage.{png,pdf}

Usage:
    python scripts/plot_rule_sample_montage.py
"""
import sys, os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parity_lib      import sample_group_parity_vec
from core.exact_k_lib     import sample_exact_k_vec
from core.row_k_lib       import (sample_row_k_batch,
                                   sample_row_variable_k_batch,
                                   sample_global_k_batch)
from circuit_toolkit.plot_utils import to_imgrid, saveallforms

# ── Layout ───────────────────────────────────────────────────────────────────
N_SIDE   = 5          # samples per axis  → 5×5 grid
CELL_PX  = 6          # pixels per grid cell after upscaling
PAD      = 2          # padding (pixels) between tiles
PAD_COLOR = (255, 182, 200)   # pinkish RGB padding between tiles
FIGDIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "rule_montage")

# ── Rule definitions ─────────────────────────────────────────────────────────
def _reshape(arr, n=6):
    """Flatten last dims if needed → (N, n, n)."""
    if arr.ndim == 2:
        return arr.reshape(len(arr), n, n)
    return arr

N25 = N_SIDE * N_SIDE

RULE_GROUPS = [
    ("Parity", [
        ("G=2",   lambda: _reshape(sample_group_parity_vec(N25, 36, group_size=2))),
        ("G=6",   lambda: _reshape(sample_group_parity_vec(N25, 36, group_size=6))),
        ("G=12",  lambda: _reshape(sample_group_parity_vec(N25, 36, group_size=12))),
    ]),
    ("Exact-K", [
        ("K=3",   lambda: _reshape(sample_exact_k_vec(N25, 36, 3))),
        ("K=6",   lambda: _reshape(sample_exact_k_vec(N25, 36, 6))),
        ("K=9",   lambda: _reshape(sample_exact_k_vec(N25, 36, 9))),
    ]),
    ("Row-K", [
        ("K=1",   lambda: sample_row_k_batch(N25, 6, 1).astype(float)),
        ("K=2",   lambda: sample_row_k_batch(N25, 6, 2).astype(float)),
        ("K=3",   lambda: sample_row_k_batch(N25, 6, 3).astype(float)),
    ]),
    ("RowVar-K", [
        ("K∈{1,5}",   lambda: sample_row_variable_k_batch(N25, 6, [1, 5]).astype(float)),
        ("K∈{3,4}",   lambda: sample_row_variable_k_batch(N25, 6, [3, 4]).astype(float)),
        ("K∈{0,2,4,6}", lambda: sample_row_variable_k_batch(N25, 6, [0, 2, 4, 6]).astype(float)),
    ]),
    ("Global-K", [
        ("K∈{1,5}",   lambda: sample_global_k_batch(N25, 6, [1, 5])[0].astype(float)),
        ("K∈{2,4}",   lambda: sample_global_k_batch(N25, 6, [2, 4])[0].astype(float)),
        ("K∈{1,3,5}", lambda: sample_global_k_batch(N25, 6, [1, 3, 5])[0].astype(float)),
    ]),
]

# ── Rendering helpers ─────────────────────────────────────────────────────────

def samples_to_imgrid(samples, n_side=N_SIDE, cell_px=CELL_PX,
                      pad=PAD, pad_color=PAD_COLOR):
    """
    samples: np.ndarray (N, n, n) in {-1, +1}
    Returns: PIL image of an n_side × n_side montage with pinkish padding.
    """
    imgs = (samples.astype(np.float32) + 1.0) / 2.0           # → [0,1]
    imgs = np.repeat(np.repeat(imgs, cell_px, axis=1), cell_px, axis=2)  # (N,H,W)
    # convert each tile to uint8 RGB PIL
    tiles = [Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
             for img in imgs]
    tw, th = tiles[0].size   # tile width/height
    # stitch into n_side × n_side grid with custom pad color
    gw = n_side * tw + (n_side + 1) * pad
    gh = n_side * th + (n_side + 1) * pad
    grid = Image.new("RGB", (gw, gh), pad_color)
    for idx, tile in enumerate(tiles):
        col = idx % n_side
        row = idx // n_side
        x = pad + col * (tw + pad)
        y = pad + row * (th + pad)
        grid.paste(tile, (x, y))
    return grid


def pil_to_ax(ax, pil_img, title):
    ax.imshow(np.array(pil_img))
    ax.set_title(title, fontsize=8.5, pad=2)
    ax.axis("off")


# ── Build figure ──────────────────────────────────────────────────────────────

n_groups = len(RULE_GROUPS)
n_rules  = 3   # variants per group

fig = plt.figure(figsize=(n_rules * 2.8, n_groups * 2.8))
fig.suptitle("Rule-type sample montage  (5×5 training samples per variant)",
             fontsize=11, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(
    n_groups, n_rules,
    figure=fig,
    hspace=0.12,     # tight row spacing
    wspace=0.08,
    left=0.11, right=0.99, top=0.97, bottom=0.01,
)

for gi, (group_name, rules) in enumerate(RULE_GROUPS):
    for ri, (rule_label, sample_fn) in enumerate(rules):
        ax = fig.add_subplot(gs[gi, ri])
        samples = np.array(sample_fn())
        pil = samples_to_imgrid(samples)
        pil_to_ax(ax, pil, rule_label)
    # group label on the left of the first column
    pos = gs[gi, 0].get_position(fig)
    fig.text(
        pos.x0 - 0.08,
        pos.y0 + pos.height / 2,
        group_name,
        ha="right", va="center",
        fontsize=9.5, fontweight="bold",
        rotation=90,
        transform=fig.transFigure,
    )

os.makedirs(FIGDIR, exist_ok=True)
saveallforms(FIGDIR, "rule_sample_montage", fig, fmts=("png", "pdf"))
print(f"Saved to {FIGDIR}/rule_sample_montage.{{png,pdf}}")
plt.close(fig)
