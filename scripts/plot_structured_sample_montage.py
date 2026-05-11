"""
plot_structured_sample_montage.py

3×3 sample montage for structured categorical rules:
  - RowOnly  (permutation rows, no column constraint)
  - Latin Square  (n=5 and n=6)
  - Sudoku  (6×6 with 2×3 blocks)

Each sample is a categorical n×n grid displayed with a discrete colormap.
For sudoku, block separators are drawn.

Saves to figures/rule_montage/structured_sample_montage.{png,pdf}

Usage:
    python scripts/plot_structured_sample_montage.py
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.latin_square_lib import (sample_latin_square_vec,
                                    sample_row_permutation_matrix,
                                    sample_sudoku_dataset)
from circuit_toolkit.plot_utils import saveallforms

# ── Layout ────────────────────────────────────────────────────────────────────
N_SIDE   = 3          # 3×3 grid of samples per variant
CELL_PX  = 20         # pixels per cell → larger for readability
PAD      = 4          # padding between tiles
PAD_COLOR = (255, 182, 200)   # pinkish
FIGDIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figures", "rule_montage")

# Discrete colormaps per n (number of categories)
def _make_cmap(n):
    """Return list of n distinct RGB uint8 colors."""
    base = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [(np.array(base(i / max(n-1,1))[:3]) * 255).astype(np.uint8)
            for i in range(n)]

# ── Rule definitions ──────────────────────────────────────────────────────────
N9 = N_SIDE * N_SIDE

def _int_grid(arr, n):
    """(N, n²) int → (N, n, n) int."""
    return arr.reshape(len(arr), n, n)

RULE_GROUPS = [
    # (group_name, n, block_hw_or_None, [(label, samples_int_nn), ...])
    # samples_int_nn: np.ndarray (N9, n, n) with values 0..n-1
    ("RowOnly\n(n=6)", 6, None, [
        ("n=6, rep 1", lambda: _int_grid(sample_row_permutation_matrix(N9, 6), 6)),
        ("n=6, rep 2", lambda: _int_grid(sample_row_permutation_matrix(N9, 6), 6)),
        ("n=6, rep 3", lambda: _int_grid(sample_row_permutation_matrix(N9, 6), 6)),
    ]),
    ("Latin Square\n(n=5)", 5, None, [
        ("n=5, rep 1", lambda: _int_grid(sample_latin_square_vec(N9, 5), 5)),
        ("n=5, rep 2", lambda: _int_grid(sample_latin_square_vec(N9, 5), 5)),
        ("n=5, rep 3", lambda: _int_grid(sample_latin_square_vec(N9, 5), 5)),
    ]),
    ("Latin Square\n(n=6)", 6, None, [
        ("n=6, rep 1", lambda: _int_grid(sample_latin_square_vec(N9, 6), 6)),
        ("n=6, rep 2", lambda: _int_grid(sample_latin_square_vec(N9, 6), 6)),
        ("n=6, rep 3", lambda: _int_grid(sample_latin_square_vec(N9, 6), 6)),
    ]),
    ("Sudoku\n(6×6, 2×3)", 6, (2, 3), [
        ("rep 1", lambda: _int_grid(sample_sudoku_dataset(N9, 6, block_h=2, block_w=3), 6)),
        ("rep 2", lambda: _int_grid(sample_sudoku_dataset(N9, 6, block_h=2, block_w=3), 6)),
        ("rep 3", lambda: _int_grid(sample_sudoku_dataset(N9, 6, block_h=2, block_w=3), 6)),
    ]),
]

# ── Rendering helpers ─────────────────────────────────────────────────────────

def grid_to_pil(grid_int, n, cell_px=CELL_PX, block_hw=None):
    """
    grid_int: (n, n) int array, values 0..n-1
    Returns: PIL RGB image of size (n*cell_px, n*cell_px).
    Draws block separators if block_hw=(bh, bw) is given.
    """
    colors = _make_cmap(n)
    img = Image.new("RGB", (n * cell_px, n * cell_px), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    for r in range(n):
        for c in range(n):
            val = int(grid_int[r, c])
            col = tuple(colors[val].tolist())
            x0, y0 = c * cell_px, r * cell_px
            draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=col)
    # block separators for sudoku
    if block_hw is not None:
        bh, bw = block_hw
        sep_col = (30, 30, 30)
        sep_w   = max(2, cell_px // 8)
        for rb in range(1, n // bh):
            y = rb * bh * cell_px
            draw.rectangle([0, y - sep_w // 2, n * cell_px, y + sep_w // 2], fill=sep_col)
        for cb in range(1, n // bw):
            x = cb * bw * cell_px
            draw.rectangle([x - sep_w // 2, 0, x + sep_w // 2, n * cell_px], fill=sep_col)
    return img


def samples_to_imgrid(samples_nn, n, n_side=N_SIDE,
                      cell_px=CELL_PX, pad=PAD,
                      pad_color=PAD_COLOR, block_hw=None):
    """
    samples_nn: (N9, n, n) int
    Returns: PIL image of n_side × n_side montage.
    """
    tiles = [grid_to_pil(s, n, cell_px=cell_px, block_hw=block_hw)
             for s in samples_nn]
    tw, th = tiles[0].size
    gw = n_side * tw + (n_side + 1) * pad
    gh = n_side * th + (n_side + 1) * pad
    canvas = Image.new("RGB", (gw, gh), pad_color)
    for idx, tile in enumerate(tiles):
        col = idx % n_side
        row = idx // n_side
        canvas.paste(tile, (pad + col * (tw + pad), pad + row * (th + pad)))
    return canvas


def pil_to_ax(ax, pil_img, title):
    ax.imshow(np.array(pil_img), interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")


# ── Build figure ──────────────────────────────────────────────────────────────

n_groups = len(RULE_GROUPS)
n_cols   = 3   # variants per group

fig = plt.figure(figsize=(n_cols * 2.8, n_groups * 2.8))
fig.suptitle("Structured rule sample montage  (3×3 samples, categorical coloring)",
             fontsize=11, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(
    n_groups, n_cols,
    figure=fig,
    hspace=0.12,
    wspace=0.08,
    left=0.13, right=0.99, top=0.97, bottom=0.01,
)

for gi, (group_name, n, block_hw, variants) in enumerate(RULE_GROUPS):
    for vi, (label, sample_fn) in enumerate(variants):
        ax = fig.add_subplot(gs[gi, vi])
        samples = np.array(sample_fn())        # (N9, n, n)
        pil = samples_to_imgrid(samples, n, block_hw=block_hw)
        pil_to_ax(ax, pil, label)
    # group label on the left
    pos = gs[gi, 0].get_position(fig)
    fig.text(
        pos.x0 - 0.09,
        pos.y0 + pos.height / 2,
        group_name,
        ha="right", va="center",
        fontsize=9, fontweight="bold",
        rotation=90,
        transform=fig.transFigure,
    )

# ── Color legend (one per unique n) ──────────────────────────────────────────
# small patch legend at bottom
from matplotlib.patches import Patch
for n_val in [5, 6]:
    colors = _make_cmap(n_val)
    handles = [Patch(facecolor=tuple(c / 255 for c in col), label=str(i))
               for i, col in enumerate(colors)]
    fig.legend(handles=handles, title=f"Value (n={n_val})",
               loc="lower center",
               bbox_to_anchor=(0.25 if n_val == 5 else 0.75, -0.03),
               ncol=n_val, fontsize=7, title_fontsize=7,
               handlelength=1, handleheight=1)

os.makedirs(FIGDIR, exist_ok=True)
saveallforms(FIGDIR, "structured_sample_montage", fig, fmts=("png", "pdf"))
print(f"Saved to {FIGDIR}/structured_sample_montage.{{png,pdf}}")
plt.close(fig)
