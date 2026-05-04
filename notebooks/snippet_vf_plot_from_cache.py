"""
Snippet: plot vector field for G3 rep2 checkpoints using existing cache.

Uses plot_ckpt logic from 20260429_G3_vector_field_three_sample_plane.ipynb
but reads entirely from cache (model=None is safe when cache exists).

Run interactively or import into a notebook cell.

Key grid parameters (must match cache generation):
  NGRID      = 50
  alpha_ax   = linspace(-1.75, 3.75, 50)   # both axes are SAME square range
  beta_ax    = linspace(-1.75, 3.75, 50)   # NOT derived from L_perp
  PLANE_HASH = "191059703a35"
  RANGE_TAG  = "a-1.75_3.75"
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/n/home12/binxuwang/Github/DiffusionAttnConsistency")

from core.vector_field_lib import (
    load_training_data, eval_field_on_grid, project_to_basis, DEFAULT_SAVEROOT,
)

# ── Config ────────────────────────────────────────────────────────────────────
EXP_NAME   = "DiT_mini_parity_N4096_D36_G3_even_rep2"
EXP_DIR    = os.path.join(DEFAULT_SAVEROOT, EXP_NAME)
CACHE_DIR  = os.path.join(EXP_DIR, "vector_field_cache")
DEVICE     = "cpu"
NGRID      = 50
PLANE_HASH = "191059703a35"
RANGE_TAG  = "a-1.75_3.75"

# ── Plane setup ───────────────────────────────────────────────────────────────
x_train = load_training_data(EXP_NAME).numpy()
x_a = x_train[0].copy()
x_b = x_a.copy(); x_b[0] *= -1; x_b[1] *= -1   # valid novel (Hamming-2, parity preserved)
x_c = x_a.copy(); x_c[0] *= -1                  # invalid (Hamming-1, parity broken)

ab = x_b - x_a; L_ab = float(np.linalg.norm(ab)); v_ab = ab / L_ab
ac = x_c - x_a
ac_perp = ac - ac.dot(v_ab) * v_ab
L_perp  = float(np.linalg.norm(ac_perp)); v_ac = ac_perp / L_perp
xc_alpha = float(ac.dot(v_ab))
xc_beta  = float(np.linalg.norm(ac_perp))

# Grid axes: fixed square range [-1.75, 3.75] on BOTH axes (matches cache)
alpha_ax = np.linspace(-1.75, 3.75, NGRID, dtype=np.float32)
beta_ax  = np.linspace(-1.75, 3.75, NGRID, dtype=np.float32)
A, B     = np.meshgrid(alpha_ax, beta_ax, indexing='ij')
# grid_x must be (n1, n2, D) for eval_field_on_grid
grid_x   = (x_a[None, None, :] +
            A[:, :, None] * v_ab[None, None, :] +
            B[:, :, None] * v_ac[None, None, :]).astype(np.float32)

print(f"L_ab={L_ab:.3f}  L_perp={L_perp:.3f}")
print(f"Markers: x_a=(0,0)  x_b=({L_ab:.2f},0)  x_c=({xc_alpha:.2f},{xc_beta:.2f})  x_d=({xc_alpha:.2f},{-xc_beta:.2f})")


# ── plot_ckpt (from 20260429_G3_vector_field_three_sample_plane.ipynb) ────────
def plot_ckpt(ckpt_label, sigmas=(0.2, 0.5, 1.0, 2.0)):
    """
    Two-row figure per sigma:
      Row 0: score magnitude heatmap + score arrows
      Row 1: D·v_ab heatmap + D_pull arrows

    ckpt_label: zero-padded epoch string, e.g. "000049"
    model=None is safe because eval_field_on_grid returns from cache immediately.
    """
    s = 4
    grid_sp = (alpha_ax[-1] - alpha_ax[0]) / len(alpha_ax)

    fig, axes = plt.subplots(2, len(sigmas), figsize=(4.5 * len(sigmas), 9.5))
    fig.suptitle(
        f"G3 rep2  ep {ckpt_label}\n"
        f"Plane: x_a(train) – x_b(valid novel) – x_c(invalid)  [L2 scale]",
        fontsize=10, fontweight='bold',
    )

    for col, sigma in enumerate(sigmas):
        res = eval_field_on_grid(
            None, grid_x, sigma, device=DEVICE,
            cache_dir=CACHE_DIR,
            cache_key=f"ep{ckpt_label}_{PLANE_HASH}_{RANGE_TAG}",
        )
        u_s, v_s       = project_to_basis(res['score'],  v_ab, v_ac)
        Du,  _         = project_to_basis(res['D'],      v_ab, v_ac)
        disp_u, disp_v = project_to_basis(res['D_pull'], v_ab, v_ac)

        # ── row 0: score magnitude + arrows ──────────────────────────────────
        ax   = axes[0][col]
        mag  = res['mag_score']
        vmax = np.percentile(mag, 95)
        im   = ax.pcolormesh(alpha_ax, beta_ax, mag.T, cmap='magma',
                             vmin=0, vmax=vmax, shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax, shrink=0.75, label='‖score‖')
        ax.quiver(A[::s, ::s], B[::s, ::s], u_s[::s, ::s], v_s[::s, ::s],
                  color='white', alpha=0.85,
                  scale=vmax / (1.5 * grid_sp), scale_units='xy', angles='xy', width=0.004)
        ax.plot(0,        0,         'c*', ms=13, label='x_a (train)',       zorder=10, mec='k', mew=0.5)
        ax.plot(L_ab,     0,         'g*', ms=13, label='x_b (valid novel)', zorder=10, mec='k', mew=0.5)
        ax.plot(xc_alpha, xc_beta,   'r*', ms=13, label='x_c (invalid)',     zorder=10, mec='k', mew=0.5)
        ax.plot(xc_alpha, -xc_beta,  'r^', ms=10, label='x_d (mirror)',      zorder=10, mec='k', mew=0.5, alpha=0.6)
        ax.set_aspect('equal')
        ax.set_xlim(alpha_ax[0], alpha_ax[-1]); ax.set_ylim(beta_ax[0], beta_ax[-1])
        ax.set_xlabel('α (L2)', fontsize=9); ax.set_ylabel('β (L2)', fontsize=9)
        ax.set_title(f'σ={sigma:.2f}  ‖score‖', fontsize=9)
        ax.axvline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
        ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
        if col == 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.7)

        # ── row 1: D·v_ab + D_pull arrows ────────────────────────────────────
        ax   = axes[1][col]
        clim = np.percentile(np.abs(Du), 97)
        im2  = ax.pcolormesh(alpha_ax, beta_ax, Du.T, cmap='RdBu_r',
                             vmin=-clim, vmax=clim, shading='auto', rasterized=True)
        plt.colorbar(im2, ax=ax, shrink=0.75, label='D·v_ab')
        dscale = np.percentile(np.sqrt(disp_u**2 + disp_v**2), 90) / (1.5 * grid_sp)
        ax.quiver(A[::s, ::s], B[::s, ::s], disp_u[::s, ::s], disp_v[::s, ::s],
                  color='k', alpha=0.75,
                  scale=dscale, scale_units='xy', angles='xy', width=0.004)
        ax.plot(0,        0,         'c*', ms=13, zorder=10, mec='k', mew=0.5)
        ax.plot(L_ab,     0,         'g*', ms=13, zorder=10, mec='k', mew=0.5)
        ax.plot(xc_alpha, xc_beta,   'r*', ms=13, zorder=10, mec='k', mew=0.5)
        ax.plot(xc_alpha, -xc_beta,  'r^', ms=10, zorder=10, mec='k', mew=0.5, alpha=0.6)
        ax.set_aspect('equal')
        ax.set_xlim(alpha_ax[0], alpha_ax[-1]); ax.set_ylim(beta_ax[0], beta_ax[-1])
        ax.set_xlabel('α (L2)', fontsize=9); ax.set_ylabel('β (L2)', fontsize=9)
        ax.set_title(f'σ={sigma:.2f}  D·v_ab  (arrows: D−x = σ²·score)', fontsize=9)
        ax.axvline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
        ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.4)

    plt.tight_layout()
    return fig


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for epoch in [49, 14251, 119377, 492388]:
        fig = plot_ckpt(f"{epoch:06d}", sigmas=(0.2, 0.5, 1.0, 2.0))
        plt.show()
