"""
Vector field visualization for EDM diffusion models on 2D slices.

For parity/binary experiments the "natural" 2D slices are:
  1. Parity group plane  — vary 2 bits within one parity group, fix others
  2. Valid↔Invalid axis  — vary one bit (flipping one group bit), + a second free axis
  3. Two-sample plane    — linear interpolation between two data points, + orthogonal axis

Key model interface (EDMDiTPrecondWrapper):
  denoiser(x, sigma) → D(x, σ)    shape: (B, 1, H, W)
  score(x, sigma)    → s(x, σ) = (D(x,σ) - x) / σ²

Visualization strategies:
  - Denoiser D(x,σ)       : best as a heatmap of a scalar projection
                            (e.g. onto one axis) or as an attractor target position
  - Score s(x,σ)          : contractive vector field → quiver plot
                            divergence, magnitude as heatmap
  - D(x,σ) - x_baseline  : displacement from a reference — shows direction of pull
"""

import os
import sys
import json
import numpy as np
import torch

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_SAVEROOT = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "DL_Projects/DiffusionParityLearning"
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(exp_name, ckpt_epoch, device='cpu', saveroot=DEFAULT_SAVEROOT):
    """
    Load EDMDiTPrecondWrapper from a checkpoint.

    Parameters
    ----------
    exp_name   : str   experiment folder name
    ckpt_epoch : int   training epoch to load (e.g. 50000)
    device     : str   'cpu' or 'cuda'
    saveroot   : str

    Returns
    -------
    model_precd : EDMDiTPrecondWrapper  (eval mode, on device)
    sigma_data  : float
    args        : easydict  training args
    """
    from easydict import EasyDict as edict
    from core.DiT_model_lib import DiT
    from core.diffusion_edm_lib import EDMDiTPrecondWrapper

    savedir = os.path.join(saveroot, exp_name)
    with open(os.path.join(savedir, "config.json")) as f:
        config = edict(json.load(f))
    with open(os.path.join(savedir, "args.json")) as f:
        args = edict(json.load(f))

    sigma_data = float(getattr(args, "sigma_data", 1.0))
    DiT_model  = DiT(**config)
    model_precd = EDMDiTPrecondWrapper(DiT_model, sigma_data=sigma_data,
                                       sigma_min=0.002, sigma_max=80, rho=7.0)
    ckpt_path = os.path.join(savedir, "ckpts", f"model_epoch_{ckpt_epoch:06d}.pth")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_precd.load_state_dict(state)
    model_precd.eval().to(device)
    return model_precd, sigma_data, args


def load_training_data(exp_name, saveroot=DEFAULT_SAVEROOT):
    """Load training set as (N, D) float32 tensor in {-1, +1}."""
    path = os.path.join(saveroot, exp_name, "training_data_tsr.pt")
    x = torch.load(path, map_location="cpu", weights_only=False)
    return x.reshape(x.shape[0], -1).float()   # (N, D)


# ── Denoiser / score evaluation ───────────────────────────────────────────────

@torch.no_grad()
def eval_denoiser(model, x_flat, sigma, device='cpu', batch_size=512):
    """
    Evaluate D(x, σ) for a batch of flat inputs.

    Parameters
    ----------
    model   : EDMDiTPrecondWrapper
    x_flat  : (N, D) float tensor  — flat inputs (continuous, not quantized)
    sigma   : float
    device  : str
    batch_size : int  — chunk size for GPU memory

    Returns
    -------
    D_flat  : (N, D) float32 numpy  — denoiser output
    """
    model.eval()
    N, D = x_flat.shape
    # infer spatial shape from model config (6×6=36 or 1×D)
    side = int(round(D ** 0.5))
    H, W = (side, side) if side * side == D else (1, D)

    sigma_t = torch.tensor([sigma], dtype=torch.float32, device=device)
    D_out = []
    for i in range(0, N, batch_size):
        xb = x_flat[i:i+batch_size].to(device).reshape(-1, 1, H, W)
        sig_b = sigma_t.expand(xb.shape[0])
        d = model(xb, sig_b, cond=None)
        D_out.append(d.reshape(-1, D).cpu().float())
    return torch.cat(D_out, dim=0).numpy()


@torch.no_grad()
def eval_score(model, x_flat, sigma, device='cpu', batch_size=512):
    """
    Evaluate score s(x, σ) = (D(x,σ) - x) / σ² for flat inputs.

    Returns
    -------
    score_flat : (N, D) float32 numpy
    D_flat     : (N, D) float32 numpy
    """
    x_np = x_flat.numpy() if hasattr(x_flat, 'numpy') else np.asarray(x_flat, dtype=np.float32)
    D_np = eval_denoiser(model, torch.from_numpy(x_np).float(), sigma, device, batch_size)
    score_np = (D_np - x_np) / (sigma ** 2)
    return score_np, D_np


# ── 2D grid construction ──────────────────────────────────────────────────────

def make_group_plane_grid(x_baseline, group_bits, v_range=(-2.5, 2.5), n_grid=30):
    """
    Parity group plane: vary exactly 2 bits (the group_bits pair) over a 2D grid,
    fix all other dimensions to x_baseline values.

    Parameters
    ----------
    x_baseline : (D,) float  — base sample (all other bits fixed here)
    group_bits : (2,) int    — indices of the two bits to vary
    v_range    : (lo, hi)    — range of continuous values on each axis
    n_grid     : int         — grid resolution per axis

    Returns
    -------
    grid_x   : (n_grid, n_grid, D) float32  — all grid inputs
    v1_ax    : (n_grid,)  — axis values for bit group_bits[0]
    v2_ax    : (n_grid,)  — axis values for bit group_bits[1]
    """
    D = len(x_baseline)
    v1_ax = np.linspace(v_range[0], v_range[1], n_grid, dtype=np.float32)
    v2_ax = np.linspace(v_range[0], v_range[1], n_grid, dtype=np.float32)

    grid_x = np.tile(x_baseline.astype(np.float32), (n_grid, n_grid, 1))  # (n,n,D)
    V1, V2 = np.meshgrid(v1_ax, v2_ax, indexing='ij')   # (n,n)
    grid_x[:, :, group_bits[0]] = V1
    grid_x[:, :, group_bits[1]] = V2

    return grid_x, v1_ax, v2_ax


def make_two_sample_plane_grid(x_a, x_b, alpha_range=(-0.3, 1.3),
                                beta_range=(-0.5, 0.5), n_grid=30):
    """
    Two-sample interpolation plane:
      x(α, β) = (1-α)*x_a + α*x_b  +  β * perp
    where perp is a unit vector orthogonal to (x_b - x_a) in R^D.

    α=0 → x_a,  α=1 → x_b.  β adds an out-of-plane component.

    Parameters
    ----------
    x_a, x_b   : (D,) float  — two anchor points
    alpha_range : (lo, hi)    — interpolation range (0..1 is between the samples)
    beta_range  : (lo, hi)    — perpendicular range
    n_grid      : int

    Returns
    -------
    grid_x  : (n_grid, n_grid, D) float32
    alpha_ax, beta_ax : axes
    v_ab    : (D,) unit vector from a to b
    v_perp  : (D,) unit vector perpendicular to ab
    """
    x_a = np.asarray(x_a, dtype=np.float32)
    x_b = np.asarray(x_b, dtype=np.float32)
    ab  = x_b - x_a
    v_ab = ab / (np.linalg.norm(ab) + 1e-8)

    # find a random perpendicular direction
    rng   = np.random.default_rng(0)
    rand  = rng.standard_normal(len(x_a)).astype(np.float32)
    rand -= rand.dot(v_ab) * v_ab
    v_perp = rand / (np.linalg.norm(rand) + 1e-8)

    alpha_ax = np.linspace(alpha_range[0], alpha_range[1], n_grid, dtype=np.float32)
    beta_ax  = np.linspace(beta_range[0],  beta_range[1],  n_grid, dtype=np.float32)
    A, B     = np.meshgrid(alpha_ax, beta_ax, indexing='ij')  # (n,n)

    grid_x = (x_a[None, None, :]
               + A[:, :, None] * ab[None, None, :]
               + B[:, :, None] * v_perp[None, None, :])

    return grid_x.astype(np.float32), alpha_ax, beta_ax, v_ab, v_perp


def make_bit_flip_plane_grid(x_baseline, flip_bit, free_bit,
                              v_range=(-2.5, 2.5), n_grid=30):
    """
    Valid↔Invalid transition plane:
      - axis 1 (flip_bit):  varies a single bit that controls rule validity
      - axis 2 (free_bit):  varies an independent bit freely

    All other bits fixed to x_baseline.

    Returns
    -------
    grid_x : (n_grid, n_grid, D) float32
    v1_ax, v2_ax : axes for flip_bit and free_bit
    """
    return make_group_plane_grid(x_baseline, [flip_bit, free_bit], v_range, n_grid)


# ── Project score / denoiser onto 2D axes ────────────────────────────────────

def project_to_axes(vec_field, axis1_idx, axis2_idx):
    """
    Extract the 2D components of a vector field along two coordinate axes.

    Parameters
    ----------
    vec_field   : (n_grid, n_grid, D) or (N, D)
    axis1_idx, axis2_idx : int  — dimension indices

    Returns
    -------
    u : component along axis1
    v : component along axis2
    """
    return vec_field[..., axis1_idx], vec_field[..., axis2_idx]


def project_to_basis(vec_field, v1, v2):
    """
    Project vector field onto two arbitrary basis vectors v1, v2.

    Parameters
    ----------
    vec_field : (..., D)
    v1, v2    : (D,) unit vectors

    Returns
    -------
    u, v : (...,) projections
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    u  = (vec_field * v1).sum(axis=-1)
    v  = (vec_field * v2).sum(axis=-1)
    return u, v


def denoiser_pull(D_flat, x_flat, axis1_idx, axis2_idx):
    """
    Displacement (D - x) projected onto two axes.
    Shows direction and magnitude of denoiser 'pull'.
    """
    diff = D_flat - x_flat
    return diff[..., axis1_idx], diff[..., axis2_idx]


# ── High-level evaluation on a 2D grid ───────────────────────────────────────

def eval_field_on_grid(model, grid_x, sigma, device='cpu', batch_size=1024):
    """
    Evaluate denoiser and score on a (n1, n2, D) grid.

    Returns
    -------
    result dict with keys:
      'D'       : (n1, n2, D)  denoiser output
      'score'   : (n1, n2, D)  score = (D - x) / σ²
      'D_pull'  : (n1, n2, D)  D - x  (denoiser displacement)
      'mag_score' : (n1, n2)   ||score||
      'mag_pull'  : (n1, n2)   ||D - x||
    """
    n1, n2, D = grid_x.shape
    x_flat = torch.from_numpy(grid_x.reshape(-1, D).astype(np.float32))

    score_np, D_np = eval_score(model, x_flat, sigma, device, batch_size)
    score_np  = score_np.reshape(n1, n2, D)
    D_np      = D_np.reshape(n1, n2, D)
    D_pull_np = D_np - grid_x

    return dict(
        x      = grid_x,
        D      = D_np,
        score  = score_np,
        D_pull = D_pull_np,
        mag_score = np.linalg.norm(score_np, axis=-1),
        mag_pull  = np.linalg.norm(D_pull_np, axis=-1),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_vector_field_2d(ax, v1_ax, v2_ax, field_2d,
                          u, v,
                          quiver_stride=3,
                          quiver_scale=None,
                          cmap='RdBu_r',
                          clim=None,
                          xlabel='bit 1', ylabel='bit 2',
                          title='',
                          valid_corners=None,
                          arrow_color='k',
                          arrow_alpha=0.8):
    """
    2D heatmap of a scalar field + quiver overlay of a vector field.

    Parameters
    ----------
    ax          : matplotlib Axes
    v1_ax, v2_ax : (n,) grid axes (x and y)
    field_2d    : (n1, n2)  scalar field for heatmap (e.g. score magnitude)
    u, v        : (n1, n2)  vector components for quiver
    quiver_stride : int  — subsample quiver arrows
    quiver_scale  : float or None  — passed to ax.quiver; None = auto
    cmap        : colormap for heatmap
    clim        : (vmin, vmax) or None
    xlabel, ylabel, title : str
    valid_corners : list of (v1, v2) tuples to mark as valid {-1,+1} corners
    arrow_color : str or 'mag' (color by vector magnitude)
    arrow_alpha : float
    """
    V1, V2 = np.meshgrid(v1_ax, v2_ax, indexing='ij')

    # heatmap
    vmin, vmax = (clim if clim else (field_2d.min(), field_2d.max()))
    im = ax.pcolormesh(v1_ax, v2_ax, field_2d.T, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    # quiver (subsampled)
    s = quiver_stride
    V1s = V1[::s, ::s]; V2s = V2[::s, ::s]
    us  = u[::s, ::s];  vs  = v[::s, ::s]

    if arrow_color == 'mag':
        mag = np.sqrt(us**2 + vs**2)
        ax.quiver(V1s, V2s, us, vs, mag,
                  cmap='hot_r', alpha=arrow_alpha, scale=quiver_scale,
                  angles='xy', scale_units='xy' if quiver_scale else None)
    else:
        ax.quiver(V1s, V2s, us, vs,
                  color=arrow_color, alpha=arrow_alpha, scale=quiver_scale,
                  angles='xy', scale_units='xy' if quiver_scale else None)

    # mark valid corners
    if valid_corners:
        for vc in valid_corners:
            ax.plot(*vc, 'g*', ms=14, zorder=10)

    # mark {-1, +1} grid corners
    for xi in [-1, 1]:
        for yi in [-1, 1]:
            ax.plot(xi, yi, 'w+', ms=8, mew=1.5, zorder=9)

    ax.set_xlim(v1_ax[0], v1_ax[-1])
    ax.set_ylim(v2_ax[0], v2_ax[-1])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
    return im


def plot_denoiser_target_2d(ax, v1_ax, v2_ax, D_u, D_v,
                             xlabel='bit 1', ylabel='bit 2', title='',
                             cmap='bwr', quiver_stride=3, valid_corners=None):
    """
    Visualize denoiser output D(x,σ) projected onto 2D axes.
    D encodes the attractor target — more constant than the score.

    Uses:
      - Background heatmap: D_u (denoiser output along axis 1)
      - Arrows: (D_u - grid_v1, D_v - grid_v2) = displacement toward attractor
    """
    n1, n2 = D_u.shape
    V1, V2 = np.meshgrid(v1_ax, v2_ax, indexing='ij')

    # displacement arrows: where the denoiser "wants" the point to go
    disp_u = D_u - V1
    disp_v = D_v - V2

    im = ax.pcolormesh(v1_ax, v2_ax, D_u.T, cmap=cmap,
                       vmin=-1.5, vmax=1.5, shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='D₁(x,σ)')

    s = quiver_stride
    ax.quiver(V1[::s,::s], V2[::s,::s], disp_u[::s,::s], disp_v[::s,::s],
              color='k', alpha=0.7, angles='xy')

    if valid_corners:
        for vc in valid_corners:
            ax.plot(*vc, 'g*', ms=14, zorder=10)
    for xi in [-1, 1]:
        for yi in [-1, 1]:
            ax.plot(xi, yi, 'w+', ms=8, mew=1.5, zorder=9)

    ax.set_xlim(v1_ax[0], v1_ax[-1]); ax.set_ylim(v2_ax[0], v2_ax[-1])
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)


def multi_sigma_checkpoint_grid(model_dict, grid_x, sigmas, group_bits,
                                 v1_ax, v2_ax,
                                 field='score_mag',
                                 device='cpu',
                                 valid_corners=None,
                                 suptitle=''):
    """
    Plot a grid: rows = checkpoints, columns = sigma levels.
    model_dict : OrderedDict {label: model}  (e.g. {'ep 1k': model1, ...})
    field      : 'score_mag' | 'score_quiver' | 'denoiser'

    Returns fig
    """
    n_rows = len(model_dict)
    n_cols = len(sigmas)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.2 * n_rows),
                              squeeze=False)

    a1, a2 = group_bits
    xlabel = f'bit {a1}'; ylabel = f'bit {a2}'

    for row, (ckpt_label, model) in enumerate(model_dict.items()):
        for col, sigma in enumerate(sigmas):
            ax = axes[row][col]
            res = eval_field_on_grid(model, grid_x, sigma, device)

            if field == 'score_mag':
                u, v = project_to_axes(res['score'], a1, a2)
                plot_vector_field_2d(ax, v1_ax, v2_ax, res['mag_score'],
                                     u, v, quiver_stride=4,
                                     xlabel=xlabel, ylabel=ylabel,
                                     title=f'{ckpt_label}  σ={sigma:.2f}',
                                     valid_corners=valid_corners,
                                     cmap='viridis')
            elif field == 'score_quiver':
                u, v = project_to_axes(res['score'], a1, a2)
                # divergence-free colormap centered at 0
                mag = res['mag_score']
                lim = np.percentile(mag, 95)
                plot_vector_field_2d(ax, v1_ax, v2_ax, mag,
                                     u, v, quiver_stride=4,
                                     xlabel=xlabel, ylabel=ylabel,
                                     title=f'{ckpt_label}  σ={sigma:.2f}',
                                     valid_corners=valid_corners,
                                     cmap='hot', clim=(0, lim),
                                     arrow_color='white', arrow_alpha=0.7)
            elif field == 'denoiser':
                Du, Dv = project_to_axes(res['D'], a1, a2)
                plot_denoiser_target_2d(ax, v1_ax, v2_ax, Du, Dv,
                                        xlabel=xlabel, ylabel=ylabel,
                                        title=f'{ckpt_label}  σ={sigma:.2f}',
                                        valid_corners=valid_corners)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


# ── Parity utils: which bits belong to which group ────────────────────────────

def parity_group_bits(group_idx, group_size, n_bits=36):
    """
    Return bit indices for a given parity group.
    Groups are contiguous: group k covers bits [k*group_size, (k+1)*group_size).

    E.g. G=2, group_idx=0 → [0, 1]
         G=3, group_idx=1 → [3, 4, 5]
    """
    start = group_idx * group_size
    return list(range(start, min(start + group_size, n_bits)))


def valid_parity_corners(group_bits_idx, parity_val=0):
    """
    For a 2-bit parity group, return the two valid {-1,+1}² corners.
    parity_val=0 (even): valid iff b1*b2 = +1 → {(-1,-1),(+1,+1)}
    parity_val=1 (odd) : valid iff b1*b2 = -1 → {(-1,+1),(+1,-1)}
    """
    if parity_val == 0:
        return [(-1, -1), (1, 1)]
    else:
        return [(-1, 1), (1, -1)]
