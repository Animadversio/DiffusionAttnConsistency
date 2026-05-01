#!/usr/bin/env python3
"""
Integrated analysis script for G3 rep2 (DiT_mini_parity_N4096_D36_G3_even_rep2).

Produces six publication-ready figures (PNG + PDF, fonttype=42, no right/top spines):

  01_sample_raster               — per-sample 4-state trajectory raster
  02_transition_heatmap_*        — transition count/probability heatmaps at two windows
  03_vector_field_checkpoints    — score landscape at ep 49, 14251, 119377, 492388
  04_dsm_loss_sigma02_2_vs_step  — DSM loss in σ∈[0.2,2] vs training step
  05_dsm_loss_vs_sigma_loglog    — loss vs σ (log-log) at four checkpoints
  06_attractor_basin_profiles    — attractor basin profiles along three directions

All vector field / basin data is read from existing caches — no new model evaluations.

Usage:
  python scripts/analyze_G3rep2_integrated.py
  python scripts/analyze_G3rep2_integrated.py --figdir /tmp/G3rep2_figs
"""

import os, sys, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, "/n/home12/binxuwang/Github/DiffusionAttnConsistency")

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top']   = False
plt.rcParams['figure.dpi'] = 120

from core.vector_field_lib import (
    load_training_data, make_plane_hash, project_to_basis,
    DEFAULT_SAVEROOT,
)
from core.basin_lib import basin_plot_profiles
from scripts.plot_sample_evolution import (
    load_data, plot_raster, compute_transition_matrix,
    plot_transition_heatmap_both,
)
from scripts.plot_sigma_loss_evolution import (
    load_sigma_data, bin_mean, _draw_sigma_panel,
)

# ── Constants ─────────────────────────────────────────────────────────────────
EXP_NAME   = "DiT_mini_parity_N4096_D36_G3_even_rep2"
SAVEROOT   = DEFAULT_SAVEROOT
EXP_DIR    = os.path.join(SAVEROOT, EXP_NAME)

VF_EPOCHS  = [49, 14251, 119377, 492388]
VF_SIGMA   = 1.0
VF_CACHE   = os.path.join(EXP_DIR, "vector_field_cache")
PLANE_HASH = "191059703a35"
RANGE_TAG  = "a-1.75_3.75"
NGRID      = 50

BASIN_EPOCHS = [7017, 20309, 492388]
BASIN_SIGMA  = 1.0
BASIN_N      = 30
BASIN_CACHE  = os.path.join(EXP_DIR, "basin_analysis", "line_cache")
BASIN_LABELS = {
    7017:   "ep 7017 (pre rule-learning)",
    20309:  "ep 20309 (post rule-learning)",
    492388: "ep 492388 (memorization onset)",
}
BASIN_COLORS = {7017: 'C0', 20309: 'C1', 492388: 'C2'}

SIGMA_LOSS_IDX  = [10, 26, 32, 36]
SIGMA_LOSS_LBLS = [
    'ep 49\n(pre rule)', 'ep 14,251\n(rule plateau)',
    'ep 119,377\n(rule learned)', 'ep 492,388\n(mem onset)',
]

SPLIT_STYLES = {
    'train':  dict(color='#2166ac', lw=1.8, label='Train (in-dist)'),
    'test':   dict(color='#d73027', lw=1.8, label='Test (valid, unseen)'),
    'random': dict(color='#555555', lw=1.4, ls='--', label='Random ±1'),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig_both(fig, figdir, name):
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(figdir, name + ext), dpi=300, bbox_inches='tight')
    print(f"  → {name}.png/pdf")


def load_vf_cache(epoch, sigma=VF_SIGMA, ngrid=NGRID):
    fname = os.path.join(VF_CACHE,
        f"vf_ep{epoch:06d}_{PLANE_HASH}_{RANGE_TAG}_sig{sigma:.4f}_n{ngrid}.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    return np.load(fname)


# ── Panel functions ───────────────────────────────────────────────────────────

def panel1_raster(d, figdir):
    print("Panel 1: sample raster ...")
    fig = plot_raster(d, EXP_NAME, figdir=figdir, save=False)
    savefig_both(fig, figdir, '01_sample_raster')
    plt.close(fig)


def panel2_transition(d, figdir):
    print("Panel 2: transition heatmaps ...")
    T_count, T_prob, epochs_ev, ep_mid = compute_transition_matrix(d)
    for win_label, win_lo, win_hi in [
        ('rule_learning', 1_000,  50_000),
        ('memorization', 50_000, 800_000),
    ]:
        mask = (ep_mid >= win_lo) & (ep_mid < win_hi)
        T_win = T_count[mask].sum(axis=0)
        fig = plot_transition_heatmap_both(
            T_win,
            title=f"Transition counts — ep {win_lo//1000}k–{win_hi//1000}k",
        )
        savefig_both(fig, figdir, f'02_transition_heatmap_{win_label}')
        plt.close(fig)


def panel3_vector_field(figdir):
    print("Panel 3: vector field ...")
    x_train = load_training_data(EXP_NAME, saveroot=SAVEROOT).numpy()
    x_a = x_train[0].copy()
    # x_b: valid novel — flip bits 0+1 (parity preserved in group-0)
    x_b = x_a.copy(); x_b[0] *= -1; x_b[1] *= -1
    # x_c: invalid — flip bit-0 only (breaks group-0 parity)
    x_c = x_a.copy(); x_c[0] *= -1

    # Verify hash
    computed_hash = make_plane_hash(x_a, x_b, x_c)
    if not computed_hash.startswith(PLANE_HASH):
        print(f"  [warn] plane hash mismatch: {computed_hash} vs {PLANE_HASH}")

    # L2-scaled basis
    ab = x_b - x_a
    ac = x_c - x_a
    L_ab = float(np.linalg.norm(ab)); v_ab = ab / L_ab
    ac_perp = ac - ac.dot(v_ab) * v_ab
    L_perp  = float(np.linalg.norm(ac_perp)); v_ac = ac_perp / L_perp

    margin = 1.75
    alpha_ax = np.linspace(-margin, L_ab + margin, NGRID)
    beta_ax  = np.linspace(-margin, L_perp + margin, NGRID)
    A_g, B_g = np.meshgrid(alpha_ax, beta_ax, indexing='ij')

    xc_alpha = float(ac.dot(v_ab))
    xc_beta  = float(np.linalg.norm(ac_perp))
    markers = [
        (0,       0,        'x_a', 'white'),
        (L_ab,    0,        'x_b', 'cyan'),
        (xc_alpha, xc_beta, 'x_c', 'yellow'),
        (xc_alpha, -xc_beta,'x_d', 'orange'),
    ]

    fig, axes = plt.subplots(2, len(VF_EPOCHS), figsize=(16, 7),
                              gridspec_kw={'hspace': 0.15, 'wspace': 0.07})

    for ei, epoch in enumerate(VF_EPOCHS):
        try:
            res = load_vf_cache(epoch)
        except FileNotFoundError as e:
            print(f"  [warn] cache missing: {e}")
            continue

        disp_u, disp_v = project_to_basis(res['D_pull'], v_ab, v_ac)
        disp_u = disp_u.reshape(NGRID, NGRID)
        disp_v = disp_v.reshape(NGRID, NGRID)
        score_mag = np.linalg.norm(res['score'].reshape(NGRID, NGRID, -1), axis=2)

        # Row 0: score magnitude + displacement arrows
        ax0 = axes[0, ei]
        ax0.pcolormesh(alpha_ax, beta_ax, score_mag.T, cmap='magma', shading='auto',
                       rasterized=True, zorder=0)
        ax0.set_rasterization_zorder(1)
        stride = max(1, NGRID // 12)
        ax0.quiver(A_g[::stride, ::stride], B_g[::stride, ::stride],
                   disp_u[::stride, ::stride], disp_v[::stride, ::stride],
                   color='white', alpha=0.7, scale=30, width=0.004)
        for mx, my, mlbl, mc in markers:
            ax0.plot(mx, my, 'o', color=mc, ms=7, mec='black', mew=0.8)
            ax0.text(mx, my + 0.25, mlbl, color=mc, fontsize=7, ha='center')
        ax0.set_xlim(alpha_ax[0], alpha_ax[-1])
        ax0.set_ylim(beta_ax[0],  beta_ax[-1])
        ax0.set_aspect('equal')
        ax0.set_title(f"ep {epoch:,}", fontsize=10)
        if ei == 0: ax0.set_ylabel('β (L2)', fontsize=9)

        # Row 1: D_pull · v_ab
        Du_proj = disp_u  # already projected onto v_ab
        vmax = float(np.percentile(np.abs(Du_proj), 95))
        ax1 = axes[1, ei]
        ax1.pcolormesh(alpha_ax, beta_ax, Du_proj.T,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto',
                       rasterized=True, zorder=0)
        ax1.set_rasterization_zorder(1)
        ax1.contour(alpha_ax, beta_ax, Du_proj.T, levels=[0],
                    colors='k', linewidths=0.8)
        for mx, my, mlbl, mc in markers:
            ax1.plot(mx, my, 'o', color=mc, ms=7, mec='black', mew=0.8)
        ax1.set_xlim(alpha_ax[0], alpha_ax[-1])
        ax1.set_ylim(beta_ax[0],  beta_ax[-1])
        ax1.set_aspect('equal')
        ax1.set_xlabel('α (L2)', fontsize=9)
        if ei == 0:
            ax1.set_ylabel('β (L2)', fontsize=9)

    axes[0, 0].text(-0.38, 0.5, 'Score |score|', transform=axes[0, 0].transAxes,
                    fontsize=9, va='center', rotation=90)
    axes[1, 0].text(-0.38, 0.5, r'$D_{pull}\cdot v_{ab}$', transform=axes[1, 0].transAxes,
                    fontsize=9, va='center', rotation=90)
    fig.suptitle(f"{EXP_NAME}  score landscape  σ={VF_SIGMA}", fontsize=12)
    savefig_both(fig, figdir, '03_vector_field_checkpoints')
    plt.close(fig)


def panel4_dsm_loss_vs_step(figdir):
    print("Panel 4: DSM loss vs step ...")
    records = load_sigma_data(EXP_NAME)
    steps = np.array([r['epoch'] for r in records])
    sigma_grid = records[0]['sigma_grid']

    fig, ax = plt.subplots(figsize=(9, 4))
    for split, style in SPLIT_STYLES.items():
        vals = np.array([bin_mean(r[f'loss_{split}'], sigma_grid, 0.2, 2.0) for r in records])
        mask = ~np.isnan(vals)
        ax.semilogy((steps + 1)[mask], vals[mask], **style)

    ax.set_xscale('log')
    ax.set_xlabel('Training step', fontsize=10)
    ax.set_ylabel('MSE loss', fontsize=10)
    ax.set_title(f"{EXP_NAME}  DSM loss  σ∈[0.2, 2.0]", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig_both(fig, figdir, '04_dsm_loss_sigma02_2_vs_step')
    plt.close(fig)
    return records


def panel5_dsm_loss_vs_sigma(records, figdir):
    print("Panel 5: DSM loss vs σ (log-log) ...")
    fig, axes = plt.subplots(1, len(SIGMA_LOSS_IDX), figsize=(16, 4))
    for ax, widx, lbl in zip(axes, SIGMA_LOSS_IDX, SIGMA_LOSS_LBLS):
        _draw_sigma_panel(ax, records[widx], lbl)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(f"{EXP_NAME}  DSM loss vs σ (log-log)", fontsize=12)
    fig.tight_layout()
    savefig_both(fig, figdir, '05_dsm_loss_vs_sigma_loglog')
    plt.close(fig)


def panel6_basin(figdir):
    print("Panel 6: attractor basin profiles ...")
    fig = basin_plot_profiles(
        cache_dir=BASIN_CACHE,
        epochs=BASIN_EPOCHS,
        epoch_labels=BASIN_LABELS,
        epoch_colors=BASIN_COLORS,
        sigma=BASIN_SIGMA,
        n_samples=BASIN_N,
        title=(f"{EXP_NAME}  Attractor basin profiles  σ={BASIN_SIGMA}  "
               f"N={BASIN_N} samples  (5–95% CI of mean)"),
    )
    savefig_both(fig, figdir, '06_attractor_basin_profiles')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figdir', default=os.path.join(
        "/n/home12/binxuwang/Github/DiffusionAttnConsistency",
        "figures", "RuleMemDeepDive", "DiT_mini_G3_N4096_rep2"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"G3 rep2 Integrated Analysis → {args.figdir}")
    print(f"{'='*60}")

    d = load_data(EXP_NAME, SAVEROOT)
    panel1_raster(d, args.figdir)
    panel2_transition(d, args.figdir)
    panel3_vector_field(args.figdir)
    records = panel4_dsm_loss_vs_step(args.figdir)
    panel5_dsm_loss_vs_sigma(records, args.figdir)
    panel6_basin(args.figdir)

    print(f"\nAll figures saved to: {args.figdir}")


if __name__ == '__main__':
    main()
