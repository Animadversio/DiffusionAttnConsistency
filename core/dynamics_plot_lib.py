"""
dynamics_plot_lib.py
--------------------
Utilities for building and plotting training-dynamics heatmaps:
  - valid fraction over log-step × G
  - memorization ratio over log-step × G
  - innovation window (valid − mem)

Typical usage
-------------
    from core.dynamics_plot_lib import build_timeheatmap, plot_dynamics_heatmap
    from core.onset_lib import load_eval_timeseries

    G_VALS = [2, 3, 4, 6, 9, 12, 18, 36]
    model_exp_lists = [
        ("DiT-mini N=4096",
         [(G, f"DiT_mini_parity_N4096_D36_G{G}_even") for G in G_VALS],
         1e6),
        ("GPT-mini N=4096",
         [(G, f"GPT_mini_parity_N4096_D36_G{G}_even") for G in G_VALS],
         1e5),
    ]
    fig = plot_dynamics_heatmap(model_exp_lists, saveroot=SAVEROOT, n_grid=400,
                                figsize=(16, 6))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from core.onset_lib import load_eval_timeseries, get_onsets


def build_timeheatmap(exp_list, saveroot, log_step_grid=None, n_grid=300):
    """
    Load eval time series for each (G, exp_name) and interpolate onto a
    shared log-step grid.

    Parameters
    ----------
    exp_list       : list of (G, exp_name) pairs
    saveroot       : str — root directory containing experiment folders
    log_step_grid  : array or None — pre-built log10(step) grid to use
    n_grid         : int — number of grid points (used when log_step_grid is None)

    Returns
    -------
    log_grid   : (n_grid,)        — log10(step) values
    acc_mat    : (n_G, n_grid)    — valid-fraction interpolated
    mem_mat    : (n_G, n_grid)    — mem-ratio interpolated
    g_labels   : list             — G values in the same order as rows
    """
    all_acc, all_mem, g_labels = [], [], []

    for G, exp_name in exp_list:
        ts = load_eval_timeseries(exp_name, saveroot)
        if ts is None or len(ts["eval_steps"]) == 0:
            all_acc.append(None)
            all_mem.append(None)
        else:
            all_acc.append((ts["eval_steps"], ts["valid_acc"]))
            all_mem.append((ts["eval_steps"], ts["mem_ratio"]))
        g_labels.append(G)

    # Build shared log grid from all available step ranges
    valid_steps = [s for s, _ in all_acc if s is not None]
    smin = min(s.min() for s in valid_steps)
    smax = max(s.max() for s in valid_steps)

    if log_step_grid is not None:
        log_grid = log_step_grid
    else:
        log_grid = np.linspace(np.log10(smin), np.log10(smax), n_grid)
    step_grid = 10 ** log_grid

    def interp_row(data, fill=np.nan):
        if data is None:
            return np.full(len(log_grid), fill)
        steps, vals = data
        f = interp1d(steps, vals, bounds_error=False,
                     fill_value=(vals[0], vals[-1]))
        return f(step_grid)

    acc_mat = np.array([interp_row(d)           for d in all_acc])
    mem_mat = np.array([interp_row(d, fill=0.0) for d in all_mem])
    return log_grid, acc_mat, mem_mat, g_labels


def plot_dynamics_heatmap(model_exp_lists, saveroot, n_grid=300,
                          figsize=None, save_path=None, dpi=150,
                          align_x=False,
                          show_onsets=False,
                          acc_thresh=0.9, mem_thresh=0.5, n_consec=5,
                          onset_marker="*", onset_marker_kw=None):
    """
    Plot valid-fraction and mem-ratio dynamics heatmaps for one or more models.

    Parameters
    ----------
    model_exp_lists : list of (label, exp_list, max_step)
        label    : str  — row label (model name)
        exp_list : list of (G, exp_name) pairs
        max_step : float — training budget
    saveroot   : str
    n_grid     : int
    figsize    : tuple or None
    save_path  : str or None — if given, saves .pdf and .png
    dpi        : int
    align_x    : bool — if True, all rows share the same log-step x-axis range
                        (union of all experiments); default False
    show_onsets     : bool — overlay onset markers on heatmap panels; default False
                             Rule onset (*) on col 0 (valid fraction)
                             Mem onset  (*) on col 1 (mem ratio)
                             Both       (*) on col 2 (innovation)
    acc_thresh      : float — accuracy threshold for rule onset (default 0.9)
    mem_thresh      : float — mem-ratio threshold for mem onset (default 0.5)
    n_consec        : int   — consecutive eval points required (default 5)
    onset_marker    : str   — matplotlib marker string (default "*")
    onset_marker_kw : dict or None — extra kwargs passed to ax.plot for markers
                      Defaults: markersize=10, zorder=5, linestyle="none"
                      col 0 uses color="white", col 1 uses color="white",
                      col 2 uses color="black" — override via onset_marker_kw

    Returns
    -------
    fig : matplotlib Figure
    """
    if onset_marker_kw is None:
        onset_marker_kw = {}

    n_models = len(model_exp_lists)
    if figsize is None:
        figsize = (6 * 3, 3 * n_models)

    # Build a shared log grid across all models if align_x is requested
    shared_log_grid = None
    if align_x:
        all_steps = []
        for _, exp_list, _ in model_exp_lists:
            for _, exp_name in exp_list:
                ts = load_eval_timeseries(exp_name, saveroot)
                if ts is not None and len(ts["eval_steps"]) > 0:
                    all_steps.append(ts["eval_steps"])
        if all_steps:
            smin = min(s.min() for s in all_steps)
            smax = max(s.max() for s in all_steps)
            shared_log_grid = np.linspace(np.log10(smin), np.log10(smax), n_grid)

    fig, axes = plt.subplots(n_models, 3, figsize=figsize, dpi=dpi,
                             squeeze=False)

    for row, (label, exp_list, max_step) in enumerate(model_exp_lists):
        log_grid, acc_mat, mem_mat, g_labels = build_timeheatmap(
            exp_list, saveroot, log_step_grid=shared_log_grid, n_grid=n_grid)
        innov_mat = np.clip(acc_mat - mem_mat, 0, 1)

        n_G    = len(g_labels)
        extent = [log_grid[0], log_grid[-1], -0.5, n_G - 0.5]

        # decade x-ticks
        lo = int(np.floor(log_grid[0]))
        hi = int(np.ceil(log_grid[-1]))
        tick_locs = [p for p in range(lo, hi + 1)]
        tick_labs = [f"$10^{{{p}}}$" for p in tick_locs]

        # Collect onset steps per experiment if needed
        onset_rule = []  # log10(step) or np.nan, one per G
        onset_mem  = []
        if show_onsets:
            for gi, (G, exp_name) in enumerate(exp_list):
                r, m = get_onsets(exp_name, saveroot,
                                  acc_thresh=acc_thresh, mem_thresh=mem_thresh,
                                  n_consec=n_consec)
                onset_rule.append(np.log10(r) if not np.isnan(r) else np.nan)
                onset_mem.append( np.log10(m) if not np.isnan(m) else np.nan)

        panels = [
            (acc_mat,   "Valid fraction",             "YlGnBu", 0, 1),
            (mem_mat,   "Mem ratio",                  "YlOrRd", 0, 1),
            (innov_mat, "Innovation\n(valid − mem)",  "Blues",  0, 1),
        ]
        # default marker colors per column (contrast against each colormap)
        _default_colors = ["white", "white", "black"]

        for col, (mat, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(mat, aspect="auto", origin="lower",
                           extent=extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labs, fontsize=8)
            ax.set_yticks(range(n_G))
            ax.set_yticklabels([f"G={G}" for G in g_labels], fontsize=8)
            ax.set_xlabel("Step", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Onset markers
            if show_onsets:
                mkw = dict(markersize=10, zorder=5, linestyle="none",
                           color=_default_colors[col])
                mkw.update(onset_marker_kw)
                # col 0: rule onset; col 1: mem onset; col 2: both
                if col in (0, 2):
                    xs = [x for x in onset_rule if not np.isnan(x)]
                    ys = [gi for gi, x in enumerate(onset_rule) if not np.isnan(x)]
                    if xs:
                        ax.plot(xs, ys, onset_marker, **mkw)
                if col in (1, 2):
                    xs = [x for x in onset_mem if not np.isnan(x)]
                    ys = [gi for gi, x in enumerate(onset_mem) if not np.isnan(x)]
                    if xs:
                        mkw2 = dict(mkw)
                        if col == 2:
                            mkw2["color"] = mkw2.get("color", "black")
                            mkw2["marker"] = onset_marker
                            mkw2["markerfacecolor"] = "none"
                        ax.plot(xs, ys, onset_marker, **mkw2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path + ".pdf", bbox_inches="tight")
        fig.savefig(save_path + ".png", bbox_inches="tight", dpi=dpi)
        print(f"Saved → {save_path}.{{pdf,png}}")
    return fig
