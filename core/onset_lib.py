"""
onset_lib.py
------------
Utilities for detecting rule-learning and memorization onset from TensorBoard
training logs, using a sustained-threshold criterion.

Onset definition
----------------
  Rule learning : Sample_Accuracy  > acc_thresh  for n_consec consecutive eval pts
  Memorization  : Sample_Mem_Ratio > mem_thresh  for n_consec consecutive eval pts

Returns the *first* step at which the run of n_consec consecutive crossings ends,
projected back to the step where that run began.  Returns np.nan if never reached.

Typical usage
-------------
    from core.onset_lib import get_onsets, collect_onsets, first_sustained_crossing

    rule_step, mem_step = get_onsets("GPT_mini_parity_N4096_D36_G6_even_lr1e4",
                                      saveroot=SAVEROOT)

    G_VALS = [2, 3, 4, 6, 9, 12, 18, 36]
    exps   = [(G, f"GPT_mini_parity_N4096_D36_G{G}_even") for G in G_VALS]
    params, rules, mems = collect_onsets(exps, saveroot=SAVEROOT, max_step=1e5)
"""

import os
import numpy as np

# ── default TB tag names ──────────────────────────────────────────────────────
TAG_ACC = "Eval/Sample_Accuracy"
TAG_MEM = "Eval/Sample_Mem_Ratio"


def first_sustained_crossing(steps, vals, threshold, n_consec=5, above=True):
    """Return the first step at which `vals` crosses `threshold` for
    `n_consec` consecutive evaluation points.

    Parameters
    ----------
    steps    : array-like of int   — eval step numbers
    vals     : array-like of float — metric values at each step
    threshold: float
    n_consec : int                 — required consecutive crossings (default 5)
    above    : bool                — True = look for vals > threshold (default)

    Returns
    -------
    float : the step number at the *start* of the first sustained run,
            or np.nan if the threshold is never sustained.
    """
    steps = np.asarray(steps)
    vals  = np.asarray(vals)
    mask  = vals > threshold if above else vals < threshold
    count = 0
    for i, m in enumerate(mask):
        if m:
            count += 1
            if count >= n_consec:
                return float(steps[i - n_consec + 1])
        else:
            count = 0
    return np.nan


def load_eval_timeseries(exp_name, saveroot):
    """Load full eval time series for one experiment.

    Handles both formats automatically:
      - TensorBoard (newer GPT / DiT sweep runs)
      - mem_eval_stats.csv (older DiT baseline runs)

    Returns
    -------
    dict with keys:
      eval_steps : np.ndarray (T,)
      valid_acc  : np.ndarray (T,)   — Sample_Accuracy / sample_corr_acc
      mem_ratio  : np.ndarray (T,)   — Sample_Mem_Ratio / sample_mem_ratio
    or None if the experiment directory is not found.
    """
    exp_dir = os.path.join(saveroot, exp_name)
    if not os.path.isdir(exp_dir):
        return None

    tb_dir = os.path.join(exp_dir, "tensorboard")

    if os.path.isdir(tb_dir):
        from scripts.plot_tb_curves import load_tb_scalars
        d = load_tb_scalars(tb_dir, [TAG_ACC, TAG_MEM])
        if TAG_ACC not in d:
            return None
        steps = np.array(d[TAG_ACC]["steps"])
        acc   = np.array(d[TAG_ACC]["vals"])
        mem   = np.array(d[TAG_MEM]["vals"]) if TAG_MEM in d else np.full(len(steps), np.nan)
        n = min(len(steps), len(mem))
        return dict(eval_steps=steps[:n], valid_acc=acc[:n], mem_ratio=mem[:n])

    # CSV fallback
    csv_data = _load_from_csv(exp_dir)
    if csv_data is None:
        return None
    return dict(eval_steps=csv_data["acc_steps"],
                valid_acc=csv_data["acc_vals"],
                mem_ratio=csv_data["mem_vals"])


def _load_from_csv(exp_dir):
    """Load eval stats from older DiT CSV format.

    Older DiT runs have no TensorBoard dir; instead they write:
      mem_eval_stats.csv  — columns: step, sample_corr_acc, sample_mem_ratio, ...

    Returns dict with keys 'acc_steps','acc_vals','mem_steps','mem_vals',
    or None if the CSV is not found.
    """
    import pandas as pd
    csv_path = os.path.join(exp_dir, "mem_eval_stats.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "sample_corr_acc" not in df.columns or "sample_mem_ratio" not in df.columns:
        return None
    return {
        "acc_steps": df["step"].to_numpy(),
        "acc_vals":  df["sample_corr_acc"].to_numpy(),
        "mem_steps": df["step"].to_numpy(),
        "mem_vals":  df["sample_mem_ratio"].to_numpy(),
    }


def get_onsets(exp_name, saveroot,
               acc_thresh=0.9, mem_thresh=0.5, n_consec=5,
               acc_tag=TAG_ACC, mem_tag=TAG_MEM):
    """Load eval data and return (rule_onset_step, mem_onset_step).

    Tries TensorBoard first (newer runs); falls back to mem_eval_stats.csv
    for older DiT runs that predate TensorBoard logging.

    Parameters
    ----------
    exp_name  : str   — experiment folder name (relative to saveroot)
    saveroot  : str   — root directory containing experiment folders
    acc_thresh: float — accuracy threshold for rule-learning onset (default 0.9)
    mem_thresh: float — mem-ratio threshold for memorization onset (default 0.5)
    n_consec  : int   — consecutive eval points required (default 5)
    acc_tag   : str   — TensorBoard tag for accuracy (ignored for CSV fallback)
    mem_tag   : str   — TensorBoard tag for mem ratio (ignored for CSV fallback)

    Returns
    -------
    (rule_onset, mem_onset) : (float, float)
        Step numbers, or np.nan if a threshold was never sustainedly crossed.
    """
    exp_dir = os.path.join(saveroot, exp_name)
    tb_dir  = os.path.join(exp_dir, "tensorboard")
    rule = np.nan
    mem  = np.nan

    if os.path.isdir(tb_dir):
        # ── newer run: TensorBoard ────────────────────────────────────────
        from scripts.plot_tb_curves import load_tb_scalars
        d = load_tb_scalars(tb_dir, [acc_tag, mem_tag])
        if acc_tag in d:
            rule = first_sustained_crossing(
                d[acc_tag]["steps"], d[acc_tag]["vals"], acc_thresh, n_consec)
        if mem_tag in d:
            mem = first_sustained_crossing(
                d[mem_tag]["steps"], d[mem_tag]["vals"], mem_thresh, n_consec)
    else:
        # ── older DiT run: CSV fallback ───────────────────────────────────
        csv = _load_from_csv(exp_dir)
        if csv is not None:
            rule = first_sustained_crossing(
                csv["acc_steps"], csv["acc_vals"], acc_thresh, n_consec)
            mem = first_sustained_crossing(
                csv["mem_steps"], csv["mem_vals"], mem_thresh, n_consec)

    return rule, mem


def collect_onsets(exp_list, saveroot, max_step,
                   acc_thresh=0.9, mem_thresh=0.5, n_consec=5):
    """Collect onsets for a list of (param_value, exp_name) pairs.

    Parameters
    ----------
    exp_list  : list of (param_val, exp_name) tuples
                  param_val — the sweep parameter (e.g. lr or wd value)
                  exp_name  — folder name relative to saveroot
    saveroot  : str
    max_step  : float — training budget; substituted when threshold not reached
    acc_thresh, mem_thresh, n_consec : passed to get_onsets

    Returns
    -------
    params : np.ndarray  shape (N,)
    rules  : np.ndarray  shape (N,)  — rule onset steps (max_step if not reached)
    mems   : np.ndarray  shape (N,)  — mem onset steps  (max_step if not reached)
    """
    params, rules, mems = [], [], []
    for param_val, exp_name in exp_list:
        r, m = get_onsets(exp_name, saveroot,
                          acc_thresh=acc_thresh, mem_thresh=mem_thresh,
                          n_consec=n_consec)
        params.append(param_val)
        rules.append(r if not np.isnan(r) else max_step)
        mems.append( m if not np.isnan(m) else max_step)
    return np.array(params), np.array(rules), np.array(mems)
