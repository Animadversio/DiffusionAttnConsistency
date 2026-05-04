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
TAG_ACC        = "Eval/Sample_Accuracy"
TAG_MEM        = "Eval/Sample_Mem_Ratio"
TAG_PERGROUP   = "Eval/PerGroup_Accuracy"
TAG_NAN_1E1    = "Eval/NaN_Ratio"          # eps=1e-1 (loose quantization)
TAG_NAN_1E2    = "Eval/NaN_Ratio_1e2"      # eps=1e-2 (strict) — if logged separately
TAG_BITGRP_MEM = "Eval/BitGroup_Mem_Ratio"


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
      - TensorBoard (newer GPT / DiT sweep runs): loads all available tags
      - mem_eval_stats.csv (older DiT baseline runs)

    Returns
    -------
    dict with keys (all np.ndarray of shape (T,)):
      eval_steps      — training step
      valid_acc       — Sample_Accuracy / sample_corr_acc
      mem_ratio       — Sample_Mem_Ratio / sample_mem_ratio
      pergroup_acc    — PerGroup_Accuracy (NaN if not logged)
      nan_ratio_1e1   — NaN_Ratio at eps=1e-1 (loose quantization)
      nan_ratio_1e2   — NaN_Ratio at eps=1e-2 (strict); NaN for TB runs that don't log it
      bitgroup_mem    — BitGroup_Mem_Ratio (NaN if not logged)
    or None if the experiment directory is not found.
    """
    exp_dir = os.path.join(saveroot, exp_name)
    if not os.path.isdir(exp_dir):
        return None

    tb_dir = os.path.join(exp_dir, "tensorboard")

    if os.path.isdir(tb_dir):
        from scripts.plot_tb_curves import load_tb_scalars
        all_tags = [TAG_ACC, TAG_MEM, TAG_PERGROUP, TAG_NAN_1E1, TAG_NAN_1E2, TAG_BITGRP_MEM]
        d = load_tb_scalars(tb_dir, all_tags)
        if TAG_ACC not in d:
            return None

        # Use ACC steps as reference; align other tags by step value (not position)
        ref_steps = np.array(d[TAG_ACC]["steps"])

        def _align(tag):
            """Return values aligned to ref_steps; NaN where step not present."""
            if tag not in d:
                return np.full(len(ref_steps), np.nan)
            t_steps = np.array(d[tag]["steps"])
            t_vals  = np.array(d[tag]["vals"])
            if np.array_equal(t_steps, ref_steps):
                return t_vals          # fast path: already aligned
            # map step → val via dict lookup
            step_to_val = dict(zip(t_steps.tolist(), t_vals.tolist()))
            return np.array([step_to_val.get(s, np.nan) for s in ref_steps])

        return dict(
            eval_steps    = ref_steps,
            valid_acc     = _align(TAG_ACC),
            mem_ratio     = _align(TAG_MEM),
            pergroup_acc  = _align(TAG_PERGROUP),
            nan_ratio_1e1 = _align(TAG_NAN_1E1),   # eps=1e-1 (loose)
            nan_ratio_1e2 = _align(TAG_NAN_1E2),   # eps=1e-2 (strict); NaN if not logged
            bitgroup_mem  = _align(TAG_BITGRP_MEM),
        )

    # CSV fallback (older DiT runs)
    csv_data = _load_from_csv(exp_dir)
    if csv_data is None:
        return None
    return dict(
        eval_steps    = csv_data["acc_steps"],
        valid_acc     = csv_data["acc_vals"],
        mem_ratio     = csv_data["mem_vals"],
        pergroup_acc  = csv_data["pergroup_acc"],
        nan_ratio_1e1 = csv_data["nan_ratio_1e1"],
        nan_ratio_1e2 = csv_data["nan_ratio_1e2"],
        bitgroup_mem  = csv_data["bitgroup_mem"],
    )


def _load_from_csv(exp_dir):
    """Load eval stats from older DiT CSV format.

    Older DiT runs have no TensorBoard dir; instead they write:
      mem_eval_stats.csv  — columns: step, sample_corr_acc, sample_mem_ratio,
                            pergroup_parity_acc, nan_ratio_eps_1e-1,
                            bitgroup_mem_ratio, ...

    Returns dict with keys 'acc_steps','acc_vals','mem_vals','pergroup_acc',
    'nan_ratio','bitgroup_mem', or None if the CSV is not found.
    """
    import pandas as pd
    csv_path = os.path.join(exp_dir, "mem_eval_stats.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "sample_corr_acc" not in df.columns:
        return None

    def _col(name, alt=None):
        if name in df.columns:
            return df[name].to_numpy()
        if alt and alt in df.columns:
            return df[alt].to_numpy()
        return np.full(len(df), np.nan)

    return {
        "acc_steps":    df["step"].to_numpy(),
        "acc_vals":     _col("sample_corr_acc"),
        "mem_vals":     _col("sample_mem_ratio"),
        "pergroup_acc": _col("pergroup_parity_acc"),
        "nan_ratio_1e1": _col("nan_ratio_eps_1e-1"),
        "nan_ratio_1e2": _col("nan_ratio_eps_1e-2"),
        "bitgroup_mem": _col("bitgroup_mem_ratio"),
    }


def load_eval_dataframe(exp_name, saveroot):
    """Load eval time series for one experiment as a pandas DataFrame.

    Each row is one eval step. Includes exp_name column for traceability.
    Handles both CSV (older DiT runs) and TensorBoard (newer runs).

    Returns
    -------
    pd.DataFrame with columns:
      exp_name, step, valid_acc, mem_ratio, pergroup_acc,
      nan_ratio_1e1, nan_ratio_1e2, bitgroup_mem
    or None if the experiment is not found.
    """
    import pandas as pd
    ts = load_eval_timeseries(exp_name, saveroot)
    if ts is None:
        return None
    df = pd.DataFrame({
        "exp_name":     exp_name,
        "step":         ts["eval_steps"],
        "valid_acc":    ts["valid_acc"],
        "mem_ratio":    ts["mem_ratio"],
        "pergroup_acc": ts["pergroup_acc"],
        "nan_ratio_1e1": ts["nan_ratio_1e1"],
        "nan_ratio_1e2": ts["nan_ratio_1e2"],
        "bitgroup_mem": ts["bitgroup_mem"],
    })
    return df


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
