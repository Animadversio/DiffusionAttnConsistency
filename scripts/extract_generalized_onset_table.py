"""
extract_generalized_onset_table.py
------------------------------------
Batch-extract rule-learning and memorization onset times for ALL task types:
  parity, exactK, rowK, rowVarK (row_variable_k), globalK, rowOnly, latinSq, sudoku

Output columns (NaN where not applicable to a task type):
  exp_name, task_type, model_arch, model_size, rep,
  N, n_size, D, K, K_list, G, encoding, n_layer, n_embd, n_head, lr, wd, nsteps,
  support_size, stat_mem_frac, stat_mem_thresh,

  rule_onset_{85,90,95}   -- overall rule accuracy sustained threshold crossings
  row_onset_90            -- per-row/group accuracy onset (parity, rowK, rowVarK, globalK, rowOnly, latinSq, sudoku)
  col_onset_90            -- per-col accuracy onset (latinSq, sudoku, rowOnly)
  block_onset_90          -- per-block accuracy onset (sudoku only)
  mem_onset_{mem10,mem20,mem35,mem50} -- memorization ratio fixed thresholds
  mem_onset_stat          -- adaptive: 0.10 + stat_mem_frac (if support_size known)
  novel_onset_{01,05,09}  -- novel_valid = rule_acc - mem_ratio

NaN means threshold never sustainedly crossed, or metric not available for this task.

Usage:
    python scripts/extract_generalized_onset_table.py \\
        --saveroot /path/to/DiffusionParityLearning --outdir outputs/
    python scripts/extract_generalized_onset_table.py \\
        --saveroot ... --tasks parity exactK latinSq sudoku
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import comb as sp_comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.onset_lib import first_sustained_crossing
from scripts.plot_tb_curves import load_tb_scalars
from core.row_k_lib import (
    valid_set_size_global_k,
    valid_set_size_row_k,
    valid_set_size_row_variable_k,
)
from core.latin_square_lib import valid_set_size as latin_valid_set_size
from core.latin_square_lib import valid_sudoku_set_size
from core.exact_k_lib import valid_set_size as exactk_valid_set_size

# ── TB tag mapping per task type ───────────────────────────────────────────────
# Maps canonical metric name → TB scalar tag (None = not available for this task).
TASK_TAGS = {
    "parity": {
        "rule_acc":  "Eval/Sample_Accuracy",
        "row_acc":   "Eval/PerGroup_Accuracy",
        "col_acc":   None,
        "block_acc": None,
        "mem_ratio": "Eval/Sample_Mem_Ratio",
    },
    "exactK": {
        "rule_acc":  "eval/k_correct_ratio",
        "row_acc":   None,
        "col_acc":   None,
        "block_acc": None,
        "mem_ratio": "eval/sample_mem_ratio",
        "nan_ratio": "eval/nan_ratio_eps_1e-1",   # needed for renormalization
    },
    "rowK": {
        "rule_acc":  "eval/full_valid_ratio",
        "row_acc":   "eval/per_row_valid_ratio",
        "col_acc":   None,
        "block_acc": None,
        "mem_ratio": "eval/sample_mem_ratio",
    },
    "rowVarK": {
        "rule_acc":  "eval/full_valid_ratio",
        "row_acc":   "eval/per_row_valid_ratio",
        "col_acc":   None,
        "block_acc": None,
        "mem_ratio": "eval/sample_mem_ratio",
    },
    "globalK": {
        "rule_acc":  "eval/full_valid_ratio",
        "row_acc":   "eval/per_row_valid_ratio",
        "col_acc":   None,
        "block_acc": None,
        "mem_ratio": "eval/sample_mem_ratio",
    },
    "rowOnly": {
        "rule_acc":  "eval/full_valid_ratio",
        "row_acc":   "eval/row_valid_ratio",
        "col_acc":   "eval/col_valid_ratio",
        "block_acc": None,
        "mem_ratio": "eval/sample_mem_ratio",
    },
    "latinSq": {
        "rule_acc":  "eval/full_valid_ratio",
        "row_acc":   "eval/row_valid_ratio",
        "col_acc":   "eval/col_valid_ratio",
        "block_acc": None,
        "mem_ratio": "eval/sample_mem_ratio",
    },
    "sudoku": {
        "rule_acc":  "eval/full_valid_ratio",
        "row_acc":   "eval/row_valid_ratio",
        "col_acc":   "eval/col_valid_ratio",
        "block_acc": "eval/block_valid_ratio",
        "mem_ratio": "eval/sample_mem_ratio",
    },
}

ACC_THRESHOLDS = [0.85, 0.90, 0.95]
MEM_THRESHOLDS = [0.10, 0.20, 0.35, 0.50]
NOV_THRESHOLDS = [0.10, 0.50, 0.90]
SUB_THRESHOLD  = 0.90   # single threshold for row/col/block onsets


# ── Task-type detection ────────────────────────────────────────────────────────
def detect_task_type(exp_name, args):
    rule         = args.get("rule", "")
    dataset_name = args.get("dataset_name", "")

    if rule == "row_k":          return "rowK"
    if rule == "row_variable_k": return "rowVarK"
    if rule == "global_k":       return "globalK"
    if rule == "row_only":       return "rowOnly"
    if rule == "sudoku":         return "sudoku"

    if "parity" in dataset_name or args.get("parity") is not None:
        return "parity"
    if "latinSq" in dataset_name or "latinSq" in exp_name:
        return "latinSq"
    if "exactK" in dataset_name or "exactK" in exp_name:
        return "exactK"

    return "unknown"


# ── Architecture / size / rep parsing ─────────────────────────────────────────
_ARCH_RE = re.compile(r"^(DiT|GPT)", re.I)
_SIZE_RE = re.compile(r"^(?:DiT|GPT)_(nano|mini|S|B)(?=_|$)", re.I)
_REP_RE  = re.compile(r"_rep(\d+)$")

def parse_arch_size(exp_name):
    m_arch = _ARCH_RE.match(exp_name)
    m_size = _SIZE_RE.match(exp_name)
    m_rep  = _REP_RE.search(exp_name)
    arch = m_arch.group(1) if m_arch else "unknown"
    size = m_size.group(1) if m_size else "unknown"
    rep  = int(m_rep.group(1)) if m_rep else 1
    return arch, size, rep


# ── Support-size computation ───────────────────────────────────────────────────
def compute_support_size(task_type, args):
    """Return integer support_size, or None if not computable."""
    try:
        if task_type == "parity":
            G = args.get("group_size")
            D = args.get("sample_len")
            if G and D:
                return int((2 ** (G - 1)) ** (D // G))

        elif task_type == "exactK":
            D = args.get("sample_len")
            K = args.get("k_ones")   # exactK stores K as "k_ones"
            if D and K is not None:
                return int(exactk_valid_set_size(D, K))

        elif task_type == "rowK":
            n = args.get("n_size")
            K = args.get("K")
            if n and K is not None:
                return int(valid_set_size_row_k(n, K))

        elif task_type == "rowVarK":
            n      = args.get("n_size")
            K_list = args.get("K_list")
            if n and K_list:
                return int(valid_set_size_row_variable_k(n, K_list))

        elif task_type == "globalK":
            n      = args.get("n_size")
            K_list = args.get("K_list")
            if n and K_list:
                return int(valid_set_size_global_k(n, K_list))

        elif task_type == "rowOnly":
            n = args.get("n_size")
            if n:
                import math
                return int(math.factorial(n) ** n)

        elif task_type == "latinSq":
            n = args.get("n_size")
            if n:
                return int(latin_valid_set_size(n))

        elif task_type == "sudoku":
            n  = args.get("n_size")
            bh = args.get("block_h", 2)
            bw = args.get("block_w", 3)
            if n:
                vs = valid_sudoku_set_size(n, bh, bw)
                return int(vs) if vs is not None else None

    except Exception:
        pass
    return None


# ── Load TB metrics for one experiment ────────────────────────────────────────
def load_metrics(exp_dir, task_type):
    """
    Returns dict: metric_name -> (steps_array, vals_array) or (None, None).
    Falls back to None/None for any tag not found in TB.
    """
    tag_map      = TASK_TAGS.get(task_type, {})
    tags_needed  = [t for t in tag_map.values() if t is not None]

    tb_dir       = os.path.join(exp_dir, "tensorboard")
    has_tb_data  = (
        os.path.isdir(tb_dir) and
        any(f.startswith("events.out") for f in os.listdir(tb_dir))
    )

    raw = load_tb_scalars(tb_dir, tags_needed) if has_tb_data else {}

    metrics = {}
    for metric, tag in tag_map.items():
        if tag is None or tag not in raw:
            metrics[metric] = (None, None)
        else:
            d = raw[tag]
            metrics[metric] = (np.array(d["steps"]), np.array(d["vals"]))
    return metrics


# ── Metric renormalization ─────────────────────────────────────────────────────
def _align_vals(steps_src, vals_src, steps_ref):
    """Align vals_src onto steps_ref via dict lookup; missing steps → NaN."""
    if steps_src is None or vals_src is None:
        return None
    if np.array_equal(steps_src, steps_ref):
        return vals_src
    d = dict(zip(steps_src.tolist(), vals_src.tolist()))
    return np.array([d.get(int(s), np.nan) for s in steps_ref])


def normalize_metrics(metrics, task_type):
    """
    Renormalize rule_acc and mem_ratio so both are fractions of *all* generated
    samples (not conditional on valid/non-NaN subsets).

    - parity      : no change — both already use all-samples denominator
    - exactK      : both are conditional on non-NaN → multiply by (1 - nan_ratio)
    - all others  : rule_acc already all-samples; mem_ratio is P(mem | rule-valid)
                    → convert: mem_ratio_norm = mem_ratio × rule_acc
    """
    if task_type == "parity":
        return metrics

    metrics = dict(metrics)   # shallow copy — don't mutate caller's dict
    steps_rule, vals_rule = metrics.get("rule_acc",  (None, None))
    steps_mem,  vals_mem  = metrics.get("mem_ratio", (None, None))

    if task_type == "exactK":
        steps_nan, vals_nan = metrics.get("nan_ratio", (None, None))
        if steps_rule is not None and steps_nan is not None:
            nan_aligned = _align_vals(steps_nan, vals_nan, steps_rule)
            if nan_aligned is not None:
                non_nan = 1.0 - nan_aligned
                metrics["rule_acc"] = (steps_rule, vals_rule * non_nan)
                if vals_mem is not None:
                    mem_aligned = _align_vals(steps_mem, vals_mem, steps_rule)
                    if mem_aligned is not None:
                        metrics["mem_ratio"] = (steps_rule, mem_aligned * non_nan)
    else:
        # rowK, rowVarK, globalK, rowOnly, latinSq, sudoku
        # mem_ratio = P(mem | rule-valid); convert to P(mem ∩ rule-valid) / all
        if steps_rule is not None and vals_rule is not None and \
           steps_mem  is not None and vals_mem  is not None:
            mem_aligned = _align_vals(steps_mem, vals_mem, steps_rule)
            if mem_aligned is not None:
                metrics["mem_ratio"] = (steps_rule, mem_aligned * vals_rule)

    return metrics


# ── Onset extraction for one run ───────────────────────────────────────────────
def extract_row(exp_name, exp_dir, args, n_consec=5):
    task_type     = detect_task_type(exp_name, args)
    arch, size, rep = parse_arch_size(exp_name)

    N           = args.get("sample_num", np.nan)
    support     = compute_support_size(task_type, args)
    stat_frac   = (N / support) if (support and support > 0) else np.nan
    stat_thresh = (0.10 + stat_frac) if not np.isnan(stat_frac) else np.nan

    row = {
        "exp_name":        exp_name,
        "task_type":       task_type,
        "model_arch":      arch,
        "model_size":      size,
        "rep":             rep,
        "N":               N,
        "n_size":          args.get("n_size",       np.nan),
        "D":               args.get("sample_len",   np.nan),
        "K":               args.get("k_ones", np.nan) if task_type == "exactK"
                           else args.get("K", np.nan) if task_type == "rowK"
                           else np.nan,
        "K_list":          str(args.get("K_list", "")) if task_type in ("rowVarK", "globalK")
                           else "",
        "G":               args.get("group_size",   np.nan),
        "encoding":        args.get("encoding",     ""),
        "n_layer":         args.get("depth",        np.nan),
        "n_embd":          args.get("hidden_size",  np.nan),
        "n_head":          args.get("num_heads",    np.nan),
        "lr":              args.get("lr",            np.nan),
        "wd":              args.get("weight_decay",  np.nan),
        "nsteps":          args.get("nsteps",        np.nan),
        "support_size":    support if support is not None else np.nan,
        "stat_mem_frac":   stat_frac,
        "stat_mem_thresh": stat_thresh,
    }

    # ── load and renormalize TB metrics ──────────────────────────────────────
    metrics = load_metrics(exp_dir, task_type)
    metrics = normalize_metrics(metrics, task_type)
    steps_rule, vals_rule  = metrics.get("rule_acc",  (None, None))
    steps_mem,  vals_mem   = metrics.get("mem_ratio", (None, None))
    _,          vals_row   = metrics.get("row_acc",   (None, None))
    _,          vals_col   = metrics.get("col_acc",   (None, None))
    _,          vals_block = metrics.get("block_acc", (None, None))

    def _cross(steps, vals, threshold):
        if steps is None or vals is None:
            return np.nan
        return first_sustained_crossing(steps, vals, threshold, n_consec)

    # ── rule onset ──────────────────────────────────────────────────────────
    for t in ACC_THRESHOLDS:
        row[f"rule_onset_acc{int(t*100):02d}"] = _cross(steps_rule, vals_rule, t)

    # ── sub-structure onsets ─────────────────────────────────────────────────
    row["row_onset_90"]   = _cross(steps_rule, vals_row,   SUB_THRESHOLD)
    row["col_onset_90"]   = _cross(steps_rule, vals_col,   SUB_THRESHOLD)
    row["block_onset_90"] = _cross(steps_rule, vals_block, SUB_THRESHOLD)

    # ── mem onset ───────────────────────────────────────────────────────────
    for thresh, label in zip(MEM_THRESHOLDS, ["mem10", "mem20", "mem35", "mem50"]):
        row[f"mem_onset_{label}"] = _cross(steps_mem, vals_mem, thresh)

    if not np.isnan(stat_thresh):
        row["mem_onset_stat"] = _cross(steps_mem, vals_mem, stat_thresh)
    else:
        row["mem_onset_stat"] = np.nan

    # ── novel_valid onset ────────────────────────────────────────────────────
    if steps_rule is not None and vals_rule is not None and \
       steps_mem  is not None and vals_mem  is not None:

        # step-based alignment (safe across interrupted/resumed runs)
        if np.array_equal(steps_rule, steps_mem):
            nov_steps = steps_rule
            novel     = vals_rule - vals_mem
        else:
            mem_dict  = dict(zip(steps_mem, vals_mem))
            common    = np.array([s for s in steps_rule if s in mem_dict])
            idx_rule  = np.isin(steps_rule, common)
            nov_steps = common
            novel     = vals_rule[idx_rule] - np.array([mem_dict[s] for s in common])

        # sanity check — should be in [0, 1] by definition
        oob = int(np.sum((novel < -1e-3) | (novel > 1 + 1e-3)))
        if oob > 0:
            print(f"  WARNING {exp_name}: novel_valid out of [0,1] at {oob} steps "
                  f"(min={float(novel.min()):.4f}, max={float(novel.max()):.4f})")

        for thresh, label in zip(NOV_THRESHOLDS, ["01", "05", "09"]):
            row[f"novel_onset_{label}"] = first_sustained_crossing(
                nov_steps, novel, thresh, n_consec)
    else:
        for label in ["01", "05", "09"]:
            row[f"novel_onset_{label}"] = np.nan

    return row


# ── Experiment discovery ───────────────────────────────────────────────────────
def scan_all_runs(saveroot, task_filter=None):
    """
    Scan saveroot for all experiment dirs (containing args.json + tensorboard/).
    Returns list of (exp_name, exp_dir, args_dict).
    """
    runs = []
    for name in sorted(os.listdir(saveroot)):
        exp_dir   = os.path.join(saveroot, name)
        args_path = os.path.join(exp_dir, "args.json")
        tb_dir    = os.path.join(exp_dir, "tensorboard")
        if not (os.path.isdir(exp_dir) and
                os.path.isfile(args_path) and
                os.path.isdir(tb_dir)):
            continue
        with open(args_path) as f:
            args = json.load(f)
        task_type = detect_task_type(name, args)
        if task_filter and task_type not in task_filter:
            continue
        runs.append((name, exp_dir, args))
    return runs


# ── Column ordering ────────────────────────────────────────────────────────────
PARAM_COLS = [
    "exp_name", "task_type", "model_arch", "model_size", "rep",
    "N", "n_size", "D", "K", "K_list", "G", "encoding",
    "n_layer", "n_embd", "n_head", "lr", "wd", "nsteps",
    "support_size", "stat_mem_frac", "stat_mem_thresh",
]
ONSET_COLS = (
    [f"rule_onset_acc{int(t*100):02d}" for t in ACC_THRESHOLDS]
    + ["row_onset_90", "col_onset_90", "block_onset_90"]
    + ["mem_onset_mem10", "mem_onset_mem20", "mem_onset_mem35",
       "mem_onset_mem50", "mem_onset_stat"]
    + ["novel_onset_01", "novel_onset_05", "novel_onset_09"]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saveroot", required=True)
    parser.add_argument("--outdir",   default="outputs")
    parser.add_argument("--outname",  default="generalized_onset_table")
    parser.add_argument("--n_consec", type=int, default=5)
    parser.add_argument("--tasks",    nargs="+", default=None,
                        help="Task types to include (default: all). "
                             "Options: parity exactK rowK rowVarK globalK "
                             "rowOnly latinSq sudoku")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Scanning {args.saveroot} ...")
    runs = scan_all_runs(args.saveroot, task_filter=args.tasks)
    print(f"  Found {len(runs)} experiments"
          + (f" (filtered to {args.tasks})" if args.tasks else ""))

    rows        = []
    task_counts = {}
    for i, (exp_name, exp_dir, exp_args) in enumerate(runs):
        task_type = detect_task_type(exp_name, exp_args)
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
        print(f"  [{i+1:3d}/{len(runs)}] [{task_type:10s}] {exp_name}", flush=True)
        try:
            row = extract_row(exp_name, exp_dir, exp_args, n_consec=args.n_consec)
        except Exception as e:
            print(f"    ERROR: {e}")
            row = {"exp_name": exp_name, "task_type": task_type}
        rows.append(row)

    df      = pd.DataFrame(rows)
    all_cols = [c for c in PARAM_COLS + ONSET_COLS if c in df.columns] + \
               [c for c in df.columns if c not in PARAM_COLS + ONSET_COLS]
    df      = df[all_cols]

    csv_path = os.path.join(args.outdir, f"{args.outname}.csv")
    pkl_path = os.path.join(args.outdir, f"{args.outname}.pkl")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"\nSaved {len(df)} rows → {csv_path}")
    print(f"                     → {pkl_path}")
    print(f"\nTask counts: {task_counts}")

    for col, label in [("rule_onset_acc90", "Rule onset acc90"),
                       ("novel_onset_05",   "Novel onset 0.5")]:
        if col not in df.columns:
            continue
        print(f"\n{label} reach rate by task_type:")
        for tt, sub in df.groupby("task_type"):
            reached = sub[col].notna().sum()
            print(f"  {tt:12s}: {reached}/{len(sub)}")

    return df


if __name__ == "__main__":
    main()
