"""
extract_onset_table.py
----------------------
Batch-extract rule-learning and memorization onset times for all parity runs
and save results as CSV + pickle.

Onset thresholds tried:
  Rule accuracy : 0.85, 0.90, 0.95
  Mem ratio     : 0.10 (fixed reference)
                  0.20, 0.35
                  0.50 (heavy memorization)
                  0.10 + stat_mem_frac  (data-adaptive baseline)
  Novel valid   : novel_valid = valid_acc - mem_ratio
                  thresholds 0.1, 0.5, 0.9

Output columns (in addition to run parameters):
  rule_onset_acc{85,90,95}   — step of first sustained rule-learning onset
  mem_onset_mem{10,20,35,50} — step when mem_ratio > threshold sustained
  mem_onset_stat             — step when mem_ratio > (0.10 + stat_mem_frac) sustained
  innov_acc90_{label}        — mem_onset_{label} - rule_onset_acc90
  novel_onset_{01,05,09}     — step when (valid_acc - mem_ratio) > {0.1,0.5,0.9} sustained

NaN means the threshold was never sustainedly crossed within the run budget.

Usage:
    python scripts/extract_onset_table.py --saveroot /path/to/DiffusionParityLearning
    python scripts/extract_onset_table.py --saveroot ... --outdir outputs/ --n_consec 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── allow running from repo root ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.run_registry import scan_parity_runs
from core.onset_lib    import load_eval_timeseries, first_sustained_crossing

# ── threshold grids ────────────────────────────────────────────────────────────
ACC_THRESHOLDS   = [0.85, 0.90, 0.95]
NOVEL_THRESHOLDS = [0.1, 0.5, 0.9]   # thresholds for novel_valid = valid_acc - mem_ratio


def extract_row(run, saveroot, n_consec=5):
    """Compute all onset columns for one run dict.

    Loads the eval timeseries once, then applies first_sustained_crossing
    for all thresholds (rule, mem, novel_valid).

    Returns the run dict extended with onset + innovation columns.
    """
    exp_name      = run["exp_name"]
    stat_mem_frac = run["stat_mem_frac"]
    stat_thresh   = 0.10 + stat_mem_frac   # adaptive mem threshold

    row = dict(run)  # start with all parameter fields
    row["stat_mem_thresh"] = stat_thresh

    # ── load timeseries once ───────────────────────────────────────────────
    ts = load_eval_timeseries(exp_name, saveroot)
    if ts is None:
        for acc_t in ACC_THRESHOLDS:
            row[f"rule_onset_acc{int(acc_t*100):02d}"] = np.nan
        for label in ["mem10", "mem20", "mem35", "mem50", "stat"]:
            row[f"mem_onset_{label}"] = np.nan
            row[f"innov_acc90_{label}"] = np.nan
        for thresh in NOVEL_THRESHOLDS:
            row[f"novel_onset_{int(thresh*10):02d}"] = np.nan
        return row

    steps     = ts["eval_steps"]
    valid_acc = ts["valid_acc"]
    mem_ratio = ts["mem_ratio"]

    # ── novel_valid = valid_acc - mem_ratio (should be in [0,1] by definition) ─
    novel_valid  = valid_acc - mem_ratio
    out_of_range = np.sum((novel_valid < -1e-6) | (novel_valid > 1 + 1e-6))
    if out_of_range > 0:
        print(f"  WARNING {exp_name}: novel_valid out of [0,1] at {out_of_range} steps "
              f"(min={novel_valid.min():.4f}, max={novel_valid.max():.4f})")

    # ── rule onset at each acc threshold ──────────────────────────────────
    rule_onsets = {}
    for acc_t in ACC_THRESHOLDS:
        r = first_sustained_crossing(steps, valid_acc, acc_t, n_consec)
        row[f"rule_onset_acc{int(acc_t*100):02d}"] = r
        rule_onsets[acc_t] = r

    # ── mem onset at each mem threshold ───────────────────────────────────
    mem_configs = [
        ("mem10", 0.10),
        ("mem20", 0.20),
        ("mem35", 0.35),
        ("mem50", 0.50),
        ("stat",  stat_thresh),
    ]
    mem_onsets = {}
    for label, mem_t in mem_configs:
        m = first_sustained_crossing(steps, mem_ratio, mem_t, n_consec)
        row[f"mem_onset_{label}"] = m
        mem_onsets[label] = m

    # ── innovation windows (mem_onset - rule_onset) at acc90 ──────────────
    rule_90 = rule_onsets[0.90]
    for label, _ in mem_configs:
        m = mem_onsets[label]
        innov = np.nan if (np.isnan(rule_90) or np.isnan(m)) else m - rule_90
        row[f"innov_acc90_{label}"] = innov

    # ── novel_valid onset at each threshold ───────────────────────────────
    for thresh in NOVEL_THRESHOLDS:
        col = f"novel_onset_{int(thresh*10):02d}"
        row[col] = first_sustained_crossing(steps, novel_valid, thresh, n_consec)

    return row


def main():
    parser = argparse.ArgumentParser(description="Extract onset table for all parity runs")
    parser.add_argument("--saveroot", required=True,
                        help="Root dir containing experiment folders")
    parser.add_argument("--outdir", default="outputs",
                        help="Directory for output files (default: outputs/)")
    parser.add_argument("--n_consec", type=int, default=5,
                        help="Consecutive eval points required for onset (default: 5)")
    parser.add_argument("--arches", nargs="+", default=["DiT", "GPT"],
                        help="Architectures to include (default: DiT GPT)")
    parser.add_argument("--sizes", nargs="+", default=["nano", "mini", "B"],
                        help="Model sizes to include (default: nano mini B)")
    parser.add_argument("--outname", default="parity_onset_table_v2",
                        help="Output filename stem (default: parity_onset_table_v2)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── discover runs ──────────────────────────────────────────────────────
    print(f"Scanning {args.saveroot} ...")
    runs = scan_parity_runs(args.saveroot,
                            arches=tuple(args.arches),
                            sizes=tuple(args.sizes))
    print(f"  Found {len(runs)} runs")

    # ── extract onsets ─────────────────────────────────────────────────────
    rows = []
    for i, run in enumerate(runs):
        exp_name = run["exp_name"]
        print(f"  [{i+1:3d}/{len(runs)}] {exp_name}", flush=True)
        try:
            row = extract_row(run, args.saveroot, n_consec=args.n_consec)
        except Exception as e:
            print(f"    ERROR: {e}")
            row = dict(run)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── column order ───────────────────────────────────────────────────────
    param_cols = [
        "exp_name", "model_arch", "model_size",
        "G", "N", "D", "n_layer", "n_embd", "n_head",
        "lr", "wd", "nsteps", "rep",
        "support_size", "stat_mem_frac",
    ]
    onset_cols = (
        [f"rule_onset_acc{int(t*100):02d}" for t in ACC_THRESHOLDS]
        + ["mem_onset_mem10", "mem_onset_mem20", "mem_onset_mem35",
           "mem_onset_mem50", "mem_onset_stat", "stat_mem_thresh"]
        + ["innov_acc90_mem10", "innov_acc90_mem20", "innov_acc90_mem35",
           "innov_acc90_mem50", "innov_acc90_stat"]
        + [f"novel_onset_{int(t*10):02d}" for t in NOVEL_THRESHOLDS]
    )
    # keep any unexpected extra columns at the end
    extra = [c for c in df.columns if c not in param_cols + onset_cols]
    df = df[param_cols + onset_cols + extra]

    # ── save (new file, don't overwrite old table) ─────────────────────────
    csv_path = os.path.join(args.outdir, f"{args.outname}.csv")
    pkl_path = os.path.join(args.outdir, f"{args.outname}.pkl")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"\nSaved {len(df)} rows → {csv_path}")
    print(f"                     → {pkl_path}")

    # ── quick summary ──────────────────────────────────────────────────────
    print("\nRule onset (acc90) reach rate by arch/size:")
    grp = df.groupby(["model_arch", "model_size"])
    for (arch, size), sub in grp:
        reached = sub["rule_onset_acc90"].notna().sum()
        print(f"  {arch}-{size}: {reached}/{len(sub)} runs reached rule onset")

    print("\nNovel valid onset (0.5) reach rate by arch/size:")
    for (arch, size), sub in grp:
        reached = sub["novel_onset_05"].notna().sum()
        print(f"  {arch}-{size}: {reached}/{len(sub)} runs reached novel onset 0.5")

    return df


if __name__ == "__main__":
    main()
