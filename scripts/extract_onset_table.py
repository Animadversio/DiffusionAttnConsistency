"""
extract_onset_table.py
----------------------
Batch-extract rule-learning and memorization onset times for all parity runs
and save results as CSV + pickle.

Onset thresholds tried:
  Rule accuracy : 0.85, 0.90, 0.95
  Mem ratio     : 0.10 (fixed reference)
                  0.50 (heavy memorization)
                  0.10 + stat_mem_frac  (data-adaptive baseline)

Output columns (in addition to run parameters):
  rule_onset_acc{85,90,95}   — step of first sustained rule-learning onset
  mem_onset_mem10            — step when mem_ratio > 0.10 sustained
  mem_onset_mem50            — step when mem_ratio > 0.50 sustained
  mem_onset_stat             — step when mem_ratio > (0.10 + stat_mem_frac) sustained
  innov_acc90_mem10          — mem_onset_mem10  - rule_onset_acc90
  innov_acc90_mem50          — mem_onset_mem50  - rule_onset_acc90
  innov_acc90_stat           — mem_onset_stat   - rule_onset_acc90

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
from core.onset_lib    import get_onsets

# ── threshold grids ────────────────────────────────────────────────────────────
ACC_THRESHOLDS = [0.85, 0.90, 0.95]


def extract_row(run, saveroot, n_consec=5):
    """Compute all onset columns for one run dict.

    Returns the run dict extended with onset + innovation columns.
    """
    exp_name      = run["exp_name"]
    stat_mem_frac = run["stat_mem_frac"]
    stat_thresh   = 0.10 + stat_mem_frac   # adaptive mem threshold

    row = dict(run)  # start with all parameter fields

    # ── rule onset at each acc threshold ──────────────────────────────────
    rule_onsets = {}
    for acc_t in ACC_THRESHOLDS:
        col = f"rule_onset_acc{int(acc_t*100):02d}"
        r, _ = get_onsets(exp_name, saveroot,
                          acc_thresh=acc_t, mem_thresh=0.5,
                          n_consec=n_consec)
        row[col] = r
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
        col = f"mem_onset_{label}"
        _, m = get_onsets(exp_name, saveroot,
                          acc_thresh=0.9, mem_thresh=mem_t,
                          n_consec=n_consec)
        row[col] = m
        mem_onsets[label] = m

    # ── innovation windows (mem - rule) for primary acc threshold 0.90 ────
    rule_90 = rule_onsets[0.90]
    for label, _ in mem_configs:
        m = mem_onsets[label]
        if np.isnan(rule_90) or np.isnan(m):
            innov = np.nan
        else:
            innov = m - rule_90
        row[f"innov_acc90_{label}"] = innov

    # store the adaptive threshold used so it's auditable
    row["stat_mem_thresh"] = stat_thresh

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
    )
    # keep any unexpected extra columns at the end
    extra = [c for c in df.columns if c not in param_cols + onset_cols]
    df = df[param_cols + onset_cols + extra]

    # ── save ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.outdir, "parity_onset_table.csv")
    pkl_path = os.path.join(args.outdir, "parity_onset_table.pkl")
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

    return df


if __name__ == "__main__":
    main()
