#!/usr/bin/env python3
"""
export_mem_scaling_table.py
---------------------------
Fit power-law scaling of memorization onset time vs N for each model/G,
and export results as CSV and LaTeX table.

Usage:
    python scripts/export_mem_scaling_table.py \
        --input  outputs/parity_onset_table.csv \
        --outdir outputs/

Output:
    outputs/mem_scaling_table.csv
    outputs/mem_scaling_table.tex
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from core.fit_lib import fit_loglog


MODELS = [
    ("DiT", "mini", 0.00),
    ("GPT", "mini", 0.01),
    ("GPT", "B",    0.01),
]
G_VALS = [2, 3, 4, 6, 9, 12, 18, 36]


def get_scaling(sub):
    slope, intercept, r2, n, fn = fit_loglog(sub["N"], sub["mem_onset_stat"])
    return dict(
        slope=slope,
        coeff=np.exp(intercept) if np.isfinite(intercept) else np.nan,
        r2=r2,
        n_pts=n,
    )


def build_table(df):
    rows = []
    for arch, size, wd in MODELS:
        label = f"{arch}-{size}"
        sub_allG = df.query(
            f"model_arch == '{arch}' and model_size == '{size}'"
            f" and lr == 0.0001 and wd == {wd}"
        ).copy()

        # all-G pooled
        r = get_scaling(sub_allG)
        rows.append({"model": label, "G": "all", **r})

        # per G
        for G in G_VALS:
            sub = sub_allG.query("G == @G").copy()
            r = get_scaling(sub)
            rows.append({"model": label, "G": str(G), **r})

    return pd.DataFrame(rows)[["model", "G", "coeff", "slope", "r2", "n_pts"]]


def to_latex(results):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Power-law scaling of memorization onset $\tau_\mathrm{mem}$ with training set size $N$: "
                 r"$\tau_\mathrm{mem} \approx c \cdot N^\alpha$.  "
                 r"Fit using adaptive threshold $\mathrm{mem\_thresh} = 0.1 + N/\mathrm{support\_size}$; "
                 r"$R^2$ and $n$ are the fit quality and number of data points.}")
    lines.append(r"\label{tab:mem_scaling}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & $G$ & $c$ & $\alpha$ & $R^2$ & $n$ \\")
    lines.append(r"\midrule")

    prev_model = None
    for _, row in results.iterrows():
        model = row["model"]
        if model != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = model

        c      = f"{row['coeff']:.1f}"  if np.isfinite(row['coeff'])  else "---"
        alpha  = f"{row['slope']:.2f}"  if np.isfinite(row['slope'])  else "---"
        r2     = f"{row['r2']:.2f}"     if np.isfinite(row['r2'])     else "---"
        n      = str(int(row['n_pts'])) if np.isfinite(row['n_pts'])  else "---"
        G_str  = r"\textbf{all}" if row["G"] == "all" else f"${row['G']}$"
        m_str  = model if row["G"] == "all" else ""

        lines.append(f"{m_str} & {G_str} & {c} & {alpha} & {r2} & {n} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="outputs/parity_onset_table.csv")
    parser.add_argument("--outdir", default="outputs/")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = build_table(df)

    outdir = Path(args.outdir)
    csv_path = outdir / "mem_scaling_table.csv"
    tex_path = outdir / "mem_scaling_table.tex"

    results.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    tex = to_latex(results)
    tex_path.write_text(tex)
    print(f"Saved LaTeX: {tex_path}")

    print("\n" + results.to_string(index=False))
    print("\n--- LaTeX ---\n")
    print(tex)


if __name__ == "__main__":
    main()
