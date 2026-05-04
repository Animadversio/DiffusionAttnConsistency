"""
run_registry.py
---------------
Scan a SAVEROOT directory for parity experiment folders, parse their
args.json, and return a list of parameter dicts ready for onset analysis.

Naming convention expected:
  {arch}_{size}_parity_N{N}_D{D}_G{G}_even[_{suffix}...]

  arch   : DiT | GPT
  size   : B | mini | nano
  N      : training sample count
  D      : sequence length (e.g. 36)
  G      : group size (bits per parity group)
  suffix : optional tags like _rep2, _lr1e3, _wd0, _lr3e4, etc.

Derived fields (added automatically):
  support_size  = (2^(G-1))^(D//G)   — number of valid rule-consistent sequences
  stat_mem_frac = N / support_size    — fraction of support covered by training data

Typical usage:
    from core.run_registry import scan_parity_runs
    runs = scan_parity_runs(SAVEROOT)          # list of dicts
    import pandas as pd
    df = pd.DataFrame(runs)
"""

import os
import re
import json
import math

# ── regex for directory name ───────────────────────────────────────────────────
_PAT = re.compile(
    r'^(?P<arch>DiT|GPT)_(?P<size>B|mini|nano)_parity'
    r'_N(?P<N>\d+)_D(?P<D>\d+)_G(?P<G>\d+)_even(?P<suffix>.*)$'
)

_LR_PAT  = re.compile(r'_lr([\deE.+-]+)')
_WD_PAT  = re.compile(r'_wd([\deE.+-]+)')
_REP_PAT = re.compile(r'_rep(\d+)')


def _parse_sci(s):
    """Parse shorthand scientific notation like '1e3', '3e4', '1e-2'."""
    try:
        return float(s)
    except ValueError:
        return None


def _parse_suffix(suffix):
    """Extract rep, lr_tag, wd_tag from the trailing part of a run name."""
    rep = 1
    m = _REP_PAT.search(suffix)
    if m:
        rep = int(m.group(1))
    lr_tag = wd_tag = None
    m = _LR_PAT.search(suffix)
    if m:
        lr_tag = _parse_sci(m.group(1))
    m = _WD_PAT.search(suffix)
    if m:
        wd_tag = _parse_sci(m.group(1))
    return rep, lr_tag, wd_tag


def _support_size(G, D):
    """(2^(G-1))^(D//G) — number of valid rule-consistent sequences."""
    n_groups = D // G
    return (2 ** (G - 1)) ** n_groups


def _parse_one(exp_name, saveroot):
    """Return a parameter dict for one experiment dir, or None if unparseable."""
    m = _PAT.match(exp_name)
    if m is None:
        return None

    arch   = m.group("arch")
    size   = m.group("size")
    N      = int(m.group("N"))
    D      = int(m.group("D"))
    G      = int(m.group("G"))
    suffix = m.group("suffix")
    rep, lr_tag, wd_tag = _parse_suffix(suffix)

    # ── read args.json ─────────────────────────────────────────────────────
    args_path = os.path.join(saveroot, exp_name, "args.json")
    if not os.path.exists(args_path):
        return None
    with open(args_path) as f:
        args = json.load(f)

    # ── architecture-specific field names ──────────────────────────────────
    if arch == "GPT":
        n_layer = args.get("n_layer",   None)
        n_embd  = args.get("n_embd",    None)
        n_head  = args.get("n_head",    None)
    else:  # DiT
        n_layer = args.get("depth",       None)
        n_embd  = args.get("hidden_size", None)
        n_head  = args.get("num_heads",   None)

    lr     = args.get("lr",           None)
    wd     = args.get("weight_decay", args.get("wd", 0.0))
    nsteps = args.get("nsteps",       None)

    # prefer parsed tag values (from dir name) when args.json is missing them
    if lr is None:
        lr = lr_tag
    if wd is None:
        wd = wd_tag if wd_tag is not None else 0.0

    # ── derived statistical fields ─────────────────────────────────────────
    sup  = _support_size(G, D)
    stat = N / sup if sup > 0 else float("inf")

    return dict(
        exp_name     = exp_name,
        model_arch   = arch,
        model_size   = size,
        G            = G,
        N            = N,
        D            = D,
        n_layer      = n_layer,
        n_embd       = n_embd,
        n_head       = n_head,
        lr           = lr,
        wd           = wd,
        nsteps       = nsteps,
        rep          = rep,
        support_size = sup,
        stat_mem_frac= stat,
    )


def scan_parity_runs(saveroot, arches=("DiT", "GPT"), sizes=("nano", "mini", "B")):
    """Scan saveroot and return a list of parameter dicts for all parity runs.

    Parameters
    ----------
    saveroot : str   — root directory containing experiment folders
    arches   : tuple — which model architectures to include
    sizes    : tuple — which model sizes to include

    Returns
    -------
    list of dicts, one per valid experiment folder, sorted by
    (model_arch, model_size, G, N, lr, wd, rep).
    """
    runs = []
    for name in sorted(os.listdir(saveroot)):
        entry = _parse_one(name, saveroot)
        if entry is None:
            continue
        if entry["model_arch"] not in arches:
            continue
        if entry["model_size"] not in sizes:
            continue
        runs.append(entry)

    runs.sort(key=lambda r: (
        r["model_arch"], r["model_size"],
        r["G"], r["N"],
        r["lr"] or 0, r["wd"] or 0,
        r["rep"],
    ))
    return runs
