"""
fit_lib.py
----------
Utilities for fitting and plotting power-law lines (linear on log-log axes).

Typical usage
-------------
    from core.fit_lib import fit_loglog, plot_loglog_fit

    slope, intercept, r2, fn = fit_loglog(x, y)
    print(f"y ~ {np.exp(intercept):.3g} * x^{slope:.2f}  (R²={r2:.3f})")

    # overlay fit line on an existing axes
    plot_loglog_fit(ax, x, y, label="fit", color="black")
"""

import numpy as np


def fit_loglog(x, y, mask=None):
    """Fit y = a * x^b via OLS on log-log data.

    Parameters
    ----------
    x, y  : array-like  — positive values (NaN/inf rows are dropped automatically)
    mask  : bool array  — optional additional mask (True = include)

    Returns
    -------
    slope     : float  — power-law exponent b
    intercept : float  — log(a), so a = exp(intercept)
    r2        : float  — R² of the log-log fit
    n         : int    — number of valid points used in the fit
    fn        : callable(x) → y  — the fitted power-law function
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    n = int(valid.sum())
    if n < 2:
        nan = float("nan")
        return nan, nan, nan, n, lambda xv: np.full(np.asarray(xv).shape, nan)

    lx = np.log(x[valid])
    ly = np.log(y[valid])

    slope, intercept = np.polyfit(lx, ly, 1)

    # R²
    ly_hat  = slope * lx + intercept
    ss_res  = np.sum((ly - ly_hat) ** 2)
    ss_tot  = np.sum((ly - ly.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fn = lambda xv: np.exp(intercept) * np.asarray(xv) ** slope

    return slope, intercept, r2, n, fn


def fit_loglog_segment(x, y, x_min=None, x_max=None):
    """Like fit_loglog but restricted to x in [x_min, x_max].
    Returns slope, intercept, r2, n, fn."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.ones(len(x), dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    return fit_loglog(x, y, mask=mask)


def plot_loglog_fit(ax, x, y, x_min=None, x_max=None, n_pts=200,
                    label=None, show_slope=True, show_r2=False,
                    color="black", lw=1.5, ls="--", **kwargs):
    """Fit and overlay a power-law line on ax.

    Parameters
    ----------
    ax            : matplotlib Axes
    x, y          : data (positive values; NaN dropped)
    x_min, x_max  : restrict fit to this x range (None = use all valid data)
    n_pts         : number of points for the plotted line
    label         : base label string (slope / R² appended if requested)
    show_slope    : append  x^{slope}  to label
    show_r2       : append  R²=...     to label
    color,lw,ls   : line style passed to ax.plot

    Returns
    -------
    slope, intercept, r2, n : fit results (n = number of valid points used)
    """
    slope, intercept, r2, n, fn = fit_loglog_segment(x, y, x_min=x_min, x_max=x_max)

    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    x_lo = x_min if x_min is not None else x[valid].min()
    x_hi = x_max if x_max is not None else x[valid].max()
    xs   = np.geomspace(x_lo, x_hi, n_pts)

    lbl = label or ""
    if show_slope:
        lbl += f" $\\propto x^{{{slope:.2f}}}$"
    if show_r2:
        lbl += f" $R^2={r2:.2f}$"
    if n is not None:
        lbl += f" (n={n})"

    ax.plot(xs, fn(xs), color=color, lw=lw, ls=ls,
            label=lbl.strip() if lbl.strip() else None, **kwargs)

    return slope, intercept, r2, n
