"""
src/inference.py
================
Statistical inference for performance differences between weighting schemes.

The results table asserts, e.g., that equal weight's information ratio "is not
distinguishable from zero over 25 years." This module supplies the tests that
back such statements:

  * sharpe_diff_hac        -- Ledoit & Wolf (2008)-style test for the
                              difference of two Sharpe ratios, using a
                              delta-method variance with a HAC (Newey-West)
                              kernel to respect autocorrelation.
  * sharpe_diff_bootstrap  -- circular block bootstrap of the Sharpe
                              difference, a distribution-free cross-check.
  * ir_test                -- t-test on mean active return (information ratio
                              significance) with HAC standard errors.

References: Ledoit, O. & Wolf, M. (2008), "Robust performance hypothesis
testing with the Sharpe ratio," Journal of Empirical Finance 15(5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def sharpe(returns: pd.Series, ann: int = 12, rf: float | pd.Series = 0.0) -> float:
    ex = returns - rf
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan


def _nw_lrv(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West long-run variance of a (mean-adjusted) series."""
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # standard rule
    x = x - x.mean()
    gamma0 = float(x @ x) / n
    lrv = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = float(x[k:] @ x[:-k]) / n
        lrv += 2.0 * w * gk
    return lrv


# ----------------------------------------------------------------------------
# Sharpe-ratio difference: HAC delta-method test (Ledoit-Wolf style)
# ----------------------------------------------------------------------------

def sharpe_diff_hac(r_a: pd.Series, r_b: pd.Series, ann: int = 12,
                    lags: int | None = None) -> dict:
    """Test H0: Sharpe(a) = Sharpe(b) for two aligned monthly return series.

    Delta-method on the moment vector (mu_a, mu_b, m2_a, m2_b) with a
    Newey-West covariance, following the construction in Ledoit & Wolf (2008).
    Two-sided p-value from the normal limit.
    """
    both = pd.concat([r_a, r_b], axis=1, join="inner").dropna()
    a, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    n = len(a)
    if n < 24:
        raise ValueError("need at least 24 overlapping months")

    mu_a, mu_b = a.mean(), b.mean()
    m2_a, m2_b = (a ** 2).mean(), (b ** 2).mean()
    sd_a = np.sqrt(m2_a - mu_a ** 2)
    sd_b = np.sqrt(m2_b - mu_b ** 2)
    sr_a, sr_b = mu_a / sd_a, mu_b / sd_b            # per-period Sharpe
    diff = sr_a - sr_b

    # gradient of f(mu, m2) = mu / sqrt(m2 - mu^2) for each series
    def grad(mu, m2):
        var = m2 - mu ** 2
        return np.array([m2 / var ** 1.5, -0.5 * mu / var ** 1.5])

    ga, gb = grad(mu_a, m2_a), grad(mu_b, m2_b)
    # stacked influence series for (mu_a, m2_a, mu_b, m2_b)
    z = np.column_stack([a, a ** 2, b, b ** 2])
    z = z - z.mean(axis=0)
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    # HAC covariance of the moment vector
    S = (z.T @ z) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        G = (z[k:].T @ z[:-k]) / n
        S += w * (G + G.T)
    g = np.concatenate([ga, -gb])                    # d(diff)/d(moments)
    var_diff = float(g @ S @ g) / n
    se = np.sqrt(max(var_diff, 1e-18))
    z_stat = diff / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return {
        "n_months": int(n),
        "sharpe_a_ann": float(sr_a * np.sqrt(ann)),
        "sharpe_b_ann": float(sr_b * np.sqrt(ann)),
        "diff_ann": float(diff * np.sqrt(ann)),
        "z_stat": float(z_stat),
        "p_value": float(p),
        "hac_lags": int(lags),
    }


# ----------------------------------------------------------------------------
# Sharpe-ratio difference: circular block bootstrap
# ----------------------------------------------------------------------------

def sharpe_diff_bootstrap(r_a: pd.Series, r_b: pd.Series, ann: int = 12,
                          n_boot: int = 5000, block_len: int = 6,
                          seed: int = 7) -> dict:
    """Bootstrap the Sharpe difference by resampling PAIRED month blocks.

    Pairing preserves the cross-correlation between the two schemes, which is
    what makes differences testable at all when both track the same market.
    Two-sided p-value: the fraction of centered bootstrap differences at least
    as extreme as the observed one.
    """
    both = pd.concat([r_a, r_b], axis=1, join="inner").dropna()
    a, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)

    def sr(x):
        return x.mean() / x.std(ddof=1)

    obs = sr(a) - sr(b)
    n_blocks = int(np.ceil(n / block_len))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
        idx = idx[:n]
        diffs[i] = sr(a[idx]) - sr(b[idx])
    centered = diffs - diffs.mean()
    p = float((np.abs(centered) >= abs(obs)).mean())

    return {
        "n_months": int(n),
        "diff_ann": float(obs * np.sqrt(ann)),
        "p_value": p,
        "n_boot": int(n_boot),
        "block_len_months": int(block_len),
        "ci95_ann": (float(np.quantile(diffs, 0.025) * np.sqrt(ann)),
                     float(np.quantile(diffs, 0.975) * np.sqrt(ann))),
    }


# ----------------------------------------------------------------------------
# Information-ratio significance
# ----------------------------------------------------------------------------

def ir_test(r_scheme: pd.Series, r_benchmark: pd.Series, ann: int = 12,
            lags: int | None = None) -> dict:
    """HAC t-test of mean active return: is the information ratio nonzero?"""
    both = pd.concat([r_scheme, r_benchmark], axis=1, join="inner").dropna()
    active = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
    n = len(active)
    lrv = _nw_lrv(active, lags)
    se = np.sqrt(lrv / n)
    t = active.mean() / se if se > 0 else np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    te = active.std(ddof=1)
    return {
        "n_months": int(n),
        "ir_ann": float(active.mean() / te * np.sqrt(ann)) if te > 0 else np.nan,
        "active_return_ann_bps": float(active.mean() * ann * 1e4),
        "t_stat": float(t),
        "p_value": float(p),
    }


def inference_table(returns_wide: pd.DataFrame,
                    benchmark: str = "cap_weight") -> pd.DataFrame:
    """Every scheme vs. the benchmark: Sharpe-difference (HAC + bootstrap) and IR tests."""
    rows = []
    for scheme in returns_wide.columns:
        if scheme == benchmark:
            continue
        ra, rb = returns_wide[scheme], returns_wide[benchmark]
        hac = sharpe_diff_hac(ra, rb)
        boot = sharpe_diff_bootstrap(ra, rb)
        ir = ir_test(ra, rb)
        rows.append({
            "scheme": scheme,
            "sharpe_diff_ann": round(hac["diff_ann"], 3),
            "p_hac": round(hac["p_value"], 3),
            "p_bootstrap": round(boot["p_value"], 3),
            "ir_ann": round(ir["ir_ann"], 3),
            "p_ir": round(ir["p_value"], 3),
        })
    return pd.DataFrame(rows).set_index("scheme")
