"""
src/stress.py
=============
Three-layer stress-testing framework for the weighting-scheme comparison.

Layer 1 -- Historical regime windows: carve the four crisis regimes out of the
           realized return series and compare drawdown, recovery, and behavior.
Layer 2 -- Circular block bootstrap: resample return paths to build full
           distributions of drawdown and tail loss (VaR/CVaR), not just the one
           path history happened to take.
Layer 3 -- Mega-cap unwind scenario: apply a hypothetical shock (leaders -40%,
           the rest -10%) to each scheme's actual weight vector, answering the
           thesis's motivating question directly.

All functions are pure post-processing: they consume monthly return Series and
weight vectors already produced by the pipeline. No WRDS access required.

Usage: see run_stress.py at the repository root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Layer 1: historical regime windows
# ----------------------------------------------------------------------------

# Regime boundaries are pre-registered here, not fitted to results.
REGIMES = {
    "dot_com_unwind": ("2000-03-01", "2002-10-31"),
    "gfc":            ("2007-10-01", "2009-03-31"),
    "covid":          ("2020-02-01", "2020-04-30"),
    "rate_shock":     ("2022-01-01", "2022-10-31"),
}


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown of a cumulative growth path built from monthly returns."""
    curve = (1.0 + returns).cumprod()
    return curve / curve.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    return float(drawdown_series(returns).min())


def recovery_months(returns: pd.Series) -> float:
    """Months from the deepest trough back to the prior peak within the series.

    Returns np.inf if the series ends before recovery -- itself informative.
    """
    dd = drawdown_series(returns)
    if len(dd) == 0 or dd.min() >= 0:
        return 0.0
    trough = dd.idxmin()
    after = dd.loc[trough:]
    recovered = after[after >= -1e-12]
    if recovered.empty:
        return float("inf")
    return float(len(after.loc[:recovered.index[0]]) - 1)


def regime_table(returns_wide: pd.DataFrame,
                 regimes: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Per-scheme, per-regime cumulative return and max drawdown.

    Parameters
    ----------
    returns_wide : DataFrame indexed by month-end date, one column per scheme.
    """
    regimes = regimes or REGIMES
    rows = []
    for name, (a, b) in regimes.items():
        window = returns_wide.loc[a:b]
        for scheme in returns_wide.columns:
            r = window[scheme].dropna()
            rows.append({
                "regime": name,
                "scheme": scheme,
                "months": int(len(r)),
                "cum_return": float((1 + r).prod() - 1) if len(r) else np.nan,
                "max_drawdown": max_drawdown(r),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Layer 2: circular block bootstrap
# ----------------------------------------------------------------------------

def block_bootstrap_paths(returns: pd.Series, n_paths: int = 2000,
                          block_len: int = 6, horizon: int | None = None,
                          seed: int = 7) -> np.ndarray:
    """Resample return paths with a circular block bootstrap.

    Blocks of consecutive months preserve short-horizon autocorrelation and
    volatility clustering that iid resampling destroys.

    Returns array of shape (n_paths, horizon).
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n == 0:
        raise ValueError("empty return series")
    horizon = horizon or n
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block_len))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    # circular indexing so blocks can wrap
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    paths = r[idx].reshape(n_paths, n_blocks * block_len)[:, :horizon]
    return paths


def bootstrap_risk_profile(returns: pd.Series, n_paths: int = 2000,
                           block_len: int = 6, seed: int = 7) -> dict:
    """Distributional risk statistics from bootstrapped paths.

    Reports the distribution of path-level max drawdown, plus monthly VaR/CVaR
    at 95% from the pooled resampled months.
    """
    paths = block_bootstrap_paths(returns, n_paths=n_paths,
                                  block_len=block_len, seed=seed)
    curves = np.cumprod(1.0 + paths, axis=1)
    peaks = np.maximum.accumulate(curves, axis=1)
    dds = (curves / peaks - 1.0).min(axis=1)          # one max-DD per path

    pooled = paths.ravel()
    var95 = float(np.quantile(pooled, 0.05))
    cvar95 = float(pooled[pooled <= var95].mean())

    return {
        "n_paths": int(n_paths),
        "block_len_months": int(block_len),
        "maxdd_median": float(np.median(dds)),
        "maxdd_p05": float(np.quantile(dds, 0.05)),    # bad tail of drawdowns
        "maxdd_p95": float(np.quantile(dds, 0.95)),
        "monthly_var95": var95,
        "monthly_cvar95": cvar95,
    }


# ----------------------------------------------------------------------------
# Layer 3: mega-cap unwind scenario
# ----------------------------------------------------------------------------

def unwind_scenario(weights: pd.Series, mega_ids: pd.Index | list,
                    shock_mega: float = -0.40, shock_rest: float = -0.10) -> dict:
    """One-period shock applied to a scheme's actual weight vector.

    Parameters
    ----------
    weights : Series of portfolio weights indexed by security id (sums to ~1).
    mega_ids : identifiers of the mega-cap leaders (e.g., the benchmark's
               current top 10 by weight -- the SAME set for every scheme, so
               the scenario tests exposure to one common event).
    """
    w = weights / weights.sum()
    mega_mask = w.index.isin(pd.Index(mega_ids))
    mega_weight = float(w[mega_mask].sum())
    port_return = mega_weight * shock_mega + (1.0 - mega_weight) * shock_rest
    return {
        "mega_weight": mega_weight,
        "shock_mega": shock_mega,
        "shock_rest": shock_rest,
        "portfolio_return": float(port_return),
        # loss avoided vs. a hypothetical 100% mega-cap book, for context
        "vs_all_mega": float(port_return - shock_mega),
    }


def unwind_table(weights_by_scheme: dict[str, pd.Series],
                 benchmark_scheme: str = "cap_weight", top_k: int = 10,
                 shock_mega: float = -0.40, shock_rest: float = -0.10) -> pd.DataFrame:
    """Run the unwind scenario for every scheme against a COMMON mega-cap set.

    The mega-cap set is defined from the benchmark's top-k holdings so that
    every scheme is shocked on the same event; differences in outcome are then
    purely differences in exposure.
    """
    bench = weights_by_scheme[benchmark_scheme]
    mega_ids = bench.sort_values(ascending=False).head(top_k).index
    rows = []
    for scheme, w in weights_by_scheme.items():
        res = unwind_scenario(w, mega_ids, shock_mega, shock_rest)
        res["scheme"] = scheme
        rows.append(res)
    out = pd.DataFrame(rows).set_index("scheme")
    out["protection_vs_benchmark_bps"] = (
        (out["portfolio_return"] - out.loc[benchmark_scheme, "portfolio_return"]) * 1e4
    ).round(0)
    return out
