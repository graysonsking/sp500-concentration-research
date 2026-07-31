"""Concentration and diversification measurement.

These are the measures the research question actually turns on. Return and
volatility statistics answer whether an alternative scheme performed well.
These answer whether it delivered diversification, which is a different
question and the one being asked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def herfindahl(weights: pd.Series) -> float:
    """Herfindahl-Hirschman index of portfolio weights.

    Ranges from 1/N (perfectly equal) to 1.0 (single holding).
    """
    w = weights.dropna()
    return float((w ** 2).sum())


def effective_n(weights: pd.Series) -> float:
    """Effective number of holdings. The inverse Herfindahl index.

    The headline number for this research. An index nominally holding 500
    names but with an effective N in the low double digits is not delivering
    the diversification a passive allocator assumes they are buying.
    """
    hhi = herfindahl(weights)
    return 1.0 / hhi if hhi else np.nan


def top_n_weight(weights: pd.Series, n: int = 10) -> float:
    """Combined weight of the n largest holdings."""
    return float(weights.dropna().nlargest(n).sum())


def concentration_ratio(weights: pd.Series, levels: tuple[int, ...] = (5, 10, 25, 50)) -> pd.Series:
    """Cumulative weight at several concentration breakpoints."""
    return pd.Series({f"Top {n}": top_n_weight(weights, n) for n in levels})


def gini(weights: pd.Series) -> float:
    """Gini coefficient of the weight distribution.

    Zero means every constituent carries identical weight. Values approaching
    one mean weight is concentrated in a shrinking set of names.
    """
    w = np.sort(weights.dropna().values)
    n = len(w)
    if n == 0 or w.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return float((2 * (index * w).sum()) / (n * w.sum()) - (n + 1) / n)


def entropy(weights: pd.Series) -> float:
    """Shannon entropy of the weight distribution in nats."""
    w = weights.dropna()
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted average volatility divided by portfolio volatility.

    A ratio of 1.0 means no diversification benefit at all. Higher is better.
    This captures something Herfindahl misses: a portfolio can hold many names
    and still be undiversified if those names are highly correlated.
    """
    w = weights.reindex(cov.index).fillna(0.0)
    vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
    weighted_avg_vol = float((w * vols).sum())
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    return weighted_avg_vol / port_vol if port_vol else np.nan


def active_share(weights: pd.Series, benchmark_weights: pd.Series) -> float:
    """Share of the portfolio that differs from the benchmark.

    Half the sum of absolute weight differences. Ranges from 0 (identical to
    benchmark) to 1 (completely disjoint holdings).
    """
    aligned = pd.concat([weights, benchmark_weights], axis=1).fillna(0.0)
    return float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum() / 2.0)


def concentration_panel(weights_history: pd.DataFrame) -> pd.DataFrame:
    """Concentration measures over time. One row per rebalance date."""
    rows = {}
    for date, w in weights_history.iterrows():
        rows[date] = {
            "HHI": herfindahl(w),
            "Effective N": effective_n(w),
            "Top 5": top_n_weight(w, 5),
            "Top 10": top_n_weight(w, 10),
            "Gini": gini(w),
            "Entropy": entropy(w),
            "Constituents": int((w > 0).sum()),
        }
    return pd.DataFrame(rows).T
