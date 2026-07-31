"""Alternative index weighting schemes.

Each function takes the data available at a rebalance date and returns a
weight Series. Every scheme is a plug-in for `benchmark.py`, so all six run
through an identical harness and differences in result come from the scheme
rather than the test setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def cap_weight(market_caps: pd.Series) -> pd.Series:
    """Float adjusted market capitalization weighting. The benchmark."""
    mc = market_caps.dropna()
    return mc / mc.sum()


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """1/N across all constituents.

    The cleanest test of the research question. It carries no estimation error
    and maximizes effective N by construction, so it isolates the effect of
    removing cap weighting from the effect of any optimization.
    """
    cols = returns.columns
    return pd.Series(1.0 / len(cols), index=cols)


def minimum_variance(
    returns: pd.DataFrame,
    max_weight: float = 0.05,
    shrinkage: bool = True,
) -> pd.Series:
    """Minimum variance with a position cap.

    The cap matters. Unconstrained minimum variance on a 500 name universe
    concentrates into a handful of low volatility names, which defeats the
    purpose of testing whether the scheme restores diversification.
    """
    clean = returns.dropna(axis=1, how="any")
    if shrinkage:
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for Ledoit-Wolf shrinkage")
        sigma = LedoitWolf().fit(clean.values).covariance_
    else:
        sigma = clean.cov().values

    n = sigma.shape[0]
    result = minimize(
        lambda w: float(w @ sigma @ w),
        x0=np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return pd.Series(result.x, index=clean.columns)


def risk_parity(returns: pd.DataFrame, shrinkage: bool = True) -> pd.Series:
    """Equal risk contribution across constituents."""
    clean = returns.dropna(axis=1, how="any")
    if shrinkage:
        sigma = LedoitWolf().fit(clean.values).covariance_
    else:
        sigma = clean.cov().values

    n = sigma.shape[0]
    target = 1.0 / n

    def objective(w):
        pv = w @ sigma @ w
        if pv <= 0:
            return 1e6
        contrib = w * (sigma @ w) / pv
        return float(((contrib - target) ** 2).sum())

    result = minimize(
        objective,
        x0=np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    return pd.Series(result.x, index=clean.columns)


def fundamental(
    fundamentals: pd.DataFrame,
    measures: tuple[str, ...] = ("revenue", "book_value", "cash_flow", "dividends"),
) -> pd.Series:
    """Weight by accounting measures rather than price.

    The argument for fundamental indexation is that price contains noise, so
    weighting by price mechanically overweights whatever is currently
    overvalued. Weighting by fundamentals breaks that link.

    Each measure is normalized to sum to one, then averaged across measures so
    no single metric dominates because of its scale.
    """
    available = [m for m in measures if m in fundamentals.columns]
    if not available:
        raise ValueError(f"none of {measures} present in fundamentals")

    normalized = []
    for m in available:
        col = fundamentals[m].clip(lower=0).dropna()
        total = col.sum()
        if total > 0:
            normalized.append(col / total)

    combined = pd.concat(normalized, axis=1).mean(axis=1)
    return combined / combined.sum()


def momentum(
    returns: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    top_quantile: float = 0.3,
) -> pd.Series:
    """Weight the trailing return leaders, equal weighted within the group."""
    window = returns.iloc[-(lookback + skip): -skip] if skip else returns.tail(lookback)
    scores = ((1.0 + window).prod() - 1.0).dropna()
    n = max(1, int(len(scores) * top_quantile))
    winners = scores.nlargest(n).index
    w = pd.Series(0.0, index=returns.columns)
    w[winners] = 1.0 / n
    return w


def sentiment(
    sentiment_scores: pd.Series,
    floor: float = 0.0,
    top_quantile: float | None = None,
) -> pd.Series:
    """Weight by FinBERT sentiment scored on SEC filings.

    Scores are shifted to be non-negative before normalizing, since a raw
    sentiment score can be negative and negative weights are not meaningful
    for a long-only index alternative.
    """
    s = sentiment_scores.dropna()
    if top_quantile is not None:
        n = max(1, int(len(s) * top_quantile))
        s = s.nlargest(n)

    shifted = (s - s.min() + floor).clip(lower=1e-9)
    return shifted / shifted.sum()


SCHEMES = {
    "cap_weight": cap_weight,
    "equal_weight": equal_weight,
    "minimum_variance": minimum_variance,
    "risk_parity": risk_parity,
    "fundamental": fundamental,
    "momentum": momentum,
    "sentiment": sentiment,
}
