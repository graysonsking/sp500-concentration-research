"""Variance based weighting configurations.

Position caps are set before running the study, not tuned against results.
The 5 percent cap on minimum variance is the binding methodological choice:
without it the optimizer concentrates into a small set of low volatility
names and the scheme cannot answer whether it restores diversification.
"""

from __future__ import annotations

import pandas as pd

from src import weighting

MAX_WEIGHT = 0.05


def weights_fn(returns: pd.DataFrame, **kwargs) -> pd.Series:
    """Minimum variance with Ledoit-Wolf shrinkage and a position cap."""
    return weighting.minimum_variance(
        returns,
        max_weight=kwargs.get("max_weight", MAX_WEIGHT),
        shrinkage=True,
    )


def risk_parity_fn(returns: pd.DataFrame, **kwargs) -> pd.Series:
    """Equal risk contribution across constituents."""
    return weighting.risk_parity(returns, shrinkage=True)
