"""Momentum weighting configuration.

Trailing twelve month return with a one month skip. The skip is standard:
short horizon returns tend to reverse, so including the most recent month
contaminates the momentum signal with reversal.
"""

from __future__ import annotations

import pandas as pd

from src import weighting

LOOKBACK = 12
SKIP = 1
TOP_QUANTILE = 0.30


def weights_fn(returns: pd.DataFrame, **kwargs) -> pd.Series:
    return weighting.momentum(
        returns,
        lookback=kwargs.get("lookback", LOOKBACK),
        skip=kwargs.get("skip", SKIP),
        top_quantile=kwargs.get("top_quantile", TOP_QUANTILE),
    )
