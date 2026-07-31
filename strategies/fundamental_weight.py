"""Fundamental indexation configuration.

Weights come from accounting measures rather than price. The rationale is
that price contains noise, so cap weighting mechanically overweights whatever
is currently rich. Fundamental weighting severs that link.

Measures are lagged to reflect reporting delay. Using a fiscal year figure at
the fiscal year end date assumes the market had data it did not yet have.
"""

from __future__ import annotations

import pandas as pd

from src import weighting

MEASURES = ("revenue", "book_value", "cash_flow", "dividends")
REPORTING_LAG_MONTHS = 3


def weights_fn(returns: pd.DataFrame, fundamentals: pd.DataFrame | None = None, **kwargs) -> pd.Series:
    """Composite fundamental weights over the current universe.

    fundamentals: DataFrame indexed by ticker with the measure columns.
        Must already be lagged for reporting delay.
    """
    if fundamentals is None:
        raise ValueError("fundamental weighting requires a fundamentals frame")

    subset = fundamentals.reindex(returns.columns).dropna(how="all")
    return weighting.fundamental(subset, measures=kwargs.get("measures", MEASURES))
