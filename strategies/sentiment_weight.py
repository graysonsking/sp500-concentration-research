"""AI sentiment weighting configuration.

FinBERT scores computed on SEC EDGAR filings, aggregated to a per-issuer
score, then converted into index weights.

Two timing hazards to respect. Filings must be dated by their acceptance
timestamp, not by the period they cover, because the market cannot react to a
filing before it exists. And FinBERT was trained on a specific corpus, so its
calibration on filing language should be checked rather than assumed.
"""

from __future__ import annotations

import pandas as pd

from src import weighting

TOP_QUANTILE = 0.50


def weights_fn(
    returns: pd.DataFrame,
    sentiment_scores: pd.Series | None = None,
    **kwargs,
) -> pd.Series:
    """Weight by sentiment score over the current universe.

    sentiment_scores: per-ticker net sentiment as of the rebalance date,
        computed only from filings accepted before that date.
    """
    if sentiment_scores is None:
        raise ValueError("sentiment weighting requires a score series")

    subset = sentiment_scores.reindex(returns.columns).dropna()
    if subset.empty:
        raise ValueError("no sentiment scores available for the current universe")

    return weighting.sentiment(subset, top_quantile=kwargs.get("top_quantile", TOP_QUANTILE))
