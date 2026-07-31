"""Weighting scheme configurations under test.

Each module wraps a scheme from `src/weighting.py` with the exact parameters
used in the study. Keeping the configuration here rather than in the notebook
means the published results are reproducible from the repository alone.
"""

from . import fundamental_weight, momentum_weight, sentiment_weight, variance_weight

REGISTRY = {
    "equal_weight": None,          # see src.weighting.equal_weight, no parameters
    "minimum_variance": variance_weight.weights_fn,
    "risk_parity": variance_weight.risk_parity_fn,
    "fundamental": fundamental_weight.weights_fn,
    "momentum": momentum_weight.weights_fn,
    "sentiment": sentiment_weight.weights_fn,
}

__all__ = ["REGISTRY", "variance_weight", "fundamental_weight", "momentum_weight", "sentiment_weight"]
