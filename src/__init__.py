"""S&P 500 concentration risk and alternative weighting research.

Modules
-------
membership : Point-in-time index constituent reconstruction
benchmark  : Backtest engine with a plug-in weighting function
weighting  : Alternative index weighting schemes
metrics    : Concentration and diversification measurement
"""

from . import benchmark, membership, metrics, weighting

__version__ = "0.1.0"
__all__ = ["membership", "benchmark", "weighting", "metrics"]
