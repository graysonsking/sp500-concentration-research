"""Backtest harness for index weighting comparison.

Mirrors the plug-in design of the multi-strategy framework. Any scheme
producing a weight Series can be evaluated on identical terms: same universe,
same rebalance dates, same cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .metrics import concentration_panel

WeightsFn = Callable[..., pd.Series]

_ALIAS = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}


def _freq(alias: str) -> str:
    for candidate in (alias, _ALIAS.get(alias.upper(), alias)):
        try:
            pd.tseries.frequencies.to_offset(candidate)
            return candidate
        except ValueError:
            continue
    raise ValueError(f"unsupported frequency: {alias}")


@dataclass
class WeightingBacktest:
    """Compare index weighting schemes on a point-in-time universe.

    returns: monthly constituent returns, dates x tickers.
    membership: boolean membership matrix from MembershipBuilder. Strongly
        recommended. Without it the test carries survivorship bias.
    """

    returns: pd.DataFrame
    membership: pd.DataFrame | None = None
    lookback: int = 60
    rebalance: str = "M"
    cost_bps: float = 5.0

    def __post_init__(self) -> None:
        self.returns = self.returns.sort_index()
        self.rebalance = _freq(self.rebalance)

    def universe_on(self, date: pd.Timestamp) -> list[str]:
        """Point-in-time constituent list."""
        if self.membership is None:
            return list(self.returns.columns)
        prior = self.membership.loc[self.membership.index <= date]
        if prior.empty:
            return list(self.returns.columns)
        row = prior.iloc[-1]
        return [t for t in row.index[row] if t in self.returns.columns]

    def rebalance_dates(self) -> pd.DatetimeIndex:
        idx = self.returns.index
        if len(idx) < self.lookback:
            return pd.DatetimeIndex([])
        marks = idx.to_series().resample(self.rebalance).last().dropna()
        return pd.DatetimeIndex(marks[marks.index >= idx[self.lookback - 1]].values)

    def run(self, weights_fn: WeightsFn, name: str = "scheme", **params):
        """Run one weighting scheme through the schedule."""
        rows = {}
        for date in self.rebalance_dates():
            history = self.returns.loc[self.returns.index < date].tail(self.lookback)
            members = [c for c in self.universe_on(date) if c in history.columns]
            history = history[members].dropna(axis=1, how="any")
            if history.shape[1] < 2:
                continue
            rows[date] = pd.Series(weights_fn(history, **params)).astype(float)

        if not rows:
            raise ValueError("no rebalance produced weights")

        weights = pd.DataFrame(rows).T.fillna(0.0).sort_index()
        return WeightingResult(weights, self.returns, self.cost_bps, name)


@dataclass
class WeightingResult:
    """Returns and concentration statistics for one weighting scheme."""

    weights: pd.DataFrame
    asset_returns: pd.DataFrame
    cost_bps: float
    name: str

    @property
    def turnover(self) -> pd.Series:
        d = self.weights.diff()
        d.iloc[0] = self.weights.iloc[0]
        return d.abs().sum(axis=1) / 2.0

    @property
    def returns(self) -> pd.Series:
        cols = self.weights.columns.intersection(self.asset_returns.columns)
        w = self.weights[cols].reindex(self.asset_returns.index).ffill().fillna(0.0)
        r = self.asset_returns[cols].loc[w.index].fillna(0.0)
        gross = (w.shift(1).fillna(0.0) * r).sum(axis=1)
        cost = self.turnover.reindex(gross.index).fillna(0.0) * (self.cost_bps / 10_000)
        return (gross - cost).rename(self.name)

    @property
    def concentration(self) -> pd.DataFrame:
        """Concentration measures at each rebalance."""
        return concentration_panel(self.weights)
