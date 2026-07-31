"""Tests for concentration measurement. These carry the research conclusion."""

import numpy as np
import pandas as pd
import pytest

from src import metrics


@pytest.fixture
def equal_500():
    return pd.Series(1 / 500, index=[f"T{i}" for i in range(500)])


def test_effective_n_equals_count_when_equal_weighted(equal_500):
    assert metrics.effective_n(equal_500) == pytest.approx(500.0)


def test_effective_n_collapses_under_concentration():
    w = pd.Series([0.5] + [0.5 / 99] * 99, index=[f"T{i}" for i in range(100)])
    assert metrics.effective_n(w) < 5.0


def test_herfindahl_is_inverse_of_effective_n(equal_500):
    assert metrics.herfindahl(equal_500) == pytest.approx(1 / 500)


def test_gini_zero_for_equal_weights(equal_500):
    assert metrics.gini(equal_500) == pytest.approx(0.0, abs=1e-9)


def test_top_n_weight_sums_largest():
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("ABCD"))
    assert metrics.top_n_weight(w, 2) == pytest.approx(0.7)


def test_entropy_maximized_at_equal_weight():
    n = 50
    equal = pd.Series(1 / n, index=range(n))
    skewed = pd.Series([0.5] + [0.5 / (n - 1)] * (n - 1), index=range(n))
    assert metrics.entropy(equal) > metrics.entropy(skewed)
    assert metrics.entropy(equal) == pytest.approx(np.log(n))


def test_active_share_zero_against_self(equal_500):
    assert metrics.active_share(equal_500, equal_500) == pytest.approx(0.0)


def test_active_share_one_for_disjoint_holdings():
    a = pd.Series([0.5, 0.5], index=["A", "B"])
    b = pd.Series([0.5, 0.5], index=["C", "D"])
    assert metrics.active_share(a, b) == pytest.approx(1.0)


def test_diversification_ratio_one_when_perfectly_correlated():
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=list("AB"), columns=list("AB"))
    w = pd.Series([0.5, 0.5], index=list("AB"))
    assert metrics.diversification_ratio(w, cov) == pytest.approx(1.0)
