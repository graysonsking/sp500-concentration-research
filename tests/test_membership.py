"""Tests for point-in-time membership reconstruction."""

import pandas as pd
import pytest

from src.membership import IndexChange, MembershipBuilder


@pytest.fixture
def builder():
    changes = [
        IndexChange(pd.Timestamp("2020-06-01"), added="NEW", removed="OLD"),
        IndexChange(pd.Timestamp("2021-03-01"), added="NEWER", removed="STALE"),
    ]
    return MembershipBuilder(current=["KEEP", "NEW", "NEWER"], changes=changes)


def test_current_membership_returned_for_today(builder):
    today = pd.Timestamp("2025-01-01")
    assert builder.constituents_on(today) == ["KEEP", "NEW", "NEWER"]


def test_change_is_undone_before_its_date(builder):
    before = builder.constituents_on("2021-01-01")
    assert "NEWER" not in before
    assert "STALE" in before


def test_all_changes_undone_at_earliest_date(builder):
    earliest = builder.constituents_on("2020-01-01")
    assert set(earliest) == {"KEEP", "OLD", "STALE"}


def test_constituent_count_is_stable(builder):
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="QE")
    counts = builder.count_series(dates)
    assert counts.nunique() == 1, "additions and removals should net to zero"


def test_membership_matrix_shape(builder):
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="QE")
    m = builder.membership_matrix(dates)
    assert len(m) == len(dates)
    assert m.dtypes.unique().tolist() == [bool]
