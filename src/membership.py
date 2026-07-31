"""Point-in-time S&P 500 membership reconstruction.

This is the piece most retail backtests get wrong. Running a strategy over
today's constituent list across a historical window measures the performance
of companies successful enough to still be in the index. That is a selection
effect, not a strategy result, and it inflates measured returns.

The reconstruction works backward. Start from the current roster, then walk
the published index change history in reverse, undoing each addition and
deletion to recover the membership as of any prior date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class IndexChange:
    """A single addition or deletion event."""

    date: pd.Timestamp
    added: str | None
    removed: str | None


class MembershipBuilder:
    """Reconstructs index membership as of any historical date.

    current: tickers in the index as of `as_of`.
    changes: chronological list of addition and deletion events.
    """

    def __init__(
        self,
        current: list[str],
        changes: list[IndexChange],
        as_of: pd.Timestamp | None = None,
    ) -> None:
        self.current = set(current)
        self.changes = sorted(changes, key=lambda c: c.date)
        self.as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
        self._cache: dict[pd.Timestamp, frozenset[str]] = {}

    def constituents_on(self, date) -> list[str]:
        """Membership as of `date`, walking the change log backward."""
        date = pd.Timestamp(date)
        if date in self._cache:
            return sorted(self._cache[date])

        members = set(self.current)
        # Undo every change that happened after the target date, newest first.
        for change in reversed(self.changes):
            if change.date <= date:
                break
            if change.added and change.added in members:
                members.discard(change.added)
            if change.removed:
                members.add(change.removed)

        self._cache[date] = frozenset(members)
        return sorted(members)

    def membership_matrix(self, dates) -> pd.DataFrame:
        """Boolean matrix of membership, dates x tickers.

        Convenient for masking a returns panel to the point-in-time universe.
        """
        rows = {pd.Timestamp(d): self.constituents_on(d) for d in dates}
        universe = sorted({t for members in rows.values() for t in members})
        return pd.DataFrame(
            {t: [t in rows[d] for d in rows] for t in universe},
            index=list(rows.keys()),
        )

    def count_series(self, dates) -> pd.Series:
        """Constituent count over time. A useful sanity check.

        The S&P 500 holds roughly 500 names. Counts drifting far from that
        indicate the change log is incomplete or misparsed.
        """
        return pd.Series(
            {pd.Timestamp(d): len(self.constituents_on(d)) for d in dates},
            name="constituents",
        )

    # ------------------------------------------------------------ persistence

    def to_parquet(self, path: str | Path, dates) -> None:
        """Cache the membership matrix. Reconstruction is slow to repeat."""
        self.membership_matrix(dates).to_parquet(path)

    @staticmethod
    def from_parquet(path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)


def parse_change_history(raw: pd.DataFrame) -> list[IndexChange]:
    """Convert a scraped change table into IndexChange records.

    Expects columns: date, added, removed. Blank cells mean the event was an
    addition only or a removal only.
    """
    changes = []
    for _, row in raw.iterrows():
        added = row.get("added")
        removed = row.get("removed")
        changes.append(
            IndexChange(
                date=pd.Timestamp(row["date"]),
                added=str(added).strip() if pd.notna(added) and str(added).strip() else None,
                removed=str(removed).strip() if pd.notna(removed) and str(removed).strip() else None,
            )
        )
    return changes


def validate_against_reference(
    reconstructed: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Compare reconstructed membership to a reference source such as CRSP.

    Returns per-date counts of matches, false inclusions, and omissions.
    Publish this table. A reconstruction nobody has checked is not evidence.
    """
    dates = reconstructed.index.intersection(reference.index)
    rows = {}
    for d in dates:
        recon = set(reconstructed.columns[reconstructed.loc[d]])
        truth = set(reference.columns[reference.loc[d]])
        rows[d] = {
            "reconstructed": len(recon),
            "reference": len(truth),
            "matched": len(recon & truth),
            "false_inclusions": len(recon - truth),
            "omissions": len(truth - recon),
            "accuracy": len(recon & truth) / len(truth) if truth else float("nan"),
        }
    return pd.DataFrame(rows).T
