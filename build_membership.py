"""Build the point-in-time membership reconstruction.

Neither vendor source carries S&P 500 spell history under this entitlement:
CRSP index files are not licensed, and Compustat's `idxcst_his` for gvkeyx
000003 holds 503 open spells and zero recorded exits, so it is a current
member snapshot. That is verified in `diagnose_history.py`.

This scrapes the current constituent list and the change log from Wikipedia
and walks the log backward through `src/membership.py`.

    python build_membership.py
    python build_membership.py --validate    # compare against Compustat today

Output: data/membership.parquet, consumed by
`run_results.py --membership reconstruction`.

The reconstruction is the weakest link in the data chain and the methodology
should say so plainly. What it buys is coverage of departed constituents,
which no available vendor source provides, and which a concentration study
cannot do without: omitting exits biases every historical date toward
today's winners.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from src.membership import MembershipBuilder, parse_change_history

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def scrape(html_path: str | None = None):
    """Pull the current constituents and the change log.

    Table 0 is the current list; table 1 is the change history with a
    two-level header (Added/Removed each split into Ticker and Security).
    """
    import io

    if html_path:
        print(f"parsing saved page: {html_path}")
        tables = pd.read_html(
            io.StringIO(Path(html_path).read_text(encoding="utf-8", errors="ignore"))
        )
        return _extract(tables)

    import requests

    # Wikipedia rejects urllib's default user agent with a 403, and
    # pandas.read_html(url) uses urllib. Fetch it properly, then parse the
    # HTML string. Their policy asks for a descriptive agent with contact
    # information, so identify the project rather than impersonating a browser.
    headers = {
        "User-Agent": (
            "sp500-concentration-research/1.0 "
            "(academic research; https://github.com/graysonsking) python-requests"
        )
    }
    try:
        resp = requests.get(WIKI, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:
        raise SystemExit(
            f"Could not read Wikipedia ({type(e).__name__}: {e}).\n"
            f"If this is a network or proxy issue, save the page manually as\n"
            f"data/sp500_wiki.html and rerun with --html data/sp500_wiki.html"
        )
    print(f"fetched {len(tables)} tables")
    return _extract(tables)


def _extract(tables):
    """Pull the constituent list and change log out of the parsed tables."""
    current = tables[0]
    tick_col = next(c for c in current.columns if "ymbol" in str(c) or "icker" in str(c))
    tickers = [str(t).strip().upper().replace(".", "-") for t in current[tick_col]]
    print(f"current constituents: {len(tickers)}")

    changes_raw = tables[1]
    changes_raw.columns = [
        "_".join(str(p) for p in c).lower() if isinstance(c, tuple) else str(c).lower()
        for c in changes_raw.columns
    ]

    def find(*keys):
        for c in changes_raw.columns:
            if all(k in c for k in keys):
                return c
        return None

    date_col = find("date")
    add_col = find("added", "ticker") or find("added", "symbol")
    rem_col = find("removed", "ticker") or find("removed", "symbol")
    if not all([date_col, add_col, rem_col]):
        raise SystemExit(f"Unexpected change table columns: {list(changes_raw.columns)}")

    tidy = pd.DataFrame({
        "date": pd.to_datetime(changes_raw[date_col], errors="coerce"),
        "added": changes_raw[add_col].astype(str).str.strip().str.upper()
                 .str.replace(".", "-", regex=False),
        "removed": changes_raw[rem_col].astype(str).str.strip().str.upper()
                   .str.replace(".", "-", regex=False),
    }).dropna(subset=["date"])

    for c in ("added", "removed"):
        tidy[c] = tidy[c].replace({"NAN": None, "": None, "—": None, "-": None})

    print(f"change events: {len(tidy)}, "
          f"{tidy['date'].min():%Y-%m} to {tidy['date'].max():%Y-%m}")
    return tickers, tidy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/membership.parquet")
    ap.add_argument("--html", help="parse a saved copy of the page instead of fetching")
    ap.add_argument("--validate", action="store_true",
                    help="compare the latest reconstructed date against Compustat")
    args = ap.parse_args()

    tickers, tidy = scrape(args.html)
    changes = parse_change_history(tidy)

    builder = MembershipBuilder(current=tickers, changes=changes)
    dates = pd.date_range(config.START, config.END, freq="ME")

    print(f"reconstructing {len(dates)} month ends...")
    matrix = builder.membership_matrix(dates)

    counts = matrix.sum(axis=1)
    print(f"\n{matrix.shape[1]} distinct tickers over the sample")
    print(f"constituent count: min {counts.min()}, max {counts.max()}, "
          f"median {int(counts.median())}")
    print("\ncount by year:")
    print(counts.groupby(counts.index.year).median().astype(int).to_string())

    if counts.median() < 400:
        print("\nWARNING: median below 400. The change log may not reach far "
              "enough back to cover the start of the sample.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out)
    print(f"\nwrote {out}")

    if args.validate:
        import wrds
        from src.membership import validate_against_reference

        print("\nvalidating the most recent date against Compustat...")
        db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
        ref = db.raw_sql("""
            SELECT DISTINCT n.ticker
            FROM comp.idxcst_his AS i
            JOIN crsp.ccmxpf_lnkhist AS l
              ON i.gvkey = l.gvkey AND l.linktype IN ('LU','LC','LS')
             AND l.linkprim IN ('P','C')
            JOIN crsp.msenames AS n
              ON l.lpermno = n.permno AND n.nameendt >= CURRENT_DATE - 400
            WHERE i.gvkeyx = '000003' AND i.thru IS NULL
        """)
        db.close()

        truth = {str(t).strip().upper() for t in ref["ticker"].dropna()}
        last = matrix.index[-1]
        recon = set(matrix.columns[matrix.loc[last]])
        ref_df = pd.DataFrame(
            [[t in truth for t in matrix.columns]],
            index=[last], columns=matrix.columns,
        )
        print(validate_against_reference(matrix.loc[[last]], ref_df).to_string())
        print(f"\nreconstructed {len(recon)}, reference {len(truth)}")
        print("Publish this table. A reconstruction nobody has checked is not evidence.")


if __name__ == "__main__":
    main()
