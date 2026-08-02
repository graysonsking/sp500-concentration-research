"""CRSP and Compustat retrieval through WRDS.

Replaces the Wikipedia change-history reconstruction in `membership.py`. CRSP
carries the index membership file directly, so point-in-time constituents are
read rather than inferred, which removes the largest approximation in the
methodology.

Everything is keyed on PERMNO, not ticker. Tickers are reused and reassigned
across companies; PERMNO is stable for the life of the security. Tickers are
carried alongside for display only.

All pulls are cached to parquet. WRDS queries are slow and rate limited, so
nothing here should run twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


def connect(username: str | None = None):
    """Open a WRDS session.

    On first use this prompts for a password and offers to create a `.pgpass`
    file. Say yes: subsequent runs then connect without prompting.
    """
    try:
        import wrds
    except ImportError:
        raise SystemExit("wrds not installed. Run: python -m pip install wrds")
    return wrds.Connection(wrds_username=username or config.WRDS_USERNAME)


def _cache(name: str) -> Path:
    config.CACHE_DIR.mkdir(exist_ok=True)
    return config.CACHE_DIR / name


# ------------------------------------------------------------- membership


def _membership_from_crsp(db, start, end):
    """Preferred source: crsp.msp500list. Needs the crsp_a_indexes schema."""
    return db.raw_sql(
        """
        SELECT permno, start, ending
        FROM crsp.msp500list
        WHERE ending >= %(start)s AND start <= %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["start", "ending"],
    )


def _membership_from_compustat(db, start, end):
    """Fallback: Compustat index constituents mapped to PERMNO via CCM.

    `comp.idxcst_his` is keyed on GVKEY, so it has to be linked through the
    CCM table to reach PERMNO. The index code comes from config.INDEX_GVKEYX:
    000003 is the S&P 500 current-member snapshot, which carries no exits.
    Run find_index_code.py to identify a code with full spell history.

    `from` and `thru` are reserved words in Postgres and must stay quoted.
    """
    raw = db.raw_sql(
        """
        SELECT i.gvkey, i."from" AS start, i.thru AS ending,
               l.lpermno AS permno, l.linkdt, l.linkenddt
        FROM comp.idxcst_his AS i
        JOIN crsp.ccmxpf_lnkhist AS l
          ON i.gvkey = l.gvkey
         AND l.linktype IN ('LU', 'LC', 'LS')
         AND l.linkprim IN ('P', 'C')
        WHERE i.gvkeyx = %(gx)s
          AND (i.thru IS NULL OR i.thru >= %(start)s)
          AND i."from" <= %(end)s
        """,
        params={"start": start, "end": end,
                "gx": getattr(config, "INDEX_GVKEYX", "000003")},
        date_cols=["start", "ending", "linkdt", "linkenddt"],
    )
    raw = raw.dropna(subset=["permno"])

    # Clip each membership spell to the window where the GVKEY-PERMNO link is
    # actually valid. Without this a spell is credited to a permno that the
    # gvkey was not linked to at the time, and spells whose link window does
    # not overlap the membership window are silently kept as bad rows.
    raw["ending"] = raw["ending"].fillna(pd.Timestamp(end))
    raw["linkenddt"] = raw["linkenddt"].fillna(pd.Timestamp(end))
    raw["start"] = raw[["start", "linkdt"]].max(axis=1)
    raw["ending"] = raw[["ending", "linkenddt"]].min(axis=1)
    raw = raw[raw["start"] <= raw["ending"]]

    return raw[["permno", "start", "ending"]]


def fetch_membership(db, start: str, end: str) -> pd.DataFrame:
    """Monthly point-in-time S&P 500 membership, dates x permno, boolean.

    Tries CRSP first, then Compustat. Entitlements differ by institution and
    many subscriptions include the CRSP stock files without the index files,
    so the fallback is the normal path rather than an edge case.

    A permno can appear more than once if a company left the index and later
    returned, so spells are expanded and unioned rather than assumed unique.
    """
    path = _cache(f"membership_{start}_{end}.parquet")
    if path.exists():
        print(f"cache hit: {path.name}")
        return _validate_membership(pd.read_parquet(path))

    spells, source = None, None
    for label, fn in (("crsp.msp500list", _membership_from_crsp),
                      ("comp.idxcst_his", _membership_from_compustat)):
        try:
            print(f"trying index membership from {label}...")
            spells = fn(db, start, end)
            if spells is not None and not spells.empty:
                source = label
                break
        except Exception as e:
            reason = "no entitlement" if "permission denied" in str(e) else type(e).__name__
            print(f"  unavailable ({reason})")

    if spells is None or spells.empty:
        raise SystemExit(
            "No index membership source available.\n"
            "Run check_wrds_access.py to confirm, then either request the\n"
            "crsp_a_indexes entitlement or fall back to the reconstruction in\n"
            "src/membership.py and document the choice in the methodology."
        )

    print(f"  using {source}")
    month_ends = pd.date_range(start, end, freq="ME")
    permnos = sorted(spells["permno"].astype(int).unique())
    mat = pd.DataFrame(False, index=month_ends, columns=permnos)

    for _, row in spells.iterrows():
        s = pd.Timestamp(row["start"])
        e = pd.Timestamp(row["ending"]) if pd.notna(row["ending"]) else month_ends[-1]
        mask = (month_ends >= s) & (month_ends <= e)
        mat.loc[mask, int(row["permno"])] = True

    mat.attrs["source"] = source
    mat.to_parquet(path)
    print(f"  using {source}")
    return _validate_membership(mat)


def _validate_membership(mat: pd.DataFrame) -> pd.DataFrame:
    """Sanity check the constituent count. Runs on cached loads too.

    A silent undercount is the worst failure mode for this project: the
    backtest still runs, the tables still format, and every concentration
    number is wrong in the direction that flatters the thesis.
    """
    counts = mat.sum(axis=1)
    med = int(counts.median())
    print(f"  {mat.shape[1]} distinct permnos over the sample, "
          f"count ranges {counts.min()} to {counts.max()}, median {med}")

    if med < 440 or med > 530:
        raise SystemExit(
            f"\nMembership validation FAILED: median constituent count is {med},\n"
            f"expected 450 to 520 for the S&P 500.\n\n"
            f"The backtest would still run and produce plausible looking numbers,\n"
            f"so this is a hard stop rather than a warning.\n\n"
            f"Run diagnose_membership.py to find which join stage drops rows,\n"
            f"then delete cache/membership_*.parquet and pull again."
        )
    if med < 490:
        shortfall = 503 - med
        print(f"  NOTE: median of {med} against a true count of about 503, a\n"
              f"        shortfall of roughly {shortfall} names per month. For the\n"
              f"        reconstruction path this is the ticker to PERMNO mapping:\n"
              f"        tickers with no CRSP name record on that date are dropped.\n"
              f"        Report the coverage rate in the methodology.")
    if mat.shape[1] < 800:
        print(f"  NOTE: only {mat.shape[1]} distinct permnos across the sample. "
              f"Roughly 1,000 companies pass through the index over 25 years.")
    return mat


# ------------------------------------------------------------ prices


def fetch_monthly(db, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Monthly returns, market caps, and a permno to ticker map.

    Returns are CRSP `ret`, which includes dividends. Market cap is
    |prc| * shrout; the absolute value matters because CRSP signs the price
    negative when it is a bid/ask midpoint rather than a close.

    Delisting returns are merged in. Omitting them is the classic source of
    survivorship bias in CRSP work: a company that goes to zero simply stops
    appearing, and its final loss never enters the return series.

    No share code filter is applied. The usual `shrcd IN (10, 11)` screen is
    correct for asset pricing studies that want clean domestic common equity,
    but wrong for index replication: it drops foreign-incorporated members
    (shrcd 12) and REITs (shrcd 18), which together account for roughly 50 of
    the 500. Membership already defines the universe, so the screen is
    redundant here and biases the result.
    """
    r_path, m_path, t_path = (
        _cache(f"returns_{start}_{end}.parquet"),
        _cache(f"mktcap_{start}_{end}.parquet"),
        _cache(f"tickers_{start}_{end}.parquet"),
    )
    if r_path.exists():
        print(f"cache hit: {r_path.name}")
        return (
            pd.read_parquet(r_path),
            pd.read_parquet(m_path),
            pd.read_parquet(t_path)["ticker"],
        )

    print("pulling monthly returns and market caps from crsp.msf...")
    raw = db.raw_sql(
        """
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout, b.ticker, b.shrcd
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
          ON a.permno = b.permno
         AND b.namedt <= a.date
         AND a.date <= b.nameendt
        WHERE a.date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["date"],
    )

    print("pulling delisting returns from crsp.msedelist...")
    dl = db.raw_sql(
        """
        SELECT permno, dlstdt AS date, dlret
        FROM crsp.msedelist
        WHERE dlstdt BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["date"],
    )

    raw["date"] = raw["date"] + pd.offsets.MonthEnd(0)
    if not dl.empty:
        dl["date"] = dl["date"] + pd.offsets.MonthEnd(0)
        raw = raw.merge(dl, on=["permno", "date"], how="left")
        # Compound the delisting return into the final monthly return.
        both = raw["ret"].notna() & raw["dlret"].notna()
        raw.loc[both, "ret"] = (1 + raw.loc[both, "ret"]) * (1 + raw.loc[both, "dlret"]) - 1
        only = raw["ret"].isna() & raw["dlret"].notna()
        raw.loc[only, "ret"] = raw.loc[only, "dlret"]
        print(f"  merged {int(dl.shape[0])} delisting returns")

    raw["permno"] = raw["permno"].astype(int)
    raw["mktcap"] = raw["prc"].abs() * raw["shrout"]

    if "shrcd" in raw.columns:
        mix = raw.drop_duplicates("permno")["shrcd"].value_counts().sort_index()
        print("  share code mix: " + ", ".join(f"{int(k)}:{v}" for k, v in mix.items()))

    returns = raw.pivot_table(index="date", columns="permno", values="ret")
    mktcap = raw.pivot_table(index="date", columns="permno", values="mktcap")
    tickers = (
        raw.dropna(subset=["ticker"])
        .sort_values("date")
        .groupby("permno")["ticker"]
        .last()
    )

    returns.to_parquet(r_path)
    mktcap.to_parquet(m_path)
    tickers.to_frame().to_parquet(t_path)
    print(f"  {returns.shape[0]} months, {returns.shape[1]} securities")
    return returns, mktcap, tickers


# ------------------------------------------------------ fundamentals


def fetch_fundamentals(db, start: str, end: str, lag_months: int = 3) -> pd.DataFrame:
    """Annual Compustat fundamentals mapped to PERMNO, with a reporting lag.

    The lag is the point. `datadate` is the fiscal period end, not the date the
    figures became public. Weighting on a fiscal year that had not yet been
    filed is look-ahead, and it is the most common error in fundamental
    indexation backtests.

    Returns a long frame: date, permno, revenue, book_value, cash_flow,
    dividends. `date` is already lagged and snapped to month end.
    """
    path = _cache(f"fundamentals_{start}_{end}.parquet")
    if path.exists():
        print(f"cache hit: {path.name}")
        return pd.read_parquet(path)

    print("pulling Compustat fundamentals and CCM link...")
    raw = db.raw_sql(
        """
        SELECT DISTINCT
            c.gvkey, c.datadate,
            c.revt, c.ceq, c.oancf, c.dvc,
            l.lpermno AS permno
        FROM comp.funda AS c
        JOIN crsp.ccmxpf_linktable AS l
          ON c.gvkey = l.gvkey
         AND l.linktype IN ('LU', 'LC')
         AND l.linkprim IN ('P', 'C')
         AND l.linkdt <= c.datadate
         AND (c.datadate <= l.linkenddt OR l.linkenddt IS NULL)
        WHERE c.indfmt = 'INDL' AND c.datafmt = 'STD'
          AND c.popsrc = 'D' AND c.consol = 'C'
          AND c.datadate BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["datadate"],
    )

    raw = raw.dropna(subset=["permno"])
    raw["permno"] = raw["permno"].astype(int)
    raw["date"] = (
        raw["datadate"] + pd.DateOffset(months=lag_months) + pd.offsets.MonthEnd(0)
    )

    out = raw.rename(
        columns={"revt": "revenue", "ceq": "book_value",
                 "oancf": "cash_flow", "dvc": "dividends"}
    )[["date", "permno", "revenue", "book_value", "cash_flow", "dividends"]]

    out.to_parquet(path)
    print(f"  {len(out)} firm-years, {out['permno'].nunique()} permnos, "
          f"{lag_months} month reporting lag applied")
    return out


def fundamentals_asof(fundamentals: pd.DataFrame, date, permnos) -> pd.DataFrame:
    """Most recent lagged fiscal year available on or before `date`."""
    hist = fundamentals[fundamentals["date"] <= date]
    if hist.empty:
        return pd.DataFrame()
    latest = hist.sort_values("date").groupby("permno").last()
    return latest.reindex([p for p in permnos if p in latest.index])

def membership_from_reconstruction(db, start: str, end: str,
                                   path: str = "data/membership.parquet") -> pd.DataFrame:
    """Third source: the project's own reconstruction, mapped to PERMNO.

    `src/membership.py` builds point-in-time membership from index change
    history. That output is keyed on ticker, so it is mapped here to PERMNO
    through `crsp.msenames` with the name window enforced, which is what makes
    the mapping point-in-time rather than as-of-today.

    This is the fallback when neither vendor membership table carries full
    history. It is the weaker source and the methodology should say so, but it
    covers departed constituents, which is the property that matters most for
    a concentration study.
    """
    src = Path(path)
    if not src.exists():
        raise SystemExit(
            f"Reconstruction not found at {src}.\n"
            f"Build it first with src/membership.py, then rerun."
        )

    recon = pd.read_parquet(src)
    recon.index = pd.to_datetime(recon.index)
    tickers = [str(c).upper() for c in recon.columns]

    print(f"mapping {len(tickers)} tickers to PERMNO via crsp.msenames...")
    names = db.raw_sql(
        """
        SELECT permno, ticker, namedt, nameendt
        FROM crsp.msenames
        WHERE ticker IN %(t)s
        """,
        params={"t": tuple(tickers)},
        date_cols=["namedt", "nameendt"],
    )
    names["ticker"] = names["ticker"].str.upper()

    month_ends = pd.date_range(start, end, freq="ME")
    permnos = sorted(names["permno"].astype(int).unique())
    mat = pd.DataFrame(False, index=month_ends, columns=permnos)

    for date in month_ends:
        if date not in recon.index:
            prior = recon.index[recon.index <= date]
            if len(prior) == 0:
                continue
            row = recon.loc[prior[-1]]
        else:
            row = recon.loc[date]

        active = {str(t).upper() for t in row.index[row.astype(bool)]}
        # Enforce the name window so a reused ticker maps to the company that
        # actually held it on this date.
        valid = names[
            names["ticker"].isin(active)
            & (names["namedt"] <= date)
            & (names["nameendt"] >= date)
        ]
        mat.loc[date, valid["permno"].astype(int).unique()] = True

    unmapped = len(set(tickers)) - names["ticker"].nunique()
    if unmapped:
        print(f"  {unmapped} tickers had no PERMNO match")
    return _validate_membership(mat)
