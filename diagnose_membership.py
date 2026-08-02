"""Find where S&P 500 constituents are being lost.

The Compustat membership fallback links GVKEY to PERMNO through CCM. That join
drops rows for several distinct reasons, and this script counts survivors at
each stage so the cause is identified rather than guessed.

    python diagnose_membership.py
"""

from __future__ import annotations

import pandas as pd
import wrds

import config

TEST_DATE = "2024-06-30"   # a date where the true count is known to be ~503


def main():
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    print(f"\ndiagnosing membership as of {TEST_DATE}\n")

    # Stage 1: raw Compustat index constituents, no link at all.
    raw = db.raw_sql(
        """
        SELECT gvkey, "from" AS start, thru AS ending
        FROM comp.idxcst_his
        WHERE gvkeyx = '000003'
          AND "from" <= %(d)s
          AND (thru IS NULL OR thru >= %(d)s)
        """,
        params={"d": TEST_DATE},
        date_cols=["start", "ending"],
    )
    print(f"stage 1  raw Compustat constituents on {TEST_DATE}: {len(raw)}")
    print(f"         distinct gvkeys: {raw['gvkey'].nunique()}")
    print("         (expect roughly 500)\n")

    # Stage 2: same, joined to CCM with the filters currently in use.
    linked = db.raw_sql(
        """
        SELECT i.gvkey, l.lpermno AS permno, l.linktype, l.linkprim,
               l.linkdt, l.linkenddt
        FROM comp.idxcst_his AS i
        JOIN crsp.ccmxpf_linktable AS l
          ON i.gvkey = l.gvkey
         AND l.linktype IN ('LU', 'LC')
         AND l.linkprim IN ('P', 'C')
        WHERE i.gvkeyx = '000003'
          AND i."from" <= %(d)s
          AND (i.thru IS NULL OR i.thru >= %(d)s)
        """,
        params={"d": TEST_DATE},
        date_cols=["linkdt", "linkenddt"],
    )
    print(f"stage 2  after CCM join (current filters): {len(linked)} rows")
    print(f"         distinct gvkeys: {linked['gvkey'].nunique()}")
    print(f"         distinct permnos: {linked['permno'].nunique()}")
    lost = set(raw["gvkey"]) - set(linked["gvkey"])
    print(f"         gvkeys with NO link at all: {len(lost)}\n")

    # Stage 3: apply the link date window, which the current query omits.
    d = pd.Timestamp(TEST_DATE)
    valid = linked[
        (linked["linkdt"] <= d)
        & (linked["linkenddt"].isna() | (linked["linkenddt"] >= d))
    ]
    print(f"stage 3  links valid on {TEST_DATE}: {len(valid)} rows")
    print(f"         distinct permnos: {valid['permno'].nunique()}")
    dupes = valid.groupby("gvkey")["permno"].nunique()
    print(f"         gvkeys mapping to >1 permno: {(dupes > 1).sum()}")
    print("         (this is the number that should be near 500)\n")

    # Stage 4: how many of those permnos exist in msf with shrcd 10/11.
    permnos = valid["permno"].dropna().astype(int).unique().tolist()
    if permnos:
        in_msf = db.raw_sql(
            """
            SELECT DISTINCT a.permno
            FROM crsp.msf AS a
            JOIN crsp.msenames AS b
              ON a.permno = b.permno
             AND b.namedt <= a.date AND a.date <= b.nameendt
            WHERE a.date BETWEEN '2024-06-01' AND '2024-06-30'
              AND b.shrcd IN (10, 11)
              AND a.permno IN %(p)s
            """,
            params={"p": tuple(permnos)},
        )
        print(f"stage 4  present in crsp.msf with shrcd 10/11: {len(in_msf)}")
        missing = set(permnos) - set(in_msf["permno"].astype(int))
        print(f"         dropped by the shrcd filter or absent: {len(missing)}\n")

    # Stage 5: does the alternate link table do better?
    try:
        alt = db.raw_sql(
            """
            SELECT i.gvkey, l.lpermno AS permno
            FROM comp.idxcst_his AS i
            JOIN crsp.ccmxpf_lnkhist AS l
              ON i.gvkey = l.gvkey
             AND l.linktype IN ('LU', 'LC', 'LS')
             AND l.linkprim IN ('P', 'C')
             AND l.linkdt <= %(d)s
             AND (l.linkenddt IS NULL OR l.linkenddt >= %(d)s)
            WHERE i.gvkeyx = '000003'
              AND i."from" <= %(d)s
              AND (i.thru IS NULL OR i.thru >= %(d)s)
            """,
            params={"d": TEST_DATE},
        )
        print(f"stage 5  via crsp.ccmxpf_lnkhist, LU/LC/LS: "
              f"{alt['permno'].nunique()} distinct permnos")
    except Exception as e:
        print(f"stage 5  lnkhist query failed: {type(e).__name__}")

    # Stage 6: what is the actual end of the return data?
    end = db.raw_sql("SELECT MAX(date) AS d FROM crsp.msf")
    print(f"\nlatest date in crsp.msf: {end['d'].iloc[0]}")

    db.close()
    print("\nRead the stage where the count falls off. That is the cause.")


if __name__ == "__main__":
    main()
