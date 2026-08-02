"""Report which WRDS tables this account can actually read.

Run this before anything else. Entitlements vary by institution and the error
messages are unhelpful, so it is faster to probe than to guess.

    python check_wrds_access.py
"""

from __future__ import annotations

import config

PROBES = [
    # (label, schema.table, why it matters)
    ("Index membership (CRSP)", "crsp.msp500list",
     "point-in-time S&P 500 constituents, preferred source"),
    ("Index membership (Compustat)", "comp.idxcst_his",
     "fallback constituents, needs CCM link to reach PERMNO"),
    ("Monthly stock file", "crsp.msf",
     "returns, prices, shares outstanding. Required."),
    ("Security names", "crsp.msenames",
     "ticker and share code history. Required."),
    ("Delisting returns", "crsp.msedelist",
     "final return on delisted names. Survivorship correction."),
    ("Daily stock file", "crsp.dsf",
     "only needed if you move to daily rebalancing"),
    ("Compustat annual", "comp.funda",
     "revenue, book value, cash flow, dividends for fundamental weighting"),
    ("CCM link table", "crsp.ccmxpf_linktable",
     "maps Compustat gvkey to CRSP permno. Required for fundamentals."),
    ("CCM link (alt name)", "crsp.ccmxpf_lnkhist",
     "alternate name for the link table at some institutions"),
]


def main():
    try:
        import wrds
    except ImportError:
        raise SystemExit("wrds not installed. Run: python -m pip install wrds")

    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    print(f"\nconnected as {config.WRDS_USERNAME}\n")
    print(f"{'STATUS':<8}  {'TABLE':<28}  NOTE")
    print("-" * 92)

    available = []
    for label, table, why in PROBES:
        try:
            db.raw_sql(f"SELECT * FROM {table} LIMIT 1")
            print(f"{'OK':<8}  {table:<28}  {label}")
            available.append(table)
        except Exception as e:
            msg = str(e)
            if "permission denied" in msg:
                reason = "no entitlement"
            elif "does not exist" in msg or "relation" in msg:
                reason = "table not found"
            else:
                reason = type(e).__name__
            print(f"{'BLOCKED':<8}  {table:<28}  {reason} — {why}")

    print("\n" + "=" * 92)
    core = {"crsp.msf", "crsp.msenames"}
    if not core.issubset(available):
        print("BLOCKED on the core stock file. Nothing in this project can run.")
        print("Contact your WRDS administrator about CRSP stock entitlements.")
    else:
        print("Core stock data is available.")
        if "crsp.msp500list" in available:
            print("Membership source: crsp.msp500list (preferred).")
        elif "comp.idxcst_his" in available:
            print("Membership source: comp.idxcst_his via CCM link (fallback).")
        else:
            print("Membership source: NEITHER. Fall back to your Wikipedia")
            print("reconstruction in src/membership.py, and document why.")
        if "crsp.msedelist" not in available:
            print("WARNING: no delisting returns. Survivorship bias will remain.")
        if "comp.funda" not in available:
            print("NOTE: no Compustat. Drop 'fundamental' from --schemes.")

    print("\nAlso listing libraries this account can see:\n")
    try:
        libs = db.list_libraries()
        interesting = [l for l in libs if any(
            k in l for k in ("crsp", "comp", "ccm"))]
        print(", ".join(sorted(interesting)) or "(none matched)")
    except Exception as e:
        print(f"could not list libraries: {e}")

    db.close()


if __name__ == "__main__":
    main()
