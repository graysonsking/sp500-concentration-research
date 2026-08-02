"""Find which gvkeyx carries S&P 500 membership WITH history.

comp.idxcst_his holds several index codes. Some are current-member snapshots
(member count equals the index size exactly). Others carry full spell history.
This tests each candidate on two dates twenty years apart: the right code
returns roughly 500 on both.
"""
import pandas as pd
import wrds
import config

CANDIDATES = ["000003", "000010", "165188", "165186", "150918",
              "151015", "165181", "000208", "165157"]
DATES = ["2005-06-30", "2015-06-30", "2024-06-30"]

db = wrds.Connection(wrds_username=config.WRDS_USERNAME)

names = db.raw_sql("SELECT gvkeyx, conm FROM comp.idx_index")
lookup = dict(zip(names["gvkeyx"].astype(str), names["conm"]))

rows = []
for gx in CANDIDATES:
    meta = db.raw_sql("""
        SELECT COUNT(*) AS spells,
               SUM(CASE WHEN thru IS NOT NULL THEN 1 ELSE 0 END) AS closed
        FROM comp.idxcst_his WHERE gvkeyx = %(g)s
    """, params={"g": gx})
    rec = {
        "gvkeyx": gx,
        "name": str(lookup.get(gx, "?"))[:38],
        "spells": int(meta["spells"].iloc[0]),
        "closed": int(meta["closed"].iloc[0] or 0),
    }
    for d in DATES:
        c = db.raw_sql("""
            SELECT COUNT(DISTINCT gvkey) AS n FROM comp.idxcst_his
            WHERE gvkeyx = %(g)s AND "from" <= %(d)s
              AND (thru IS NULL OR thru >= %(d)s)
        """, params={"g": gx, "d": d})
        rec[d[:4]] = int(c["n"].iloc[0])
    rows.append(rec)

df = pd.DataFrame(rows)
print("\nA usable S&P 500 source has closed > 0 and roughly 500 in every year.\n")
print(df.to_string(index=False))

good = df[(df["closed"] > 0) & df[["2005", "2015", "2024"]].ge(450).all(axis=1)
          & df[["2005", "2015", "2024"]].le(530).all(axis=1)]
print("\n" + "=" * 70)
if good.empty:
    print("No code carries full S&P 500 history. Use --membership reconstruction.")
else:
    print("USE THIS CODE. Set INDEX_GVKEYX in config.py to:")
    print(good[["gvkeyx", "name", "closed", "2005", "2015", "2024"]].to_string(index=False))
db.close()
