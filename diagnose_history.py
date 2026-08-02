"""Does comp.idxcst_his actually carry departed members?

A count that rises monotonically from 181 to 470 suggests the table holds
current constituents only. This checks directly.
"""
import wrds
import config

db = wrds.Connection(wrds_username=config.WRDS_USERNAME)

q = db.raw_sql("""
    SELECT
      COUNT(*) AS spells,
      COUNT(DISTINCT gvkey) AS gvkeys,
      SUM(CASE WHEN thru IS NULL THEN 1 ELSE 0 END) AS open_spells,
      SUM(CASE WHEN thru IS NOT NULL THEN 1 ELSE 0 END) AS closed_spells,
      MIN("from") AS earliest_from,
      MIN(thru) AS earliest_thru
    FROM comp.idxcst_his
    WHERE gvkeyx = '000003'
""")
print("\ncomp.idxcst_his, gvkeyx 000003, all history:")
print(q.T.to_string(header=False))

exits = db.raw_sql("""
    SELECT EXTRACT(YEAR FROM thru) AS yr, COUNT(*) AS exits
    FROM comp.idxcst_his
    WHERE gvkeyx = '000003' AND thru IS NOT NULL
    GROUP BY 1 ORDER BY 1
""")
print("\nindex exits by year (expect roughly 20 to 25 per year):")
print(exits.to_string(index=False) if not exits.empty else "  NONE. Table is current members only.")

print("\nother S&P index codes available:")
idx = db.raw_sql("""
    SELECT gvkeyx, COUNT(DISTINCT gvkey) AS members
    FROM comp.idxcst_his GROUP BY 1 ORDER BY 2 DESC LIMIT 12
""")
print(idx.to_string(index=False))
db.close()
