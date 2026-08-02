"""Run configuration for the capstone backtest.

Requires a WRDS account with CRSP and Compustat entitlements.
"""

from __future__ import annotations

import os
from pathlib import Path

# ==================================================================== WRDS
# The password is never stored here. On first connection the `wrds` package
# offers to write a .pgpass file; accept, and later runs connect silently.
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "graysonking")

# ================================================================== sample
START = "2000-01-01"
END = "2024-12-31"      # CRSP monthly coverage ends here under this subscription

LOOKBACK = 60           # months of trailing returns for estimation
REBALANCE = "ME"         # month end
COST_BPS = 5.0          # one way, on turnover
REPORTING_LAG = 3       # months applied to Compustat datadate

# ================================================================= schemes
MIN_VAR_CAP = 0.05      # position cap on minimum variance
MOMENTUM_LOOKBACK = 12
MOMENTUM_SKIP = 1
MOMENTUM_TOP_QUANTILE = 0.30

# Cap levels for the robustness exhibit (roadmap item 6).
CAP_LEVELS = [0.02, 0.05, 0.10]

# Cost levels for the sensitivity sweep (roadmap item 4).
COST_LEVELS = [5.0, 10.0, 20.0, 40.0]

# ==================================================================== paths
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "images"

# Compustat index code in comp.idxcst_his. 000003 is a current-member
# snapshot with zero recorded exits, which undercounts every historical
# date. Run find_index_code.py to find a code carrying full spell history.
INDEX_GVKEYX = "000003"
