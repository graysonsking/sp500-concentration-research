"""
run_stress.py
=============
Runs the three-layer stress framework (src/stress.py) and the statistical
inference suite (src/inference.py) over the scheme return series produced by
run_results.py, and writes results/stress_summary.md plus a JSON artifact.

Inputs
------
--returns   Wide CSV of monthly returns: a date index column plus one column
            per scheme (default: results/monthly_returns.csv). Column names
            should match the schemes in the main results table; the benchmark
            column is set with --benchmark (default: cap_weight).
--weights   Optional CSV of the LATEST weight vector per scheme for the
            unwind scenario: columns [scheme, id, weight]
            (default: results/latest_weights.csv; layer 3 is skipped with a
            note if the file is absent).

  python run_stress.py                       # real inputs
  python run_stress.py --demo                # synthetic end-to-end smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:                                   # repo layout: src/ package
    from src import stress, inference
except ImportError:                    # flat layout fallback
    import stress, inference           # noqa: E401


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Returns file not found: {path}\n"
            "Expected a wide CSV with a date index column and one column per "
            "scheme (e.g., cap_weight, equal_weight, min_variance, ...). "
            "Point --returns at the file run_results.py produces, or run --demo."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def load_weights(path: Path) -> dict[str, pd.Series] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = {"scheme", "id", "weight"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Weights file {path} must have columns {sorted(need)}")
    return {s: g.set_index("id")["weight"] for s, g in df.groupby("scheme")}


def demo_inputs(seed: int = 11):
    """Synthetic returns + weights that exercise every code path."""
    rng = np.random.default_rng(seed)
    n = 300
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    market = rng.normal(0.006, 0.042, n)
    market[20:30] -= 0.05        # a synthetic crisis inside the dot-com window
    rets = pd.DataFrame({
        "cap_weight": market,
        "equal_weight": market + rng.normal(0.0004, 0.008, n),
        "min_variance": market * 0.75 + rng.normal(0.0, 0.004, n),
    }, index=idx)
    ids = list(range(1, 51))
    caps = np.array([60 - i for i in ids], dtype=float)
    weights = {
        "cap_weight": pd.Series(caps / caps.sum(), index=ids),
        "equal_weight": pd.Series(1.0 / len(ids), index=ids),
        "min_variance": pd.Series(
            np.r_[np.full(10, 0.002), np.full(40, 0.0245)], index=ids),
    }
    return rets, weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--returns", default="results/monthly_returns.csv")
    ap.add_argument("--weights", default="results/latest_weights.csv")
    ap.add_argument("--benchmark", default="cap_weight")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        rets, weights = demo_inputs()
    else:
        rets = load_returns(Path(args.returns))
        weights = load_weights(Path(args.weights))

    if args.benchmark not in rets.columns:
        # auto-detect display-style names, e.g. "Cap weight (benchmark)"
        candidates = [c for c in rets.columns
                      if "benchmark" in c.lower() or "cap weight" in c.lower()]
        if len(candidates) == 1:
            print(f"[benchmark] '{args.benchmark}' not found; using '{candidates[0]}'")
            args.benchmark = candidates[0]
        else:
            raise SystemExit(
                f"benchmark column '{args.benchmark}' not in {list(rets.columns)}; "
                "pass --benchmark with the exact column name")

    # ---- Layer 1: regime windows -------------------------------------------
    regimes = stress.regime_table(rets)

    # ---- Layer 2: bootstrap risk profile per scheme ------------------------
    boot = {s: stress.bootstrap_risk_profile(rets[s]) for s in rets.columns}
    boot_df = pd.DataFrame(boot).T

    # ---- Layer 3: mega-cap unwind (needs weights) --------------------------
    unwind = None
    if weights is not None and args.benchmark in weights:
        unwind = stress.unwind_table(weights, benchmark_scheme=args.benchmark)

    # ---- Inference ----------------------------------------------------------
    inf = inference.inference_table(rets, benchmark=args.benchmark)

    # ---- Write artifacts -----------------------------------------------------
    payload = {
        "regimes": regimes.to_dict(orient="records"),
        "bootstrap": boot,
        "unwind": (unwind.reset_index().to_dict(orient="records")
                   if unwind is not None else "skipped: no weights file"),
        "inference": inf.reset_index().to_dict(orient="records"),
    }
    (outdir / "stress_results.json").write_text(json.dumps(payload, indent=2, default=str))

    md = ["# Stress Testing & Statistical Inference\n"]
    md.append("## Layer 1 — Historical regime windows\n")
    md.append(regimes.round(4).to_markdown(index=False))
    md.append("\n\n## Layer 2 — Block-bootstrap risk profile "
              "(2,000 resampled paths per scheme)\n")
    md.append(boot_df.round(4).to_markdown())
    md.append("\n\n## Layer 3 — Mega-cap unwind scenario "
              "(benchmark top-10 shocked \u221240%, all else \u221210%)\n")
    md.append(unwind.round(4).to_markdown() if unwind is not None
              else "_Skipped: provide results/latest_weights.csv "
                   "(columns: scheme, id, weight)._")
    md.append("\n\n## Statistical inference vs. benchmark\n")
    md.append(inf.to_markdown())
    md.append("\n\n_Sharpe-difference tests: HAC delta-method (Ledoit-Wolf 2008 "
              "construction) and paired circular block bootstrap. IR test: "
              "Newey-West t-test on mean active return. A high p-value means the "
              "difference is not statistically distinguishable from zero._\n")
    (outdir / "stress_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {outdir/'stress_summary.md'} and stress_results.json")
    print("\nInference vs. benchmark:")
    print(inf.to_string())
    if unwind is not None:
        print("\nUnwind scenario:")
        print(unwind[["mega_weight", "portfolio_return",
                      "protection_vs_benchmark_bps"]].round(4).to_string())


if __name__ == "__main__":
    main()