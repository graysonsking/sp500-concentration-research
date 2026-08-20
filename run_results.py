"""Reproduce the capstone results.

Pulls CRSP point-in-time membership, returns, and market caps, plus Compustat
fundamentals, then runs every weighting scheme through an identical
walk-forward schedule and reports both performance and concentration.

    python run_results.py                 # full run
    python run_results.py --offline       # synthetic data, tests the plumbing
    python run_results.py --sweeps        # add cost and cap robustness
    python run_results.py --schemes cap,equal,minvar

Settings live in `config.py`. Sentiment is excluded unless a scored panel is
supplied via --sentiment, since that pipeline is separate.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config
from src import metrics, weighting
from src.benchmark import WeightingBacktest


# --------------------------------------------------------------- data


def load_real(membership_source="auto", recon_path="data/membership.parquet"):
    from src import wrds_data

    db = wrds_data.connect()
    try:
        if membership_source == "reconstruction":
            membership = wrds_data.membership_from_reconstruction(
                db, config.START, config.END, recon_path
            )
        else:
            membership = wrds_data.fetch_membership(db, config.START, config.END)
        returns, mktcap, tickers = wrds_data.fetch_monthly(db, config.START, config.END)
        fundamentals = wrds_data.fetch_fundamentals(
            db, config.START, config.END, config.REPORTING_LAG
        )
    finally:
        db.close()

    # Keep only securities that are index members at some point AND have both
    # a return and a market cap series. CRSP can carry a return for a month
    # with no usable price or shares outstanding, so the two pivots do not
    # always share a column set.
    members = [c for c in membership.columns
               if c in returns.columns and c in mktcap.columns]
    dropped = len([c for c in membership.columns if c in returns.columns]) - len(members)
    if dropped:
        print(f"  {dropped} permnos have returns but no market cap; excluded")
    returns = returns[members]
    mktcap = mktcap[members]
    membership = membership[members]
    print(f"  final universe: {len(members)} securities")
    return returns, mktcap, membership, fundamentals, tickers


def load_synthetic(n=120, months=320):
    rng = np.random.default_rng(0)
    idx = pd.date_range(config.START, periods=months, freq="ME")
    permnos = list(range(10000, 10000 + n))

    mkt = rng.standard_normal(months) * 0.045 + 0.006
    beta = rng.uniform(0.5, 1.6, n)
    idio = rng.standard_normal((months, n)) * rng.uniform(0.03, 0.09, n)
    returns = pd.DataFrame(mkt[:, None] * beta + idio, index=idx, columns=permnos)

    # Cap weights that concentrate over time, mimicking the phenomenon studied.
    size = np.exp(np.cumsum(returns.values, axis=0)) * rng.lognormal(10, 1.6, n)
    mktcap = pd.DataFrame(size, index=idx, columns=permnos)

    membership = pd.DataFrame(True, index=idx, columns=permnos)
    for p in permnos[: n // 4]:            # some names enter late
        membership.loc[: idx[rng.integers(20, 120)], p] = False

    fundamentals = pd.DataFrame(
        [
            {"date": d, "permno": p,
             "revenue": float(mktcap.loc[d, p]) * rng.uniform(0.3, 1.5),
             "book_value": float(mktcap.loc[d, p]) * rng.uniform(0.1, 0.6),
             "cash_flow": float(mktcap.loc[d, p]) * rng.uniform(0.02, 0.2),
             "dividends": float(mktcap.loc[d, p]) * rng.uniform(0.0, 0.05)}
            for d in idx[::12] for p in permnos
        ]
    )
    tickers = pd.Series({p: f"SYN{p}" for p in permnos})
    return returns, mktcap, membership, fundamentals, tickers


# ------------------------------------------------------- scheme builders


def build_schemes(mktcap, fundamentals, sentiment_panel, names, cap=None):
    """Weighting functions matching the harness contract.

    The harness calls `weights_fn(history, **params)` and passes no date, so
    date-dependent inputs are looked up from `history.index[-1]`, which is the
    last month strictly before the rebalance. That keeps every scheme on the
    same information set and preserves the look-ahead guard.
    """
    cap = cap if cap is not None else config.MIN_VAR_CAP

    def cap_weight(history, **_):
        d = history.index[-1]
        row = mktcap.loc[d, [c for c in history.columns if c in mktcap.columns]]
        return weighting.cap_weight(row.dropna())

    def fundamental(history, **_):
        from src.wrds_data import fundamentals_asof

        d = history.index[-1]
        f = fundamentals_asof(fundamentals, d, list(history.columns))
        if f.empty or len(f) < 20:
            return weighting.equal_weight(history)
        return weighting.fundamental(f)

    def sentiment(history, **_):
        d = history.index[-1]
        prior = sentiment_panel.loc[sentiment_panel.index <= d]
        if prior.empty:
            return weighting.equal_weight(history)
        scores = prior.iloc[-1].reindex(history.columns).dropna()
        if len(scores) < 20:
            return weighting.equal_weight(history)
        return weighting.sentiment(scores)

    all_schemes = {
        "Cap weight (benchmark)": cap_weight,
        "Equal weight": lambda h, **_: weighting.equal_weight(h),
        "Minimum variance": lambda h, **_: weighting.minimum_variance(
            h, max_weight=cap, shrinkage=True
        ),
        "Risk parity": lambda h, **_: weighting.risk_parity(h, shrinkage=True),
        "Momentum": lambda h, **_: weighting.momentum(
            h,
            lookback=config.MOMENTUM_LOOKBACK,
            skip=config.MOMENTUM_SKIP,
            top_quantile=config.MOMENTUM_TOP_QUANTILE,
        ),
        "Fundamental": fundamental,
        "Sentiment": sentiment,
    }
    return {k: v for k, v in all_schemes.items() if k in names}


SHORT = {
    "cap": "Cap weight (benchmark)", "equal": "Equal weight",
    "minvar": "Minimum variance", "riskparity": "Risk parity",
    "momentum": "Momentum", "fundamental": "Fundamental", "sentiment": "Sentiment",
}


# ------------------------------------------------------------- statistics


def performance(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    total = float((1 + r).prod())
    years = len(r) / 12
    cagr = total ** (1 / years) - 1 if total > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(12))
    curve = (1 + r).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    out = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": cagr / vol if vol else np.nan,
        "Max Drawdown": dd,
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        active = r.loc[common] - b.loc[common]
        te = float(active.std(ddof=1) * np.sqrt(12))
        out["Tracking Error"] = te
        out["Information Ratio"] = float(active.mean() * 12 / te) if te else np.nan
    return pd.Series(out)


def concentration_summary(result) -> pd.Series:
    """Average concentration across rebalances."""
    w = result.weights
    rows = {
        "Effective N": w.apply(metrics.effective_n, axis=1).mean(),
        "HHI": w.apply(metrics.herfindahl, axis=1).mean(),
        "Top 10 Weight": w.apply(lambda r: metrics.top_n_weight(r, 10), axis=1).mean(),
        "Gini": w.apply(metrics.gini, axis=1).mean(),
        "Avg Turnover": result.turnover.mean(),
    }
    return pd.Series(rows)


# -------------------------------------------------------------- reporting


def fmt(stats: pd.DataFrame) -> pd.DataFrame:
    pct = ["CAGR", "Volatility", "Max Drawdown", "Tracking Error", "Top 10 Weight"]
    out = pd.DataFrame(index=stats.index)
    for c in stats.columns:
        v = ((stats[c] * 100).round(2).astype(str) + "%") if c in pct else stats[c].round(3).astype(str)
        out[c] = v.where(stats[c].notna(), "—")
    out.index.name = "Scheme"
    return out


def write_outputs(stats, curves, conc_ts, meta, sweeps=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(config.RESULTS_DIR / "summary.csv")

    lines = ["# Capstone Results", "", "Generated by `run_results.py`.", "",
             "## Run parameters", ""]
    lines += [f"- **{k}:** {v}" for k, v in meta.items()]
    lines += ["", "## Performance and concentration", "", fmt(stats).to_markdown(), ""]
    if sweeps:
        for title, table in sweeps.items():
            lines += [f"## {title}", "", fmt(table).to_markdown(), ""]
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, c in curves.items():
        style = "--" if "benchmark" in name else "-"
        lw = 2.0 if "benchmark" in name else 1.4
        c.plot(ax=ax, linewidth=lw, linestyle=style,
               color="black" if "benchmark" in name else None, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1.00 (log scale)")
    ax.set_xlabel("")
    ax.set_title("Weighting schemes, net of transaction costs")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)

    # The exhibit the thesis is actually about.
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, s in conc_ts.items():
        style = "--" if "benchmark" in name else "-"
        s.plot(ax=ax, linewidth=1.8 if "benchmark" in name else 1.3,
               linestyle=style, color="black" if "benchmark" in name else None,
               label=name)
    ax.set_ylabel("Effective number of holdings")
    ax.set_xlabel("")
    ax.set_title("Effective N through time")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "effective_n.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote {config.RESULTS_DIR / 'summary.md'}")
    print(f"wrote {config.FIGURES_DIR / 'effective_n.png'}")
    print("\n" + "=" * 70)
    print("PASTE THIS INTO THE README RESULTS SECTION")
    print("=" * 70 + "\n")
    print(fmt(stats).to_markdown())
    print("\n![Effective N](docs/images/effective_n.png)\n")


# ----------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sweeps", action="store_true", help="cost and cap robustness")
    ap.add_argument("--schemes", default="cap,equal,minvar,riskparity,momentum,fundamental")
    ap.add_argument("--sentiment", help="parquet of dates x permno sentiment scores")
    ap.add_argument("--cost-bps", type=float, default=config.COST_BPS)
    ap.add_argument("--membership", default="auto",
                    choices=["auto", "reconstruction"],
                    help="'auto' tries CRSP then Compustat; 'reconstruction' "
                         "uses the project's own point-in-time build")
    ap.add_argument("--recon-path", default="data/membership.parquet")
    args = ap.parse_args()

    if args.offline:
        print("OFFLINE MODE: synthetic data. The numbers are meaningless.\n")
        returns, mktcap, membership, fundamentals, tickers = load_synthetic()
    else:
        returns, mktcap, membership, fundamentals, tickers = load_real(
            args.membership, args.recon_path
        )

    sentiment_panel = pd.DataFrame()
    if args.sentiment:
        sentiment_panel = pd.read_parquet(args.sentiment)
        print(f"loaded sentiment panel: {sentiment_panel.shape}")
        if "sentiment" not in args.schemes:
            args.schemes += ",sentiment"

    names = [SHORT[s.strip()] for s in args.schemes.split(",") if s.strip() in SHORT]
    print(f"\n{returns.shape[0]} months, {returns.shape[1]} securities")
    print(f"{returns.index[0]:%Y-%m} to {returns.index[-1]:%Y-%m}")
    print(f"median constituents: {int(membership.sum(axis=1).median())}\n")

    engine = WeightingBacktest(
        returns=returns, membership=membership,
        lookback=config.LOOKBACK, rebalance=config.REBALANCE,
        cost_bps=args.cost_bps,
    )
    schemes = build_schemes(mktcap, fundamentals, sentiment_panel, names)

    results, curves, conc_ts, rows = {}, {}, {}, {}
    for name, fn in schemes.items():
        print(f"running {name}...")
        try:
            res = engine.run(fn, name=name)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            continue
        results[name] = res
        r = res.returns
        curves[name] = (1 + r).cumprod().rename(name)
        conc_ts[name] = res.weights.apply(metrics.effective_n, axis=1).rename(name)
        rows[name] = res

    bench_name = "Cap weight (benchmark)"
    bench = results[bench_name].returns if bench_name in results else None

    stats = pd.DataFrame({
        n: pd.concat([performance(r.returns, bench), concentration_summary(r)])
        for n, r in rows.items()
    }).T

    # ---- exports for run_stress.py (stress testing + inference) ----
    config.RESULTS_DIR.mkdir(exist_ok=True)
    returns_wide = pd.DataFrame({n: r.returns for n, r in rows.items()})
    returns_wide.to_csv(config.RESULTS_DIR / "monthly_returns.csv")

    wrows = []
    for n, r in rows.items():
        w = r.weights.iloc[-1].dropna()
        w = w[w > 0]
        for pid, wt in w.items():
            wrows.append({"scheme": n, "id": pid, "weight": float(wt)})
    pd.DataFrame(wrows).to_csv(config.RESULTS_DIR / "latest_weights.csv", index=False)
    print(f"wrote {config.RESULTS_DIR / 'monthly_returns.csv'} and latest_weights.csv")

    sweeps = {}
    if args.sweeps:
        print("\nrunning cost sweep...")
        cost_rows = {}
        for c in config.COST_LEVELS:
            eng = WeightingBacktest(returns=returns, membership=membership,
                                    lookback=config.LOOKBACK,
                                    rebalance=config.REBALANCE, cost_bps=c)
            for name, fn in schemes.items():
                try:
                    cost_rows[f"{name} @ {c:.0f}bps"] = performance(
                        eng.run(fn, name=name).returns, bench
                    )
                except Exception:
                    continue
        sweeps["Cost sensitivity"] = pd.DataFrame(cost_rows).T

        print("running position cap sweep...")
        cap_rows = {}
        for lvl in config.CAP_LEVELS:
            fn = build_schemes(mktcap, fundamentals, sentiment_panel,
                               ["Minimum variance"], cap=lvl)["Minimum variance"]
            try:
                res = engine.run(fn, name=f"Min var cap {lvl:.0%}")
                cap_rows[f"Minimum variance, cap {lvl:.0%}"] = pd.concat(
                    [performance(res.returns, bench), concentration_summary(res)]
                )
            except Exception:
                continue
        sweeps["Position cap robustness"] = pd.DataFrame(cap_rows).T

    meta = {
        "Data source": "Synthetic" if args.offline else "CRSP monthly, Compustat annual, via WRDS",
        "Identifier": "PERMNO. Tickers carried for display only.",
        "Membership": ("Synthetic" if args.offline else
                       ("Project reconstruction, mapped to PERMNO via crsp.msenames"
                        if args.membership == "reconstruction"
                        else "Vendor index history, point in time")),
        "Delisting returns": "Not applicable" if args.offline else "Merged from crsp.msedelist",
        "Sample": f"{returns.index[0]:%Y-%m} to {returns.index[-1]:%Y-%m}",
        "Median constituents": int(membership.sum(axis=1).median()),
        "Estimation window": f"{config.LOOKBACK} months trailing",
        "Rebalance": "Monthly, month end",
        "Transaction cost": f"{args.cost_bps:.0f} bps one way, on turnover",
        "Minimum variance": f"Ledoit-Wolf shrinkage, {config.MIN_VAR_CAP:.0%} position cap",
        "Momentum": (f"top {config.MOMENTUM_TOP_QUANTILE:.0%}, "
                     f"{config.MOMENTUM_LOOKBACK}m formation, {config.MOMENTUM_SKIP}m skip"),
        "Fundamental": f"revenue, book value, cash flow, dividends; {config.REPORTING_LAG}m reporting lag",
        "Schemes run": ", ".join(rows.keys()),
    }

    write_outputs(stats, curves, conc_ts, meta, sweeps or None)


if __name__ == "__main__":
    main()
