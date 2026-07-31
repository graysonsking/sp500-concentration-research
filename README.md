# S&P 500 Concentration Risk and Alternative Weighting

M.S. capstone research. Tests whether alternative weighting methodologies can restore diversification to the S&P 500 as market cap concentration has risen.

## Research Question

Cap-weighted index concentration has increased sharply, with the top ten names accounting for a growing share of total index weight. This raises a practical question for allocators: does a passive cap-weighted position still deliver the diversification it is assumed to provide, and can alternative weighting schemes recover it without giving up too much return?

The study evaluates six weighting methodologies against the cap-weighted benchmark over 2000 to 2025.

## Methodologies Tested

| Scheme | Construction |
|---|---|
| Cap weight (benchmark) | Float-adjusted market capitalization |
| Equal weight | 1/N across all constituents |
| Minimum variance | Quadratic optimization on a Ledoit-Wolf shrunk covariance matrix |
| Risk parity | Equal marginal risk contribution |
| Fundamental | Weights from accounting measures rather than price |
| Momentum | Trailing return ranking with a skip month |
| AI sentiment | FinBERT sentiment scored on SEC EDGAR filings |

## Data

- **Returns and prices:** WRDS / CRSP
- **Fundamentals:** Compustat
- **Index membership:** Point-in-time constituent list reconstructed from published index change history
- **Text:** SEC EDGAR filings, parsed and scored with FinBERT

Point-in-time membership is the piece that most retail backtests get wrong. Using today's constituent list over a historical window builds in survivorship bias and inflates measured returns. The reconstruction step here rebuilds the roster as of each rebalance date.

## Validation

The reconstructed universe was validated against a WRDS reference series. Tracking difference against the benchmark was reduced across successive iterations to within a documented tolerance. Validation output is in `validation/`.

## Repository Layout

```
sp500-concentration-research/
|
|-- README.md
|-- LICENSE
|-- .gitignore
|-- requirements.txt
|
|-- src/
|   |-- __init__.py
|   |-- benchmark.py
|   |-- membership.py
|   |-- metrics.py
|   `-- weighting.py
|
|-- strategies/
|   |-- __init__.py
|   |-- fundamental_weight.py
|   |-- momentum_weight.py
|   |-- sentiment_weight.py
|   `-- variance_weight.py
|
|-- docs/
|   |-- methodology.md
|   `-- roadmap.md
|
|-- results/
|   `-- .gitkeep
|
|-- tests/
|   |-- __init__.py
|   |-- test_membership.py
|   `-- test_metrics.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

WRDS credentials are read from the environment and are never committed:

```bash
export WRDS_USERNAME=your_username
```

## Metrics Reported

Annualized return, volatility, Sharpe ratio, maximum drawdown, turnover, effective number of constituents, Herfindahl-Hirschman index of weights, and active share against the cap-weighted benchmark.

## Tests

```bash
python -m pytest tests -q
```

14 tests covering concentration metrics and point-in-time membership reconstruction.

## Results

Populate after final runs. Do not publish figures that have not been reproduced end to end.

## Status

In development. Defense scheduled August 2026.

## License

MIT

---

*Academic research. Not investment advice.*
