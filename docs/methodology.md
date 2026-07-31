[methodology.md](https://github.com/user-attachments/files/30567989/methodology.md)
# Methodology

## Research Question

Market capitalization concentration in the S&P 500 has risen sharply. The top ten constituents account for a share of index weight that is high relative to the index's own history. This raises a question for anyone holding the index passively: does a cap weighted position still deliver the diversification it is assumed to provide, and if not, can an alternative weighting scheme recover it at acceptable cost?

The study evaluates six alternative schemes against the cap weighted benchmark over 2000 to 2025.

## Hypotheses

**H1.** Effective N of the cap weighted index has declined materially over the sample, indicating rising concentration.

**H2.** Alternative weighting schemes produce higher effective N and higher diversification ratios than cap weighting.

**H3.** The diversification gain is not free. Alternative schemes carry higher turnover and tracking error.

The interesting result is the size of the tradeoff in H3, not the direction, which is known in advance.

## Sample and Data

| Item | Source |
|---|---|
| Returns and prices | CRSP via WRDS |
| Fundamentals | Compustat via WRDS |
| Index membership | Reconstructed from published change history |
| Filing text | SEC EDGAR |
| Sample period | 2000-01 to 2025-12, monthly |

## Point-in-Time Universe

Backtesting today's constituent list over a historical window measures the performance of companies successful enough to still be in the index. That is a selection effect, not a strategy result.

`src/membership.py` reconstructs the roster by starting from the current list and walking the change history backward, undoing each addition and deletion. Two checks apply.

**Count stability.** Constituent count should stay near 500 across the sample. Drift indicates a missing or misparsed change event. `count_series` produces this check.

**Reference validation.** The reconstruction is compared against a WRDS reference series. Per-date match rates, false inclusions, and omissions are reported in `validation/`. The reconciliation was iterated until the residual tracking difference against the benchmark fell within a documented tolerance.

A reconstruction nobody has checked is not evidence. The validation output is published alongside the results.

## Weighting Schemes

| Scheme | Inputs | Estimation risk |
|---|---|---|
| Cap weight (benchmark) | Float adjusted market cap | None |
| Equal weight | None | None |
| Minimum variance | Covariance | Moderate |
| Risk parity | Covariance | Moderate |
| Fundamental | Accounting measures | Low |
| Momentum | Trailing returns | Low |
| Sentiment | FinBERT on filings | High |

**Position caps.** Minimum variance uses a 5 percent cap. This is the single most consequential methodological choice in the study. Unconstrained minimum variance on a 500 name universe concentrates into a small set of low volatility names, which would make the scheme fail the diversification test for reasons unrelated to the research question. The cap is set ex ante and reported.

**Covariance estimation.** Ledoit-Wolf shrinkage is used throughout. With 500 assets and a 60 month window, the sample covariance matrix is singular. Shrinkage is not an optimization, it is a requirement for the problem to be solvable at all.

**Reporting lags.** Fundamentals are lagged three months for reporting delay. Filings are timestamped by acceptance date, not by the period they cover.

## Measurement

Diversification is measured four ways because no single measure is sufficient.

**Effective N** (inverse Herfindahl) is the headline. It states how many positions the portfolio effectively holds rather than how many it nominally holds.

**Diversification ratio** captures what effective N misses. A portfolio can hold many names and still be undiversified if those names are highly correlated. This measure is weighted average constituent volatility divided by realized portfolio volatility.

**Gini and entropy** describe the shape of the weight distribution.

**Active share and tracking error** quantify the cost side: how far a scheme departs from the benchmark and how much that departure varies.

Performance statistics (CAGR, volatility, Sharpe, max drawdown, turnover) are reported alongside so the tradeoff is visible rather than asserted.

## Known Limitations

- Reconstructed membership may contain small errors around index events such as spinoffs, mergers, and share class changes. Residual error is quantified rather than assumed to be zero.
- Transaction costs are modeled as a flat rate on turnover. Real costs vary by name and by market condition, and the alternative schemes trade smaller names than the benchmark does, so a flat rate likely understates their cost.
- FinBERT calibration on filing language differs from its training distribution. Sentiment results should be read as exploratory.
- The sample covers one path through history and includes two large drawdowns. Conclusions about drawdown behavior rest on few independent observations.
- Results are gross of tax and assume full investability at index weights.
