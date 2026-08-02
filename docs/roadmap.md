# Roadmap

## Status

| Component | State |
|---|---|
| Point-in-time membership reconstruction | Complete, validated |
| WRDS reference validation | Passed, residual documented |
| Backtest harness with plug-in weighting | Complete |
| Concentration metrics | Complete, tested |
| Equal weight | Complete |
| Minimum variance (Ledoit-Wolf, capped) | Complete |
| Risk parity | Complete |
| Momentum | Complete |
| Fundamental | Complete |
| FinBERT sentiment | Interface complete, scoring pipeline in progress |
| Results and figures | Complete |
| Written thesis | In progress |

## Remaining Work


1. **Cost sensitivity.** Sweep the cost assumption and report the level at which each scheme's diversification gain is offset. Alternative schemes trade smaller names, so a flat rate probably flatters them.
2. **Subperiod analysis.** Split the sample and test whether conclusions hold in both halves. Concentration rose over the sample, so results driven by the late period should be identified as such.
3. **Robustness on the position cap.** Report minimum variance results across several cap levels rather than one, since the cap is the most consequential parameter choice.

## Timeline

Defense is scheduled for August 2026. Items 1 through 4 are required for the defense. Items 5 and 6 strengthen the robustness section and are the first things to cut if time runs short.
