# Adaptive outline: When a perfect volatility forecast is a warning

## Archetype decision

- Problem: explain an offline realized-variance research pipeline and the small bridge that feeds its forecast into an execution schedule.
- Options considered: `strategy-backtest`, `risk-model`, and `mixed`.
- Choice: `mixed` (`risk-model` plus execution experiment).
- Why: most of the code builds and evaluates a daily realized-variance forecast; the forecast then supplies the volatility input to a simplified Almgren-Chriss schedule and is compared with time-weighted and volume-weighted baselines.
- Verify during drafting: reproduce the walk-forward metrics, prove whether the opening-window feature is distinct from the target, and trace the forecast's units into the schedule.

## Proposed sections

1. **A suspiciously good forecast**
   - Open with the linear model's mean absolute error of about `1.99e-14` variance units.
   - State the practical question: model skill or data-contract artifact?
2. **The pipeline on paper**
   - Raw 5-minute bars, log returns, realized variance, opening features, lagged features, walk-forward splits, and three forecast models.
   - Explain the tracked sample: AAPL, 55 days, 12 bars per day, 09:30 to 10:25 Eastern Time.
3. **Why the walk-forward test still leaks**
   - Show that `opening_window_bars=12` while every tracked day contains 12 bars.
   - Demonstrate that `opening_realized_variance` equals `target_realized_variance` for every modeling row.
   - Separate chronological leakage prevention from contemporaneous target leakage.
4. **What the honest benchmark says**
   - Compare persistence, rolling five-day mean, and linear-model metrics in scientific notation.
   - Treat the linear result as a diagnostic, not as forecasting evidence.
5. **From variance to execution urgency**
   - Explain the project schedule and simulator.
   - Trace the unit mismatch: the model forecasts variance, while the execution calibration parameter is documented as volatility.
   - Show that Almgren-Chriss and time-weighted average price costs are almost identical in the demo, while volume-weighted average price is costlier under this simplified impact model.
6. **What would make the experiment defensible**
   - Full-session bars, an opening feature window shorter than the target horizon, square-root conversion from variance to volatility, a dimensionally consistent Almgren-Chriss calibration, and sensitivity tests.

## Planned equations

1. Intraday log return: $r_{d,t}=\log(P_{d,t}/P_{d,t-1})$, defining day $d$, bar $t$, and close price $P$.
2. Daily realized variance: $RV_d=\sum_t r_{d,t}^2$.
3. Opening-window feature: $RV_d^{(m)}=\sum_{t=1}^{m}r_{d,t}^2$, defining opening length $m$.
4. Linear forecast: $\widehat{RV}_d=\beta_0+\sum_j\beta_j x_{d,j}$, defining coefficients $\beta$ and features $x$.
5. Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and QLIKE loss with all symbols defined.
6. Unit conversion: $\widehat{\sigma}_d=\sqrt{\widehat{RV}_d}$.
7. Simplified execution urgency: $u=\lambda\sigma$ and inventory path $x(t)=\sinh(u(1-t))/\sinh(u)$, defining risk aversion $\lambda$ and normalized time $t$.
8. Simulator impact: $I=2+25q/V$ basis points, defining slice quantity $q$ and bar volume $V$.

## Planned code excerpts

- The shifted lag/rolling construction from `research/features.py` to show what the implementation protects correctly.
- A short equality audit that compares the opening realized-variance feature with the target.
- The volatility override and urgency lines from `calibration/market_state.py` and `schedules/almgren_chriss.py` to show the units crossing the module boundary.

## Planned visual evidence

1. **Feature-target identity audit**: scatter plot of opening realized variance against target realized variance with a 45-degree line. Takeaway: all 45 modeling rows lie on the identity line.
2. **Forecast-error comparison**: logarithmic bar chart of Mean Absolute Error and Root Mean Squared Error for the three models. Takeaway: the linear model's extreme result is consistent with direct target reconstruction, not credible out-of-sample forecasting skill.
3. **Execution-cost comparison**: mean cost with dispersion for the three schedules across 55 days. Takeaway: Almgren-Chriss differs from time-weighted average price by roughly `0.00022` basis points in this setup; volume-weighted average price is materially higher because the simulator penalizes participation rate.

The generated raster cover will depict a trading order being divided across a glowing time grid, with one risk curve bending the inventory path. It will contain no labels, logos, or fake chart values.

## Known gaps and assumptions

- The tracked market data are synthetic-looking sample bars and cover only the opening hour. The post will call them tracked demo data, not live or production data.
- The project does not cite the original Almgren-Chriss paper, and its schedule is a simplified hyperbolic-sine implementation rather than a calibrated production model.
- The CLI rounds tiny variance errors to six decimals, so the article will use recomputed scientific-notation metrics frozen under `blog/data/`.
- Notebook execution currently fails because its stored kernel points to another project's deleted virtual environment. The blog analysis will run through a project-local chart script instead and record this limitation.
- No website files will be copied, built, committed, or pushed. The canonical and final package remains under `optimal-execution-engine/blog/` by explicit user instruction.

## Review decision

The outline is approved for drafting because it covers the objective, methodology, evidence, practical interpretation, and limitations without treating the near-perfect metric as a valid performance claim. The article's central argument will be the audit itself: chronological walk-forward validation cannot repair a feature that already contains the same-day target.
