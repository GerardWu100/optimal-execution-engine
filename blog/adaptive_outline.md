# Adaptive outline: repairing a causal variance-to-execution pipeline

## Archetype decision

- Problem: explain and validate a realized-variance forecast whose output changes an execution schedule.
- Options: `risk-model`, `strategy-backtest`, and `mixed`.
- Choice: `mixed` (`risk-model` plus execution experiment).
- Why: the first half constructs and evaluates a variance forecast; the second traces that forecast through an Almgren-Chriss-inspired schedule and cost simulator.
- Verify: reproduce non-overlapping feature/target windows, confirm variance-to-volatility conversion, and recompute post-cutoff execution costs.

## Section blueprint

1. The decision timeline and non-overlapping realized-variance sums.
2. Features observable at the 09:55 cutoff.
3. Walk-forward models, loss definitions, and corrected results.
4. Dimensional conversion from variance to volatility.
5. Post-cutoff schedule, simulator math, and implementation shortfall.
6. Results, limitations, and primary references.

## Planned equations and code

- Log return and separate opening/remaining realized-variance sums.
- Lagged and rolling remaining-variance features.
- Ordinary least-squares objective.
- Mean Absolute Error, Root Mean Squared Error, and QLIKE.
- $\widehat{\sigma}=\sqrt{\widehat{RV}}$ with a numerical unit example.
- Simplified Almgren-Chriss urgency and inventory path.
- Participation-rate impact and arrival-price implementation shortfall.
- Code excerpts for the target cutoff, shifted features, and square-root bridge.

## Visual evidence

1. Opening variance versus later-window variance. It proves the exact identity is gone while exposing synthetic collinearity.
2. Walk-forward errors on a logarithmic scale. It shows the fixture remains unrealistically easy after the causal fix.
3. Mean post-cutoff execution cost with dispersion. It compares only dates with out-of-sample forecasts.

The existing generated cover is retained because it is attractive, technically relevant, and free of fake values or logos.

## Gaps and assumptions

- The tracked data cover only 09:30 to 10:25, not a full session.
- Deterministic fixture construction produces near-perfect feature collinearity.
- The simplified schedule is inspired by Almgren-Chriss but is not a full calibration of their model.
- The VWAP schedule uses realized future volume and is an oracle benchmark.
- The impact model has chosen constants and no order-book mechanics.
- Canonical files remain under this project's `blog/` directory. Nothing is copied to or built in the website repository.

## Review decision

The revised outline is approved for the correction pass. It covers causal timing, math, implementation, finance interpretation, corrected evidence, limitations, and authoritative references without presenting fixture behavior as market skill.
