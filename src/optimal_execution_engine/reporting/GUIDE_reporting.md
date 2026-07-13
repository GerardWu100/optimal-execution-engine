# GUIDE_reporting.md

## Part 1: Conceptual explanation

This folder aggregates simulator outputs into schedule-comparison metrics.

Reporting levels:

1. **Single-run summary**: one simulated schedule.
2. **Cross-day batch summary**: repeated schedule evaluations by date.

Single-run metrics:

- total dollar cost,
- total basis-point cost,
- mean per-slice basis-point cost for diagnostics,
- share-weighted average fill price.

Total basis-point cost is implementation shortfall divided by arrival-price
notional. This matches the benchmark used to calculate slice dollar costs.

Cross-day metrics:

- mean and median cost,
- standard deviation and 90th percentile,
- evaluation-day count,
- win rate versus TWAP on date-matched comparisons.

## Part 2: Code reference

- `summary.py`: `summarize_execution(simulation_frame)`.
- `evaluation.py`: `summarize_experiment_batch(results)`.
- `__init__.py`: package marker docstring.

Cross-folder usage:

- consumes simulator output from
  `src/optimal_execution_engine/simulator/execution.py`.
- called by `src/optimal_execution_engine/cli.py` and the offline teaching
  notebook.

## Part 3: Short journal

- 2026-04-20: Updated guide paths for namespace package and retained explicit
  TWAP-relative win-rate semantics for interview clarity.
- 2026-07-13: Corrected order-level cost and average-fill weighting for unequal
  schedule slices.
