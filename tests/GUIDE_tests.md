# GUIDE_tests.md

## Part 1: Conceptual explanation

This folder verifies the offline workflow end to end through focused unit and
narrow integration tests.

Testing strategy:

- enforce raw-data boundary contracts (`data/raw/`, metadata keys, loader errors),
- validate research math and leakage controls,
- validate scheduling and simulation invariants,
- validate CLI output sections and metric labels used in interviews.

## Part 2: Code reference

- `test_config.py`: configuration defaults and environment overrides.
- `test_cli_output.py`: CLI formatting coverage.
- `data/`
  - `test_cache.py`: metadata validation and write enrichment.
  - `test_clickhouse.py`: session-aware SQL and coverage diagnostics.
  - `test_loaders.py`: offline-first loader behavior and catalog integration.
- `research/`
  - `test_realized_variance.py`: log returns and realized-variance targets.
  - `test_features.py`: lagging/rolling features and table alignment.
  - `test_modeling.py`: baselines and linear model fit/predict.
  - `test_walk_forward_evaluation.py`: split protocol and MAE/RMSE/QLIKE.
- `calibration/`: market-state and volume-profile tests.
- `schedules/`: TWAP, VWAP-style, Almgren-Chriss tests.
- `simulator/test_execution.py`: execution-cost mechanics.
- `reporting/test_evaluation.py`: batch summary metrics and win-rate logic.

## Part 3: Short journal

- 2026-04-20: Added research test suite and updated boundary tests for
  `data/raw/` plus tightened metadata contract checks.
