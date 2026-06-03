# GUIDE_ROOT.md

## Part 1: Conceptual explanation

This repository is an offline volatility-research-backed execution demo.

Default runtime path on a fresh machine:

1. Read tracked raw bars from `data/raw/`.
2. Build realized-variance targets and opening-window features.
3. Train and evaluate simple models with walk-forward validation.
4. Feed forecast daily volatility into execution scheduling as one compact bridge.
5. Print concise CLI output and walk through the notebook artifact.

ClickHouse is optional and one-time only for refreshing raw Parquet files.

## Part 2: Code reference

- `README.md`: project contract, runbook, and interview framing.
- `pyproject.toml`: package metadata, dependencies, console script entrypoint.
- `config.toml`: defaults for `data/raw/` and optional ClickHouse settings.
- `src/optimal_execution_engine/`: runtime package.
  - start at `src/optimal_execution_engine/cli.py`.
- `tests/`: data, research, calibration, schedules, simulator, reporting, and CLI
  tests.
- `data/raw/`: tracked raw Parquet files and metadata sidecars.
- `scripts/`: thin runnable entrypoints (notebook builder).
- `notebooks/`: generated offline teaching notebook.
- `outputs/`: generated notebook artifacts (gitignored except README).
- `logs/`: runtime log files (gitignored).
- `docs/reference/`: SQL and extraction references.
- `docs/user/`: interview walkthrough notes.
- `GUIDE_OVERVIEW.md`: architecture summary and tradeoffs.

Recommended reading order:

1. `src/optimal_execution_engine/cli.py`
2. `src/optimal_execution_engine/data/`
3. `src/optimal_execution_engine/research/`
4. `src/optimal_execution_engine/calibration/`
5. `src/optimal_execution_engine/schedules/`
6. `src/optimal_execution_engine/simulator/execution.py`
7. `src/optimal_execution_engine/reporting/`
8. `tests/`

## Part 3: Short journal

- 2026-04-20: Updated root guide for the offline research-first architecture,
  namespace package layout, and `data/raw/` contract.
