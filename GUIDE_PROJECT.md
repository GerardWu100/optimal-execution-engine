# GUIDE_PROJECT.md

This guide helps developers and AI agents navigate and extend the repository
without breaking the offline raw-Parquet contract.

## Repository Map

- `src/optimal_execution_engine/`: runtime package source.
  - `data/`: raw-data boundary contracts, metadata checks, dataset specs, optional
    ClickHouse refresh adapter, loader boundary.
  - `research/`: realized-variance targets, feature engineering, modeling, and
    walk-forward evaluation.
  - `calibration/`: market-state and volume-profile calibration.
  - `schedules/`: TWAP, VWAP-style, Almgren-Chriss schedule builders.
  - `simulator/`: slice-level execution and shortfall simulation.
  - `reporting/`: single-run and cross-day summaries.
  - `cli.py`: offline research summary plus execution bridge output.
- `tests/`: subsystem tests, including `tests/research/`.
- `data/raw/`: tracked raw Parquet payload and metadata sidecars.
- `scripts/`: deterministic notebook builder script.
- `notebooks/`: generated teaching notebook.
- `outputs/`: notebook export directory.
- `docs/`: user walkthroughs, references, and implementation plans.

## Notebook Surface

- Notebook path: `notebooks/offline_research_pipeline.ipynb`
- Deterministic builder: `scripts/build_offline_research_pipeline.py`
- Notebook documentation: `notebooks/README.md`

Always regenerate notebook structure through the builder to keep deterministic
cell ordering and stable execution checks.

## Raw Data Contract Surface

- Dataset catalog: `src/optimal_execution_engine/data/datasets.py`
- Loader boundary: `src/optimal_execution_engine/data/loaders.py`
- Metadata validation/writes: `src/optimal_execution_engine/data` cache module
- Optional refresh adapter: `src/optimal_execution_engine/data/clickhouse.py`
- Human-readable boundary notes: `data/raw/README.md`

Default demos must run with `clickhouse.host = ""` and local files only.

## Research Surface

- Realized variance: `src/optimal_execution_engine/research/realized_variance.py`
- Feature construction: `src/optimal_execution_engine/research/features.py`
- Dataset assembly: `src/optimal_execution_engine/research/dataset.py`
- Modeling baselines and linear model: `src/optimal_execution_engine/research/modeling.py`
- Walk-forward splits and metrics: `src/optimal_execution_engine/research/evaluation.py`

## Execution Bridge Surface

- Volatility override hook: `src/optimal_execution_engine/calibration/market_state.py`
- Bridge orchestration: `src/optimal_execution_engine/cli.py`

The bridge is intentionally minimal: a post-cutoff variance forecast is
square-rooted into volatility before it overrides the Almgren-Chriss input. The
order arrives after the feature window, and only later bars are simulated.

## Interview Explanation Guide

When presenting this project:

1. Explain why later-window realized variance is a causal forecast target once
   the opening feature cutoff is explicit.
2. Explain why simple baselines plus one linear model improve interpretability
   and credibility.
3. Explain why walk-forward splits are required to avoid lookahead leakage.
4. Explain why variance must be square-rooted before it changes execution urgency.

## Extending the Project Safely

1. Add or update dataset specs in `datasets.py`.
2. Refresh `data/raw/` payload with matching metadata.
3. Keep notebook and CLI offline by default (`clickhouse_client=None`).
4. Add/adjust tests in `tests/data/` and `tests/research/`.
5. Re-run tests, CLI, notebook build/execute before claiming completion.
