# GUIDE_src.md

## Part 1: Conceptual explanation

`optimal_execution_engine` implements a compact offline pipeline from raw
intraday bars to volatility-aware execution interpretation.

Package layers:

| Layer | Role | Typical output |
| --- | --- | --- |
| Data | Load and validate raw Parquet boundary | Bar DataFrame |
| Research | Build targets/features, model, evaluate | Forecast metrics and predictions |
| Calibration | Estimate market state and volume shape | Volatility, ADV, bucket weights |
| Schedules | Build TWAP, VWAP, Almgren-Chriss slices | Per-slice share schedule |
| Simulator/Reporting | Simulate fills and aggregate costs | Cost summaries |

CLI orchestration prints:

1. Offline research summary (walk-forward metrics).
2. One bridge explanation for forecast volatility in execution.
3. Single-order comparison output.
4. Cross-day experiment output.

## Part 2: Code reference

- `cli.py`: command entrypoint and research-to-execution bridge orchestration.
- `config.py`: settings models and TOML/environment loading.
- `types.py`: core dataclasses (`ParentOrder`, `MarketState`).
- `version.py`: package version.
- `data/`: loader boundary, metadata validation, dataset specs, optional refresh.
- `research/`: realized variance, features, dataset, modeling, evaluation.
- `calibration/`: market-state and volume-profile calibration.
- `schedules/`: TWAP, VWAP-style, Almgren-Chriss schedule builders.
- `simulator/`: fill and cost simulation.
- `reporting/`: single-run and batch metrics.

Recommended reading order:

1. `cli.py`
2. `research/`
3. `data/loaders.py`
4. `calibration/market_state.py`
5. `schedules/almgren_chriss.py`
6. `simulator/execution.py`
7. `reporting/evaluation.py`

## Part 3: Short journal

- 2026-04-20: Updated package guide for namespace packaging and added research
  layer to the primary architecture narrative.
