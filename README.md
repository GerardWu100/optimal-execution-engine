# Optimal Execution Engine

Offline-first quantitative finance project that teaches a clean research backbone
for volatility forecasting and a small execution bridge.

## Goal

This repository is an interview-defensible, laptop-friendly project centered on
one teaching story:

`raw intraday bars -> realized variance -> features -> model -> walk-forward evaluation -> execution interpretation`

The notebook is the primary teaching artifact. The CLI is a compact companion
that prints one research summary and one execution interpretation.

## Project Contract

Clone the repo, run `uv sync`, and execute the notebook or CLI using only the
tracked raw Parquet files in `data/raw/`. ClickHouse is needed only if you want
to refresh the raw cache.

## Offline Data Boundary

The only tracked raw-data boundary is `data/raw/`.

Each dataset is packaged as:

- one Parquet file: `<dataset>.parquet`
- one metadata sidecar: `<dataset>.meta.json`

Current tracked datasets:

- `sample_intraday_bars` (AAPL)
- `sample_intraday_bars_msft` (MSFT)
- `sample_intraday_bars_nvda` (NVDA)

All current tracked files are opening-window-only (09:30 to 10:25 Eastern Time),
5-minute bars, 55 trading days each.

## Research Backbone

The research subpackage builds daily modeling tables from local raw Parquet.

Primary target:

- same-day realized variance proxy from intraday log returns.

Primary features:

- opening-window realized variance,
- opening return,
- opening range,
- opening volume share,
- lagged realized variance,
- rolling 5-day and 10-day realized-variance means.

Primary models:

- persistence baseline,
- rolling-mean baseline,
- explicit linear regression (least squares with NumPy).

Evaluation protocol:

- walk-forward time splits,
- Mean Absolute Error (MAE),
- Root Mean Squared Error (RMSE),
- QLIKE loss.

## Execution Bridge

The execution module stays intentionally small.

- Forecast daily volatility from the research pipeline.
- Pass it into Almgren-Chriss as `override_daily_volatility`.
- Compare resulting execution costs against TWAP and VWAP-style baselines.

This keeps execution as a consumer of research, not the entire product.

## Quickstart

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run python -m pytest -q
```

Run CLI demo (offline by default):

```bash
uv run optimal-execution
```

Build deterministic notebook:

```bash
uv run python scripts/build_offline_research_pipeline.py
```

Execute notebook top-to-bottom:

```bash
uv run python -m nbconvert --to notebook --execute notebooks/offline_research_pipeline.ipynb --output /tmp/offline_research_pipeline.executed.ipynb
```

## Optional ClickHouse Refresh

ClickHouse is optional and one-time for refreshing raw Parquet payloads.

- runtime demos and notebook execution do not require database access,
- refresh logic lives behind explicit loader/client usage,
- refreshed files must still be written back to `data/raw/`.

## Interview Framing

How to explain this quickly:

1. Start with the offline contract (`data/raw/` only).
2. Walk through target and feature construction with no leakage.
3. Show walk-forward evaluation with simple interpretable models.
4. Show how forecast volatility informs execution urgency in one small bridge.

## Repository Navigation

- high-level architecture: `GUIDE_OVERVIEW.md`
- root map and reading order: `GUIDE_ROOT.md`
- detailed project map: `GUIDE_PROJECT.md`
- package guide: `src/GUIDE_src.md`
- scripts guide: `scripts/GUIDE_scripts.md`
- docs guide: `docs/GUIDE_docs.md`
- tests guide: `tests/GUIDE_tests.md`
- notebook notes: `notebooks/README.md`
