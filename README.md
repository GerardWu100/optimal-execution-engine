# Optimal Execution Engine

Offline-first quantitative finance project that teaches a clean research backbone
for volatility forecasting and a small execution bridge.

## Goal

This repository is an interview-defensible, laptop-friendly project centered on
one teaching story:

`opening bars -> causal features -> remaining-window variance -> walk-forward forecast -> volatility -> later-window execution`

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

All current tracked files contain 5-minute bars from 09:30 to 10:25 Eastern Time
for 55 trading days. The demo observes the first six bars through 09:55 and
forecasts the variance of returns ending from 10:00 through 10:25.

## Research Backbone

The research subpackage builds daily modeling tables from local raw Parquet.

Primary target:

- remaining-window realized variance from squared log returns after the opening
  information cutoff.

Primary features:

- opening-window realized variance,
- opening return,
- opening range,
- log opening volume,
- lagged remaining-window variance,
- rolling 5-day and 10-day remaining-window variance means.

Every same-day feature is observable by 09:55. In particular, the feature set
does not divide opening volume by later volume that is unknown at that time.

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

- Forecast remaining-window variance from the research pipeline.
- Convert variance to volatility with `sqrt(predicted_variance)`.
- Start the order after the feature cutoff and pass the volatility into
  Almgren-Chriss as `override_daily_volatility`.
- Compare resulting implementation shortfall against Time-Weighted Average
  Price (TWAP) and an oracle Volume-Weighted Average Price (VWAP) benchmark.

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
2. Walk through the feature cutoff and later-window target with no overlap.
3. Show walk-forward evaluation with simple interpretable models.
4. Show how square-rooted variance informs post-cutoff execution urgency.

## Repository Navigation

- high-level architecture: `GUIDE_OVERVIEW.md`
- root map and reading order: `GUIDE_ROOT.md`
- detailed project map: `GUIDE_PROJECT.md`
- package guide: `src/GUIDE_src.md`
- scripts guide: `scripts/GUIDE_scripts.md`
- docs guide: `docs/GUIDE_docs.md`
- tests guide: `tests/GUIDE_tests.md`
- notebook notes: `notebooks/README.md`
