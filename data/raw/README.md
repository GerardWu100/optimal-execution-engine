# Raw Offline Dataset Contract

This folder is the only tracked raw-data boundary for the project.

The notebook and default CLI read only these local Parquet files and metadata
sidecars. No database access is required for normal usage.

## Datasets

- `sample_intraday_bars.parquet` + `sample_intraday_bars.meta.json`
  - Symbol: `AAPL`
  - Bar frequency: `5min`
  - Session scope: opening-window-only (09:30 to 10:25 Eastern Time)
  - Intended use: modeling and demonstration
- `sample_intraday_bars_msft.parquet` + `sample_intraday_bars_msft.meta.json`
  - Symbol: `MSFT`
  - Bar frequency: `5min`
  - Session scope: opening-window-only (09:30 to 10:25 Eastern Time)
  - Intended use: modeling and demonstration
- `sample_intraday_bars_nvda.parquet` + `sample_intraday_bars_nvda.meta.json`
  - Symbol: `NVDA`
  - Bar frequency: `5min`
  - Session scope: opening-window-only (09:30 to 10:25 Eastern Time)
  - Intended use: modeling and demonstration

## Extraction Provenance

These files are extracted from `firstrate.stocks` and written as one Parquet file
per symbol with one metadata sidecar per symbol. Metadata captures:

- source table and filters,
- timestamp bounds and spacing diagnostics,
- session scope labels,
- file size and row-count checks,
- portable offline-source notes.

## Refresh Procedure

ClickHouse is optional and used only when refreshing raw files.

1. Set `CLICKHOUSE_HOST` and optional connection settings.
2. Run a refresh path that calls `load_market_data(..., clickhouse_client=...)`.
3. Verify metadata sidecars match actual timestamp coverage and session scope.

The notebook runtime never creates a ClickHouse client.

## Size Budget

- Packaging rule: one Parquet per symbol plus one metadata sidecar per symbol.
- Preferred total tracked size target: under 100 MB.
- Current tracked payload is below 1 MB and remains laptop-friendly.

## Notebook Contract

`notebooks/offline_research_pipeline.ipynb` reads only these files in `data/raw/`.
It does not instantiate `ClickHouseMarketDataClient`.
