# Optimal Execution Engine

An offline-first demo that forecasts short-horizon realized variance from
intraday bars and feeds the forecast into an Almgren-Chriss optimal-execution
schedule, to show how a research signal changes execution urgency.

## What it does

The teaching story: `opening bars -> causal features -> remaining-window
variance -> walk-forward forecast -> volatility -> later-window execution`.

- **Research backbone.** Builds daily modeling tables from 5-minute intraday
  bars. Target: remaining-window realized variance (sum of squared log
  returns) after an opening information cutoff at 09:55 Eastern Time.
  Features (all observable by the cutoff): opening-window realized variance,
  opening return, opening range, log opening volume, lagged remaining-window
  variance, and rolling 5-/10-day remaining-window variance means. Models:
  persistence baseline, rolling-mean baseline, and an explicit least-squares
  linear regression (NumPy). Evaluated with walk-forward time splits using
  MAE, RMSE, and QLIKE loss.
- **Execution bridge.** Takes the forecast variance, converts it to
  volatility with $\sqrt{\text{predicted variance}}$, and passes it into an
  [Almgren-Chriss](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
  optimal-execution schedule as `override_daily_volatility`. Compares the
  resulting implementation shortfall against a TWAP (Time-Weighted Average
  Price) schedule and an oracle VWAP (Volume-Weighted Average Price)
  benchmark. This is a simulation on historical bars, not a live or
  paper-traded result.

Data sources: three tracked sample datasets in `data/raw/` (AAPL, MSFT,
NVDA; 5-minute bars, 09:30-10:25 ET, 55 trading days each) — see
[`data/raw/README.md`](data/raw/README.md). ClickHouse is optional and used
only to refresh those Parquet files; normal runs never touch a database.

## Requirements

- Python 3.13
- No external service for normal use — the notebook and CLI run entirely
  off the tracked Parquet files in `data/raw/`.
- Optional, only for refreshing raw data: a ClickHouse server, configured
  through `.env` (`CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`,
  `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY`).

## Setup

```bash
uv sync
```

## Usage

```bash
uv run optimal-execution                                          # CLI: research summary + execution interpretation
uv run python -m pytest -q                                        # run tests
uv run python scripts/build_offline_research_pipeline.py          # (re)build the deterministic notebook
uv run python -m nbconvert --to notebook --execute \
  notebooks/offline_research_pipeline.ipynb \
  --output /tmp/offline_research_pipeline.executed.ipynb          # execute the notebook top-to-bottom
```

The notebook (`notebooks/offline_research_pipeline.ipynb`) is the primary
walkthrough; the CLI is a compact companion that prints one research summary
and one execution interpretation.

## Configuration

`config.toml`: `[cache].root_dir` (raw Parquet location),
`[research].opening_window_bars` (bars observed before forecasting),
`[execution].bar_duration_minutes`, and `[clickhouse]` (optional refresh
connection — leave `host` empty to stay offline).

## Layout

```text
src/optimal_execution_engine/data/         raw-data boundary, dataset specs, optional ClickHouse refresh
src/optimal_execution_engine/research/     variance target, features, models, walk-forward evaluation
src/optimal_execution_engine/calibration/  market-state and volume-profile calibration
src/optimal_execution_engine/schedules/    TWAP, VWAP-style, and Almgren-Chriss schedule builders
src/optimal_execution_engine/simulator/    slice-level execution and shortfall simulation
src/optimal_execution_engine/reporting/    single-run and cross-day summaries
data/raw/                                  tracked raw Parquet payload and metadata sidecars
scripts/                                   deterministic notebook builder
notebooks/                                 the teaching notebook
```

Further reading: `GUIDE_ROOT.md` (root map), `GUIDE_OVERVIEW.md`
(architecture), `GUIDE_PROJECT.md` (detailed project map), and the
per-folder `GUIDE_*.md` files under `src/`, `scripts/`, `docs/`, and
`tests/`.

## Output

The CLI prints a research summary (forecast error metrics) and an execution
interpretation (implementation shortfall vs. TWAP/VWAP) to stdout. The
notebook writes its executed form under `outputs/`.
