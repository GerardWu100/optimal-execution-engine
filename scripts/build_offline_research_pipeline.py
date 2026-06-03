"""Build the offline research teaching notebook deterministically."""

from pathlib import Path

import nbformat as nbf


def _make_cell(cell_id: str, source: str, cell_type: str) -> nbf.NotebookNode:
    """Build one notebook cell with a stable explicit ID.

    Parameters
    ----------
    cell_id
        Stable unique cell identifier.
    source
        Cell source text.
    cell_type
        Notebook cell type, either ``markdown`` or ``code``.

    Returns
    -------
    nbf.NotebookNode
        Notebook cell node with the requested ID.
    """
    if cell_type == "markdown":
        cell = nbf.v4.new_markdown_cell(source=source)
    else:
        cell = nbf.v4.new_code_cell(source=source)

    cell["id"] = cell_id
    return cell


def _markdown_cell(cell_id: str, source: str) -> nbf.NotebookNode:
    """Build one markdown cell with a stable explicit ID."""
    return _make_cell(cell_id=cell_id, source=source, cell_type="markdown")


def _code_cell(cell_id: str, source: str) -> nbf.NotebookNode:
    """Build one code cell with a stable explicit ID."""
    return _make_cell(cell_id=cell_id, source=source, cell_type="code")


def build_notebook() -> nbf.NotebookNode:
    """Assemble the offline research notebook cell-by-cell.

    Returns
    -------
    nbf.NotebookNode
        Notebook document with deterministic structure.
    """
    cells: list[nbf.NotebookNode] = [
        _markdown_cell(
            "title",
            """# Offline Research Pipeline: Raw Parquet to Volatility Forecasting

This notebook is the main teaching artifact for the repository. It starts from
tracked local Parquet files, builds realized-variance targets and features,
trains small forecasting baselines, evaluates them with walk-forward splits,
and shows one simple bridge into execution urgency.
""",
        ),
        _code_cell(
            "imports",
            """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimal_execution_engine.cli import run_batch_experiment
from optimal_execution_engine.data.bars import prepare_intraday_bars
from optimal_execution_engine.research.dataset import (
    LINEAR_MODEL_FEATURE_COLUMNS,
    build_modeling_dataset_from_bars,
)
from optimal_execution_engine.research.evaluation import (
    build_walk_forward_splits,
    compute_mean_absolute_error,
    compute_qlike,
    compute_root_mean_squared_error,
    evaluate_walk_forward_split,
)
from optimal_execution_engine.research.realized_variance import (
    compute_daily_realized_variance,
    compute_log_returns,
)


project_root = Path.cwd().resolve()
for _ in range(8):
    possible_raw = project_root / "data" / "raw"
    if possible_raw.exists():
        break
    project_root = project_root.parent

data_raw_dir = project_root / "data" / "raw"
if not data_raw_dir.exists():
    raise FileNotFoundError("Could not locate data/raw from the current working tree.")

plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.grid"] = True
""",
        ),
        _markdown_cell(
            "offline-contract",
            """## 1) Offline Contract and Raw Cache Boundary

This notebook reads only local Parquet files in `data/raw/`.
It does not create a ClickHouse client and does not require database access.
ClickHouse is only relevant when refreshing the raw cache outside notebook runtime.

The tracked datasets in this repository are opening-window-only (09:30 to 10:25
Eastern Time), so realized variance here is computed over the available intraday
window rather than full regular-hours sessions.
""",
        ),
        _code_cell(
            "load-parquet",
            """dataset_paths = sorted(data_raw_dir.glob("*.parquet"))
dataset_frames: list[pd.DataFrame] = []

for parquet_path in dataset_paths:
    frame = pd.read_parquet(parquet_path)
    frame["dataset_name"] = parquet_path.stem
    dataset_frames.append(frame)

raw_bars = pd.concat(dataset_frames, ignore_index=True)
raw_bars["ts"] = pd.to_datetime(raw_bars["ts"], utc=True)
raw_bars = raw_bars.sort_values(["symbol", "ts"]).reset_index(drop=True)

raw_bars.head()
""",
        ),
        _markdown_cell(
            "schema-coverage",
            """## 2) Schema and Session-Coverage Inspection

Before modeling, we inspect the data contract:

- rows per day,
- first and last timestamp per day,
- unique spacing in minutes,
- total days by symbol,
- file size summary from metadata.

This check protects us from hidden assumptions, especially around session
coverage and bar frequency.
""",
        ),
        _code_cell(
            "coverage-code",
            """coverage_rows: list[dict[str, object]] = []

for symbol in sorted(raw_bars["symbol"].unique()):
    symbol_frame = raw_bars.loc[raw_bars["symbol"] == symbol].copy()
    symbol_frame["trade_date"] = symbol_frame["ts"].dt.date.astype(str)

    rows_per_day = symbol_frame.groupby("trade_date").size()
    first_last = symbol_frame.groupby("trade_date")["ts"].agg(["min", "max"])
    spacing_minutes = (
        symbol_frame.sort_values("ts")["ts"].diff().dropna().dt.total_seconds().div(60.0)
    )

    coverage_rows.append(
        {
            "symbol": symbol,
            "trade_days": int(rows_per_day.size),
            "rows_per_day_min": int(rows_per_day.min()),
            "rows_per_day_max": int(rows_per_day.max()),
            "first_timestamp_utc": str(first_last["min"].iloc[0]),
            "last_timestamp_utc": str(first_last["max"].iloc[0]),
            "unique_spacing_minutes": sorted(
                [float(value) for value in spacing_minutes.round(3).unique().tolist()]
            ),
        }
    )

coverage_summary = pd.DataFrame(coverage_rows)
coverage_summary
""",
        ),
        _markdown_cell(
            "rv-construction",
            """## 3) Intraday Log Returns and Realized Variance

For each symbol and day, we compute log returns and realized variance:

- $r_t = \\log(P_t / P_{t-1})$
- $RV_{day} = \\sum_t r_t^2$

where $P_t$ is the close price at intraday bar $t$.

Because the tracked payload is opening-window-only, this realized variance is an
opening-window proxy target in the current offline dataset.
""",
        ),
        _code_cell(
            "rv-code",
            """aapl_bars = raw_bars.loc[raw_bars["symbol"] == "AAPL"].copy().reset_index(drop=True)

log_returns = compute_log_returns(bars=aapl_bars)
daily_realized_variance = compute_daily_realized_variance(bars=aapl_bars)

log_returns.head(), daily_realized_variance.head()
""",
        ),
        _markdown_cell(
            "feature-engineering",
            """## 4) Feature Engineering for Daily Volatility Forecasting

The modeling table includes:

- opening-window realized variance,
- opening return,
- opening range,
- opening volume share,
- lagged realized variance,
- rolling 5-day and 10-day realized-variance means.

Lags and rolling means are shifted so future target information never leaks into
features.
""",
        ),
        _code_cell(
            "feature-code",
            """modeling_frame = build_modeling_dataset_from_bars(
    bars=aapl_bars,
    opening_window_bars=12,
)

modeling_frame.head()
""",
        ),
        _markdown_cell(
            "walk-forward",
            """## 5) Walk-Forward Train/Test Protocol

We evaluate with rolling walk-forward windows so each test block occurs strictly
after its training history. This mirrors real forecasting conditions and avoids
lookahead bias.
""",
        ),
        _code_cell(
            "walk-forward-code",
            """splits = build_walk_forward_splits(
    frame=modeling_frame,
    train_window_size=20,
    test_window_size=5,
    step_size=5,
)

split_overview = pd.DataFrame(
    {
        "split_index": list(range(len(splits))),
        "train_size": [len(split.train_indices) for split in splits],
        "test_size": [len(split.test_indices) for split in splits],
        "first_test_date": [
            modeling_frame.iloc[split.test_indices]["trade_date"].iloc[0] for split in splits
        ],
    }
)

split_overview
""",
        ),
        _markdown_cell(
            "model-training",
            """## 6) Baselines and Linear Model Training

We compare three intentionally simple models:

1. Persistence baseline: previous-day realized variance.
2. Rolling-mean baseline: 5-day lagged mean.
3. Linear model: explicit least-squares fit on a small feature set.

These models are transparent and interview-defensible.
""",
        ),
        _code_cell(
            "model-training-code",
            """feature_columns = LINEAR_MODEL_FEATURE_COLUMNS

evaluation_rows: list[dict[str, object]] = []
linear_forecast_by_trade_date: dict[str, float] = {}

for split in splits:
    split_result = evaluate_walk_forward_split(
        modeling_frame=modeling_frame,
        split=split,
        feature_columns=feature_columns,
    )

    actual = split_result.actual
    persistence = split_result.persistence
    rolling = split_result.rolling
    linear = split_result.linear

    for trade_date, forecast_value in zip(split_result.test_trade_dates, linear, strict=True):
        linear_forecast_by_trade_date[str(trade_date)] = float(forecast_value)

    evaluation_rows.extend(
        [
            {
                "model_name": "persistence",
                "mae": compute_mean_absolute_error(actual=actual, predicted=persistence),
                "rmse": compute_root_mean_squared_error(actual=actual, predicted=persistence),
                "qlike": compute_qlike(actual=actual, predicted=persistence.clip(min=1e-12)),
            },
            {
                "model_name": "rolling_5d",
                "mae": compute_mean_absolute_error(actual=actual, predicted=rolling),
                "rmse": compute_root_mean_squared_error(actual=actual, predicted=rolling),
                "qlike": compute_qlike(actual=actual, predicted=rolling.clip(min=1e-12)),
            },
            {
                "model_name": "linear",
                "mae": compute_mean_absolute_error(actual=actual, predicted=linear),
                "rmse": compute_root_mean_squared_error(actual=actual, predicted=linear),
                "qlike": compute_qlike(actual=actual, predicted=linear),
            },
        ]
    )

evaluation_frame = pd.DataFrame(evaluation_rows)
metric_summary = (
    evaluation_frame.groupby("model_name", as_index=False)[["mae", "rmse", "qlike"]]
    .mean()
    .sort_values("mae")
    .reset_index(drop=True)
)

metric_summary
""",
        ),
        _markdown_cell(
            "evaluation-plots",
            """## 7) Evaluation Tables and Plots

We visualize mean error metrics to compare models quickly. Lower MAE, RMSE, and
QLIKE are better.
""",
        ),
        _code_cell(
            "evaluation-plots-code",
            """fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

axes[0].bar(metric_summary["model_name"], metric_summary["mae"])
axes[0].set_title("Mean Absolute Error")
axes[0].set_ylabel("Variance units")

axes[1].bar(metric_summary["model_name"], metric_summary["rmse"])
axes[1].set_title("Root Mean Squared Error")
axes[1].set_ylabel("Variance units")

axes[2].bar(metric_summary["model_name"], metric_summary["qlike"])
axes[2].set_title("QLIKE")
axes[2].set_ylabel("Loss")

for axis in axes:
    axis.set_xlabel("Model")

plt.show()
""",
        ),
        _markdown_cell(
            "interpretation",
            """## 8) Interpretation

Interpretation should prioritize robustness over one lucky split:

- compare all three metrics together,
- note whether improvements are consistent,
- keep model complexity low unless gains are material.

In this dataset, the research story is about disciplined process and leakage-safe
evaluation, not about maximizing model complexity.
""",
        ),
        _code_cell(
            "interpretation-code",
            """best_model_row = metric_summary.iloc[0]

print("Best model by MAE:")
print(best_model_row.to_string())
""",
        ),
        _markdown_cell(
            "execution-bridge",
            """## 9) Optional Execution Bridge

We now pass the linear-model daily volatility forecast into Almgren-Chriss as a
small bridge from research to execution urgency.

This keeps the project compact: one forecast signal informs one execution-model
input.
""",
        ),
        _code_cell(
            "execution-bridge-code",
            """prepared_aapl = prepare_intraday_bars(raw_bars=aapl_bars)

execution_summary = run_batch_experiment(
    bars=prepared_aapl,
    order_shares=10_000,
    risk_aversion=5.0,
    forecast_by_trade_date=linear_forecast_by_trade_date,
)

execution_summary
""",
        ),
        _markdown_cell(
            "limitations-next-steps",
            """## 10) Limitations and Next Steps

Current limitations:

- tracked data is opening-window-only, not full regular-hours coverage,
- no microstructure-level impact model,
- no hyperparameter search,
- no multi-asset coupling.

Practical next steps:

1. refresh raw payload to full-session 5-minute bars,
2. re-run the same notebook and compare forecast stability,
3. test how forecast-driven urgency changes execution outcomes across symbols.
""",
        ),
        _code_cell(
            "artifact-exports",
            """output_dir = project_root / "outputs" / "offline_research"
output_dir.mkdir(parents=True, exist_ok=True)

metric_summary.to_csv(output_dir / "model_metric_summary.csv", index=False)
execution_summary.to_csv(output_dir / "execution_bridge_summary.csv", index=False)

predictions_frame = pd.DataFrame(
    {
        "trade_date": list(linear_forecast_by_trade_date.keys()),
        "predicted_daily_variance": list(linear_forecast_by_trade_date.values()),
    }
).sort_values("trade_date")
predictions_frame.to_parquet(output_dir / "linear_predictions.parquet", index=False)

figure_path = output_dir / "model_error_comparison.png"
fig.savefig(figure_path, dpi=160)

{
    "output_dir": str(output_dir),
    "csv_files": ["model_metric_summary.csv", "execution_bridge_summary.csv"],
    "parquet_files": ["linear_predictions.parquet"],
    "png_files": ["model_error_comparison.png"],
}
""",
        ),
    ]

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    return notebook


def main() -> None:
    """Write the deterministic notebook artifact to disk."""
    notebook = build_notebook()
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "notebooks" / "offline_research_pipeline.ipynb"
    nbf.write(notebook, output_path)


if __name__ == "__main__":
    main()
