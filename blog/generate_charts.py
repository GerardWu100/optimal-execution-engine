"""Reproduce the frozen tables and charts used by the project blog post.

The script reads the tracked AAPL Parquet sample through the project's public
research and execution functions. It writes only to ``blog/data`` and
``blog/images`` so the article remains reproducible without altering product
outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimal_execution_engine.cli import run_batch_experiment
from optimal_execution_engine.config import load_settings
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


BLOG_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BLOG_DIR.parent
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "sample_intraday_bars.parquet"
CONFIG_PATH: Path = PROJECT_ROOT / "config.toml"
DATA_DIR: Path = BLOG_DIR / "data"
IMAGE_DIR: Path = BLOG_DIR / "images"

SETTINGS = load_settings(config_path=CONFIG_PATH)
OPENING_WINDOW_BARS: int = SETTINGS.research.opening_window_bars
TRAIN_WINDOW_DAYS: int = 20
TEST_WINDOW_DAYS: int = 5
STEP_DAYS: int = 5
ORDER_SHARES: int = 10_000
RISK_AVERSION: float = 5.0
BAR_DURATION_MINUTES: int = SETTINGS.execution.bar_duration_minutes
FIGURE_DPI: int = 180


def build_research_tables(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the modeling audit, per-split metrics, and forecast tables.

    Parameters
    ----------
    bars
        Tracked AAPL intraday bars with Open, High, Low, Close, Volume (OHLCV)
        columns and a timestamp column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Modeling frame with one row per day, long-form split metrics with axes
        ``split_index`` by ``model_name``, and out-of-sample forecast rows with
        one row per test date.
    """
    modeling_frame = build_modeling_dataset_from_bars(
        bars=bars,
        opening_window_bars=OPENING_WINDOW_BARS,
    )
    splits = build_walk_forward_splits(
        frame=modeling_frame,
        train_window_size=TRAIN_WINDOW_DAYS,
        test_window_size=TEST_WINDOW_DAYS,
        step_size=STEP_DAYS,
    )

    metric_rows: list[dict[str, float | int | str]] = []
    forecast_rows: list[dict[str, float | str]] = []

    # Each test block occurs after its training block, while same-day features
    # end before the remaining-window target begins.
    for split_index, split in enumerate(splits):
        result = evaluate_walk_forward_split(
            modeling_frame=modeling_frame,
            split=split,
            feature_columns=LINEAR_MODEL_FEATURE_COLUMNS,
        )
        predictions = {
            "persistence": result.persistence,
            "rolling_5d": result.rolling,
            "linear": result.linear,
        }
        for model_name, predicted in predictions.items():
            metric_rows.append(
                {
                    "split_index": split_index,
                    "model_name": model_name,
                    "mae": compute_mean_absolute_error(result.actual, predicted),
                    "rmse": compute_root_mean_squared_error(result.actual, predicted),
                    "qlike": compute_qlike(result.actual, predicted),
                }
            )

        for trade_date, actual, predicted in zip(
            result.test_trade_dates,
            result.actual,
            result.linear,
            strict=True,
        ):
            forecast_rows.append(
                {
                    "trade_date": str(trade_date),
                    "actual_remaining_variance": float(actual),
                    "linear_predicted_remaining_variance": float(predicted),
                }
            )

    return (
        modeling_frame,
        pd.DataFrame(metric_rows),
        pd.DataFrame(forecast_rows),
    )


def plot_feature_target_audit(modeling_frame: pd.DataFrame) -> None:
    """Plot opening variance against the later, non-overlapping target.

    Parameters
    ----------
    modeling_frame
        Daily model table containing ``opening_realized_variance`` and
        ``target_remaining_realized_variance`` in decimal variance units.

    Returns
    -------
    None
        Writes ``01_feature_target_partition.png`` under ``blog/images``.
    """
    opening = modeling_frame["opening_realized_variance"].to_numpy(dtype=float)
    target = modeling_frame["target_remaining_realized_variance"].to_numpy(dtype=float)
    lower = float(min(opening.min(), target.min()))
    upper = float(max(opening.max(), target.max()))

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.scatter(opening, target, color="#147d92", s=48, alpha=0.75, label="Daily rows")
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="#d07a25",
        linewidth=2,
        label="Equality reference",
    )
    axis.set_title("Opening and later variance are distinct but highly collinear")
    axis.set_xlabel("Opening realized variance (variance units)")
    axis.set_ylabel("10:00-10:25 realized variance (variance units)")
    axis.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(IMAGE_DIR / "01_feature_target_partition.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_model_errors(metric_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate and plot Mean Absolute Error and Root Mean Squared Error.

    Parameters
    ----------
    metric_frame
        Long-form rows with split, model, Mean Absolute Error (MAE), Root Mean
        Squared Error (RMSE), and QLIKE values.

    Returns
    -------
    pd.DataFrame
        One row per model with mean metrics across the five walk-forward splits.
    """
    summary = (
        metric_frame.groupby("model_name", as_index=False)[["mae", "rmse", "qlike"]]
        .mean()
        .sort_values("mae")
        .reset_index(drop=True)
    )
    labels = {
        "linear": "Linear",
        "persistence": "Persistence",
        "rolling_5d": "Rolling 5-day",
    }
    display_labels = [labels[name] for name in summary["model_name"]]
    x_positions = np.arange(len(summary))
    bar_width = 0.36

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(
        x_positions - bar_width / 2,
        summary["mae"],
        bar_width,
        label="MAE",
        color="#147d92",
    )
    axis.bar(
        x_positions + bar_width / 2,
        summary["rmse"],
        bar_width,
        label="RMSE",
        color="#d07a25",
    )
    axis.set_yscale("log")
    axis.set_xticks(x_positions, display_labels)
    axis.set_title("Synthetic smoothness still produces a near-perfect linear fit")
    axis.set_xlabel("Forecast model")
    axis.set_ylabel("Error (variance units, logarithmic scale)")
    axis.grid(axis="y", alpha=0.2, which="both")
    axis.legend(frameon=False)
    fig.savefig(IMAGE_DIR / "02_model_error_comparison.png", dpi=FIGURE_DPI)
    plt.close(fig)
    return summary


def plot_execution_costs(execution_summary: pd.DataFrame) -> None:
    """Plot mean simulated execution cost and its cross-day dispersion.

    Parameters
    ----------
    execution_summary
        One row per schedule with mean and standard deviation of simulated cost,
        both measured in basis points.

    Returns
    -------
    None
        Writes ``03_execution_cost_comparison.png`` under ``blog/images``.
    """
    labels = {
        "almgren_chriss": "Almgren-Chriss",
        "twap": "TWAP",
        "vwap": "VWAP-style",
    }
    ordered = execution_summary.set_index("schedule_name").loc[
        ["almgren_chriss", "twap", "vwap"]
    ]
    display_labels = [labels[name] for name in ordered.index]

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.bar(
        display_labels,
        ordered["mean_cost_bps"],
        yerr=ordered["std_cost_bps"],
        capsize=6,
        color=["#147d92", "#638596", "#d07a25"],
    )
    axis.bar_label(bars, fmt="%.3f bps", padding=4)
    axis.set_ylim(0.0, float(ordered["mean_cost_bps"].max()) * 1.18)
    axis.set_title("Post-cutoff execution cost across 25 forecast days")
    axis.set_xlabel("Schedule")
    axis.set_ylabel("Mean cost (basis points)")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(IMAGE_DIR / "03_execution_cost_comparison.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    """Generate all frozen blog evidence from the tracked offline sample.

    Returns
    -------
    None
        Writes comma-separated value files under ``blog/data`` and Portable
        Network Graphics figures under ``blog/images``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    raw_bars = pd.read_parquet(RAW_DATA_PATH)
    raw_bars["ts"] = pd.to_datetime(raw_bars["ts"], utc=True)
    modeling_frame, split_metrics, forecast_frame = build_research_tables(raw_bars)

    # Freeze the exact rows needed to substantiate the article without copying
    # the project's raw source payload into the blog workspace.
    audit_columns = [
        "symbol",
        "trade_date",
        "opening_realized_variance",
        "target_remaining_realized_variance",
    ]
    modeling_frame[audit_columns].to_csv(
        DATA_DIR / "feature_target_audit.csv",
        index=False,
    )
    split_metrics.to_csv(DATA_DIR / "walk_forward_split_metrics.csv", index=False)
    forecast_frame.to_csv(DATA_DIR / "linear_forecasts.csv", index=False)

    plot_feature_target_audit(modeling_frame)
    metric_summary = plot_model_errors(split_metrics)
    metric_summary.to_csv(DATA_DIR / "model_metric_summary.csv", index=False)

    forecast_variance_by_trade_date = dict(
        zip(
            forecast_frame["trade_date"],
            forecast_frame["linear_predicted_remaining_variance"],
            strict=True,
        )
    )
    prepared_bars = prepare_intraday_bars(raw_bars)
    execution_summary = run_batch_experiment(
        bars=prepared_bars,
        order_shares=ORDER_SHARES,
        risk_aversion=RISK_AVERSION,
        forecast_variance_by_trade_date=forecast_variance_by_trade_date,
        opening_window_bars=OPENING_WINDOW_BARS,
        bar_duration_minutes=BAR_DURATION_MINUTES,
    )
    execution_summary.to_csv(DATA_DIR / "execution_summary.csv", index=False)
    plot_execution_costs(execution_summary)

    minimum_partition_gap = float(
        np.abs(
            modeling_frame["opening_realized_variance"]
            - modeling_frame["target_remaining_realized_variance"]
        ).min()
    )
    opening_target_correlation = float(
        modeling_frame["opening_realized_variance"].corr(
            modeling_frame["target_remaining_realized_variance"]
        )
    )
    print(f"modeling_rows={len(modeling_frame)}")
    print(f"walk_forward_splits={split_metrics['split_index'].nunique()}")
    print(f"minimum_feature_target_gap={minimum_partition_gap:.3e}")
    print(f"opening_target_correlation={opening_target_correlation:.9f}")
    print(metric_summary.to_string(index=False))
    print(execution_summary.to_string(index=False))


if __name__ == "__main__":
    main()
