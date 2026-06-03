"""CLI entry point for the optimal execution engine."""

from pathlib import Path

import numpy as np
import pandas as pd

from optimal_execution_engine.calibration.market_state import calibrate_market_state
from optimal_execution_engine.calibration.volume_profile import estimate_volume_profile
from optimal_execution_engine.config import ClickHouseSettings, load_settings
from optimal_execution_engine.data.bars import prepare_intraday_bars
from optimal_execution_engine.data.clickhouse import ClickHouseMarketDataClient
from optimal_execution_engine.data.loaders import load_market_data
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
from optimal_execution_engine.reporting.evaluation import summarize_experiment_batch
from optimal_execution_engine.reporting.summary import summarize_execution
from optimal_execution_engine.schedules.almgren_chriss import (
    build_almgren_chriss_schedule,
)
from optimal_execution_engine.schedules.twap import build_twap_schedule
from optimal_execution_engine.schedules.vwap import build_vwap_schedule
from optimal_execution_engine.simulator.execution import simulate_schedule
from optimal_execution_engine.types import ParentOrder


DEFAULT_ORDER_SHARES: int = 10_000
DEFAULT_HORIZON_MINUTES: int = 60
DEFAULT_RISK_AVERSION: float = 5.0
DEFAULT_SLICE_COUNT: int = 6
DEFAULT_OPENING_WINDOW_BARS: int = 12
DEFAULT_TRAIN_WINDOW_DAYS: int = 20
DEFAULT_TEST_WINDOW_DAYS: int = 5
DEFAULT_WALK_FORWARD_STEP_DAYS: int = 5


def _build_optional_clickhouse_client(
    clickhouse_settings: ClickHouseSettings,
) -> ClickHouseMarketDataClient | None:
    """Create a ClickHouse client only when host configuration is present.

    Parameters
    ----------
    clickhouse_settings
        Validated ClickHouse settings from application config.

    Returns
    -------
    ClickHouseMarketDataClient | None
        Client instance when host is configured, else ``None``.
    """
    # Keep the default demo path database-free unless explicit host config exists.
    if not clickhouse_settings.host:
        return None
    return ClickHouseMarketDataClient(settings=clickhouse_settings)


def _build_parent_order_from_day(
    day_bars: pd.DataFrame,
    order_shares: int,
    risk_aversion: float,
) -> ParentOrder:
    """Build one deterministic demo order from the first bar of a trading day.

    Parameters
    ----------
    day_bars
        Intraday bars for one trading date.
    order_shares
        Parent-order size in shares.
    risk_aversion
        Risk-aversion coefficient used by Almgren-Chriss.

    Returns
    -------
    ParentOrder
        BUY parent order configured for the day frame.
    """
    first_bar = day_bars.iloc[0]

    # Use the first close as arrival benchmark to keep per-day comparisons aligned.
    arrival_price = float(first_bar["close"])
    symbol = str(first_bar.get("symbol", "AAPL"))

    return ParentOrder(
        symbol=symbol,
        side="BUY",
        shares=order_shares,
        arrival_price=arrival_price,
        horizon_minutes=DEFAULT_HORIZON_MINUTES,
        risk_aversion=risk_aversion,
    )


def _build_daily_schedule_map(
    order: ParentOrder,
    day_bars: pd.DataFrame,
    override_daily_volatility: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all schedule families for one day using consistent slice settings.

    Parameters
    ----------
    order
        Parent order to execute.
    day_bars
        Intraday bars for one trading date.
    override_daily_volatility
        Optional forecast daily volatility passed into market-state calibration.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from schedule names to slice-level schedule frames.
    """
    slice_count = min(DEFAULT_SLICE_COUNT, len(day_bars))

    # Recompute per-day calibration so each date uses only same-day information.
    market_state = calibrate_market_state(
        bars=day_bars,
        override_daily_volatility=override_daily_volatility,
    )
    volume_profile = estimate_volume_profile(bars=day_bars, bucket_column="bucket")

    return {
        "twap": build_twap_schedule(order=order, slice_count=slice_count),
        "vwap": build_vwap_schedule(
            order=order,
            volume_profile=volume_profile,
            bucket_column="bucket",
        ),
        "almgren_chriss": build_almgren_chriss_schedule(
            order=order,
            market_state=market_state,
            slice_count=slice_count,
        ),
    }


def run_batch_experiment(
    bars: pd.DataFrame,
    order_shares: int,
    risk_aversion: float,
    forecast_by_trade_date: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run all schedule families across dates and return aggregate metrics.

    Parameters
    ----------
    bars
        Prepared multi-day bar frame that includes ``trade_date`` and ``bucket``.
    order_shares
        Parent-order size used on each date.
    risk_aversion
        Risk-aversion coefficient used for daily parent orders.
    forecast_by_trade_date
        Optional mapping from trade date to forecast daily volatility override.

    Returns
    -------
    pd.DataFrame
        Cross-day summary frame from ``summarize_experiment_batch``.
    """
    experiment_rows: list[dict[str, float | str]] = []

    # Evaluate each date independently so daily market conditions drive schedules.
    for trade_date in sorted(bars["trade_date"].unique()):
        daily_bars = bars.loc[bars["trade_date"] == trade_date].reset_index(drop=True)
        order = _build_parent_order_from_day(
            day_bars=daily_bars,
            order_shares=order_shares,
            risk_aversion=risk_aversion,
        )
        predicted_sigma = None
        if forecast_by_trade_date is not None:
            predicted_sigma = forecast_by_trade_date.get(str(trade_date))

        schedule_map = _build_daily_schedule_map(
            order=order,
            day_bars=daily_bars,
            override_daily_volatility=predicted_sigma,
        )

        for schedule_name, schedule in schedule_map.items():
            simulation_frame = simulate_schedule(
                schedule=schedule,
                bars=daily_bars,
                arrival_price=order.arrival_price,
                side=order.side,
            )

            # Store mean per-slice shortfall in bps for date-matched comparison.
            experiment_rows.append(
                {
                    "trade_date": str(trade_date),
                    "schedule_name": schedule_name,
                    "cost_bps": float(simulation_frame["cost_bps"].mean()),
                }
            )

    return summarize_experiment_batch(results=pd.DataFrame(experiment_rows))


def _format_single_order_section(
    summary_by_schedule: dict[str, dict[str, float]],
) -> str:
    """Render a readable single-order comparison section for CLI output.

    Parameters
    ----------
    summary_by_schedule
        Mapping from schedule name to summary metrics.

    Returns
    -------
    str
        Multi-line text block for the single-order section.
    """
    lines: list[str] = ["Single-Order Example (BUY 10,000 shares)"]

    # Keep output order fixed so CLI snapshots remain stable across runs.
    for schedule_name in ["twap", "almgren_chriss"]:
        metrics = summary_by_schedule[schedule_name]
        lines.append(
            f"- {schedule_name}: "
            f"total_cost_dollars=${metrics['total_cost_dollars']:.2f}, "
            f"total_cost_bps={metrics['total_cost_bps']:.2f} bps, "
            f"mean_cost_bps={metrics['mean_cost_bps']:.2f} bps, "
            f"average_fill_price=${metrics['average_fill_price']:.4f}"
        )
    return "\n".join(lines)


def _format_cross_day_section(batch_summary: pd.DataFrame) -> str:
    """Render cross-day experiment metrics with explicit units and labels.

    Parameters
    ----------
    batch_summary
        Cross-day aggregated metrics from ``summarize_experiment_batch``.

    Returns
    -------
    str
        Multi-line text block for the cross-day section.
    """
    lines: list[str] = ["Cross-Day Experiment (mean cost in bps)"]

    # Show best-to-worst by mean cost so interpretation starts with the winner.
    sorted_summary = batch_summary.sort_values("mean_cost_bps").reset_index(drop=True)
    for _, row in sorted_summary.iterrows():
        lines.append(
            f"- {row['schedule_name']}: "
            f"mean={float(row['mean_cost_bps']):.2f} bps, "
            f"median={float(row['median_cost_bps']):.2f} bps, "
            f"std={float(row['std_cost_bps']):.2f} bps, "
            f"p90={float(row['p90_cost_bps']):.2f} bps, "
            f"days={int(row['evaluation_days'])}, "
            f"win_rate_vs_twap={float(row['win_rate_vs_twap']):.2f}"
        )

    return "\n".join(lines)


def _run_offline_research_summary(
    bars: pd.DataFrame,
) -> tuple[str, dict[str, float]]:
    """Build a concise offline research summary and forecast map.

    Parameters
    ----------
    bars
        Prepared intraday bars for one symbol across many dates.

    Returns
    -------
    tuple[str, dict[str, float]]
        Human-readable summary line and per-date linear-model forecasts.
    """
    modeling_frame = build_modeling_dataset_from_bars(
        bars=bars,
        opening_window_bars=DEFAULT_OPENING_WINDOW_BARS,
    )

    if len(modeling_frame) < DEFAULT_TRAIN_WINDOW_DAYS + DEFAULT_TEST_WINDOW_DAYS:
        return (
            "Offline Research Summary: insufficient rows for walk-forward windows; "
            "using lag-based fallback only.",
            {},
        )

    splits = build_walk_forward_splits(
        frame=modeling_frame,
        train_window_size=DEFAULT_TRAIN_WINDOW_DAYS,
        test_window_size=DEFAULT_TEST_WINDOW_DAYS,
        step_size=DEFAULT_WALK_FORWARD_STEP_DAYS,
    )

    if not splits:
        return (
            "Offline Research Summary: no valid walk-forward splits; "
            "using lag-based fallback only.",
            {},
        )

    all_actual: list[float] = []
    all_persistence: list[float] = []
    all_rolling: list[float] = []
    all_linear: list[float] = []
    linear_forecast_by_trade_date: dict[str, float] = {}

    for split in splits:
        split_result = evaluate_walk_forward_split(
            modeling_frame=modeling_frame,
            split=split,
            feature_columns=LINEAR_MODEL_FEATURE_COLUMNS,
        )

        all_actual.extend(split_result.actual.tolist())
        all_persistence.extend(split_result.persistence.tolist())
        all_rolling.extend(split_result.rolling.tolist())
        all_linear.extend(split_result.linear.tolist())

        for trade_date, forecast_value in zip(
            split_result.test_trade_dates,
            split_result.linear,
            strict=True,
        ):
            linear_forecast_by_trade_date[str(trade_date)] = float(forecast_value)

    actual_array = np.asarray(all_actual, dtype=float)
    persistence_array = np.asarray(all_persistence, dtype=float)
    rolling_array = np.asarray(all_rolling, dtype=float)
    linear_array = np.asarray(all_linear, dtype=float)

    persistence_mae = compute_mean_absolute_error(
        actual=actual_array,
        predicted=persistence_array,
    )
    rolling_mae = compute_mean_absolute_error(
        actual=actual_array, predicted=rolling_array
    )
    linear_mae = compute_mean_absolute_error(
        actual=actual_array, predicted=linear_array
    )

    linear_rmse = compute_root_mean_squared_error(
        actual=actual_array, predicted=linear_array
    )
    linear_qlike = compute_qlike(actual=actual_array, predicted=linear_array)

    summary_line = (
        "Offline Research Summary: "
        f"splits={len(splits)}, "
        f"persistence_mae={persistence_mae:.6f}, "
        f"rolling5_mae={rolling_mae:.6f}, "
        f"linear_mae={linear_mae:.6f}, "
        f"linear_rmse={linear_rmse:.6f}, "
        f"linear_qlike={linear_qlike:.6f}"
    )
    return summary_line, linear_forecast_by_trade_date


def main() -> None:
    """Run the default demo workflow from cache to printed comparison metrics."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.toml"
    settings = load_settings(config_path=config_path)
    clickhouse_client = _build_optional_clickhouse_client(settings.clickhouse)

    # Load the tracked default dataset, optionally refreshable when DB config exists.
    bars = load_market_data(
        cache_dir=project_root / settings.cache.root_dir,
        dataset_name="sample_intraday_bars",
        clickhouse_client=clickhouse_client,
    )
    prepared_bars = prepare_intraday_bars(raw_bars=bars)
    research_summary, forecast_by_trade_date = _run_offline_research_summary(
        bars=prepared_bars,
    )

    order = _build_parent_order_from_day(
        day_bars=prepared_bars,
        order_shares=DEFAULT_ORDER_SHARES,
        risk_aversion=DEFAULT_RISK_AVERSION,
    )

    # Single-order section uses TWAP vs Almgren-Chriss for a focused contrast.
    first_trade_date = str(prepared_bars["trade_date"].min())
    forecast_sigma = forecast_by_trade_date.get(first_trade_date)

    market_state = calibrate_market_state(
        bars=prepared_bars,
        override_daily_volatility=forecast_sigma,
    )

    twap_schedule = build_twap_schedule(order=order, slice_count=DEFAULT_SLICE_COUNT)
    ac_schedule = build_almgren_chriss_schedule(
        order=order,
        market_state=market_state,
        slice_count=DEFAULT_SLICE_COUNT,
    )

    twap_summary = summarize_execution(
        simulate_schedule(
            schedule=twap_schedule,
            bars=prepared_bars,
            arrival_price=order.arrival_price,
            side=order.side,
        )
    )
    ac_summary = summarize_execution(
        simulate_schedule(
            schedule=ac_schedule,
            bars=prepared_bars,
            arrival_price=order.arrival_price,
            side=order.side,
        )
    )

    batch_summary = run_batch_experiment(
        bars=prepared_bars,
        order_shares=DEFAULT_ORDER_SHARES,
        risk_aversion=DEFAULT_RISK_AVERSION,
        forecast_by_trade_date=forecast_by_trade_date,
    )

    bridge_line = (
        "Execution Bridge: Almgren-Chriss uses linear-model daily volatility "
        "forecast when a same-date walk-forward prediction is available; "
        "otherwise it uses bar-estimated volatility."
    )

    single_order_output = _format_single_order_section(
        summary_by_schedule={"twap": twap_summary, "almgren_chriss": ac_summary}
    )
    cross_day_output = _format_cross_day_section(batch_summary=batch_summary)

    print(research_summary)
    print(bridge_line)
    print()
    print(single_order_output)
    print()
    print(cross_day_output)


if __name__ == "__main__":
    main()
