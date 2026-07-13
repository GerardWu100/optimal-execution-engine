"""CLI formatting tests for readable interview demo output."""

import numpy as np
import pandas as pd
from pytest import MonkeyPatch

from optimal_execution_engine.cli import (
    _format_cross_day_section,
    _format_single_order_section,
    run_batch_experiment,
)
from optimal_execution_engine.types import ParentOrder


def test_single_order_section_includes_schedule_names_and_units() -> None:
    """Single-order formatter should show explicit schedule labels and units."""
    output = _format_single_order_section(
        {
            "twap": {
                "total_cost_dollars": 100.123,
                "total_cost_bps": 10.5,
                "mean_cost_bps": 10.4,
                "average_fill_price": 101.1234,
            },
            "almgren_chriss": {
                "total_cost_dollars": 98.0,
                "total_cost_bps": 10.2,
                "mean_cost_bps": 10.1,
                "average_fill_price": 101.1100,
            },
        }
    )

    assert "Single-Order Example" in output
    assert "twap" in output
    assert "almgren_chriss" in output
    assert "bps" in output
    assert "$" in output


def test_cross_day_section_includes_richer_metric_labels() -> None:
    """Cross-day formatter should include all metric names from summary output."""
    summary = pd.DataFrame(
        {
            "schedule_name": ["twap", "vwap"],
            "mean_cost_bps": [10.0, 12.0],
            "median_cost_bps": [10.0, 12.0],
            "std_cost_bps": [1.0, 2.0],
            "p90_cost_bps": [11.0, 14.0],
            "evaluation_days": [5, 5],
            "win_rate_vs_twap": [0.0, 0.2],
        }
    )

    output = _format_cross_day_section(batch_summary=summary)

    assert "Cross-Day Experiment" in output
    assert "mean=" in output
    assert "median=" in output
    assert "std=" in output
    assert "p90=" in output
    assert "days=" in output
    assert "win_rate_vs_twap=" in output


def test_batch_experiment_converts_variance_and_starts_after_cutoff(
    monkeypatch: MonkeyPatch,
) -> None:
    """Execution should receive volatility and bars strictly after the forecast."""
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "trade_date": ["2026-01-02"] * 4,
            "bucket": [0, 1, 2, 3],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000.0] * 4,
        }
    )
    captured: dict[str, object] = {}

    def fake_schedule_map(
        order: ParentOrder,
        day_bars: pd.DataFrame,
        override_daily_volatility: float | None = None,
    ) -> dict[str, pd.DataFrame]:
        captured["arrival_price"] = order.arrival_price
        captured["execution_closes"] = day_bars["close"].tolist()
        captured["volatility"] = override_daily_volatility
        schedule = pd.DataFrame({"slice_index": [0, 1], "shares": [500, 500]})
        return {"twap": schedule}

    monkeypatch.setattr(
        "optimal_execution_engine.cli._build_daily_schedule_map",
        fake_schedule_map,
    )

    run_batch_experiment(
        bars=bars,
        order_shares=1_000,
        risk_aversion=5.0,
        forecast_variance_by_trade_date={"2026-01-02": 0.0004},
        opening_window_bars=2,
        bar_duration_minutes=5,
    )

    assert captured["arrival_price"] == 101.0
    assert captured["execution_closes"] == [102.0, 103.0]
    assert np.isclose(float(captured["volatility"]), 0.02)
