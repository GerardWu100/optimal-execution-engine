"""CLI formatting tests for readable interview demo output."""

import pandas as pd

from optimal_execution_engine.cli import (
    _format_cross_day_section,
    _format_single_order_section,
)


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
