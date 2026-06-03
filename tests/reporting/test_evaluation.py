"""Tests for cross-day schedule evaluation."""

import pandas as pd

from optimal_execution_engine.reporting.evaluation import summarize_experiment_batch


def test_batch_summary_reports_mean_and_tail_cost() -> None:
    """Cross-day evaluation should report central and tail metrics."""
    frame = pd.DataFrame(
        {
            "schedule_name": ["twap", "twap", "almgren_chriss", "almgren_chriss"],
            "cost_bps": [8.0, 12.0, 7.0, 11.0],
        }
    )

    summary = summarize_experiment_batch(results=frame)

    assert {"schedule_name", "mean_cost_bps", "p90_cost_bps"}.issubset(summary.columns)


def test_batch_summary_reports_richer_metrics_and_win_rate() -> None:
    """Batch summary should include dispersion, sample count, and TWAP-relative win rate."""
    frame = pd.DataFrame(
        {
            "trade_date": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-03",
                "2026-01-03",
                "2026-01-02",
                "2026-01-03",
            ],
            "schedule_name": [
                "twap",
                "almgren_chriss",
                "twap",
                "almgren_chriss",
                "vwap",
                "vwap",
            ],
            "cost_bps": [10.0, 9.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    summary = summarize_experiment_batch(results=frame)

    expected_columns = {
        "schedule_name",
        "mean_cost_bps",
        "median_cost_bps",
        "std_cost_bps",
        "p90_cost_bps",
        "evaluation_days",
        "win_rate_vs_twap",
    }
    assert expected_columns.issubset(summary.columns)

    ac_row = summary.loc[summary["schedule_name"] == "almgren_chriss"].iloc[0]
    twap_row = summary.loc[summary["schedule_name"] == "twap"].iloc[0]

    assert int(ac_row["evaluation_days"]) == 2
    assert int(twap_row["evaluation_days"]) == 2
    assert abs(float(ac_row["win_rate_vs_twap"]) - 0.5) < 1e-9
    assert abs(float(twap_row["win_rate_vs_twap"]) - 0.0) < 1e-9


def test_batch_summary_handles_missing_trade_date_without_win_rate_calc() -> None:
    """If no trade-date column exists, win rate should default safely to zero."""
    frame = pd.DataFrame(
        {
            "schedule_name": ["twap", "twap", "almgren_chriss", "almgren_chriss"],
            "cost_bps": [8.0, 12.0, 7.0, 11.0],
        }
    )

    summary = summarize_experiment_batch(results=frame)

    assert "win_rate_vs_twap" in summary.columns
    assert float(summary["win_rate_vs_twap"].sum()) == 0.0
