"""Tests for the ClickHouse extraction adapter."""

import pandas as pd

from optimal_execution_engine.data.clickhouse import (
    build_stock_bars_query,
    summarize_timestamp_coverage,
)


def test_build_stock_bars_query_targets_firstrate_stocks() -> None:
    """The extraction query should target the chosen source table."""
    query = build_stock_bars_query(
        symbol="AAPL",
        start_date="2025-12-15",
        end_date="2026-01-05",
        session_filter="regular-hours",
    )

    assert "FROM firstrate.stocks" in query
    assert "symbol = 'AAPL'" in query
    assert "2025-12-15" in query
    assert "2026-01-05" in query
    assert "toTime(ts) >= toTime('14:30:00')" in query
    assert "toTime(ts) <= toTime('20:00:00')" in query


def test_build_stock_bars_query_supports_opening_window_session_filter() -> None:
    """Opening-window filter should cap extraction at 10:25 Eastern Time."""
    query = build_stock_bars_query(
        symbol="AAPL",
        start_date="2025-12-15",
        end_date="2026-01-05",
        session_filter="opening-window-only",
    )

    assert "toTime(ts) >= toTime('14:30:00')" in query
    assert "toTime(ts) <= toTime('15:25:00')" in query


def test_timestamp_coverage_summary_reports_spacing_and_session_checks() -> None:
    """Coverage summary should expose spacing and session-window diagnostics."""
    frame = pd.DataFrame(
        {
            "ts": [
                "2026-01-02T14:30:00Z",
                "2026-01-02T14:35:00Z",
                "2026-01-03T14:30:00Z",
            ]
        }
    )

    coverage = summarize_timestamp_coverage(frame=frame)

    assert coverage["trade_dates"] == 2
    assert 5.0 in coverage["unique_bar_spacing_minutes"]
    assert coverage["regular_session_time_check_passed"] is True
