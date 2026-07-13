"""Tests for realized-variance target construction utilities."""

import numpy as np
import pandas as pd

from optimal_execution_engine.research.realized_variance import (
    compute_daily_realized_variance,
    compute_log_returns,
    compute_opening_window_realized_variance,
    compute_remaining_window_realized_variance,
)


def _sample_intraday_bars() -> pd.DataFrame:
    """Build a deterministic two-day intraday price fixture."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "ts": [
                "2026-01-02T14:30:00Z",
                "2026-01-02T14:35:00Z",
                "2026-01-02T14:40:00Z",
                "2026-01-03T14:30:00Z",
                "2026-01-03T14:35:00Z",
                "2026-01-03T14:40:00Z",
            ],
            "open": [100.0, 101.0, 102.0, 103.0, 102.0, 101.0],
            "high": [101.0, 102.0, 103.0, 103.5, 102.2, 101.2],
            "low": [99.8, 100.8, 101.8, 102.8, 101.8, 100.8],
            "close": [100.0, 101.0, 102.0, 103.0, 102.0, 101.0],
            "volume": [1000, 1100, 1200, 1300, 1250, 1150],
        }
    )


def test_compute_log_returns_matches_expected_values() -> None:
    """Log-return helper should compute per-day close-to-close log returns."""
    bars = _sample_intraday_bars()

    result = compute_log_returns(bars=bars)

    day_one_returns = result.loc[result["trade_date"] == "2026-01-02", "log_return"]
    expected_day_one = [np.nan, np.log(101.0 / 100.0), np.log(102.0 / 101.0)]

    assert len(day_one_returns) == 3
    assert np.isnan(day_one_returns.iloc[0])
    assert np.isclose(day_one_returns.iloc[1], expected_day_one[1])
    assert np.isclose(day_one_returns.iloc[2], expected_day_one[2])


def test_compute_daily_realized_variance_sums_squared_returns() -> None:
    """Daily realized variance should equal the sum of squared intraday returns."""
    bars = _sample_intraday_bars()

    realized_variance = compute_daily_realized_variance(bars=bars)

    day_one = realized_variance.loc[
        realized_variance["trade_date"] == "2026-01-02", "target_realized_variance"
    ].iloc[0]
    expected_day_one = np.log(101.0 / 100.0) ** 2 + np.log(102.0 / 101.0) ** 2

    assert np.isclose(day_one, expected_day_one)


def test_compute_opening_window_realized_variance_respects_window_size() -> None:
    """Opening-window realized variance should only use returns inside the window."""
    bars = _sample_intraday_bars()

    opening_variance = compute_opening_window_realized_variance(
        bars=bars,
        opening_window_bars=2,
    )

    day_one = opening_variance.loc[
        opening_variance["trade_date"] == "2026-01-02", "opening_realized_variance"
    ].iloc[0]
    expected_day_one = np.log(101.0 / 100.0) ** 2

    assert np.isclose(day_one, expected_day_one)


def test_remaining_window_target_starts_after_information_cutoff() -> None:
    """The target should contain only returns ending after the opening window."""
    bars = _sample_intraday_bars()

    remaining_variance = compute_remaining_window_realized_variance(
        bars=bars,
        opening_window_bars=2,
    )

    day_one = remaining_variance.loc[
        remaining_variance["trade_date"] == "2026-01-02",
        "target_remaining_realized_variance",
    ].iloc[0]

    assert np.isclose(day_one, np.log(102.0 / 101.0) ** 2)


def test_remaining_window_target_rejects_sessions_without_future_bar() -> None:
    """Every modeled session must extend beyond the information cutoff."""
    bars = _sample_intraday_bars()

    with np.testing.assert_raises_regex(ValueError, "bar after"):
        compute_remaining_window_realized_variance(
            bars=bars,
            opening_window_bars=3,
        )


def test_realized_variance_is_zero_for_flat_prices() -> None:
    """Flat intraday prices should produce zero realized variance."""
    bars = _sample_intraday_bars()
    bars["close"] = 100.0

    realized_variance = compute_daily_realized_variance(bars=bars)

    assert float(realized_variance["target_realized_variance"].sum()) == 0.0


def test_short_days_are_handled_deterministically_with_zero_variance() -> None:
    """A one-bar day should be retained with zero realized variance."""
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "ts": [
                "2026-01-02T14:30:00Z",
                "2026-01-02T14:35:00Z",
                "2026-01-03T14:30:00Z",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        }
    )

    realized_variance = compute_daily_realized_variance(bars=bars)
    short_day_value = realized_variance.loc[
        realized_variance["trade_date"] == "2026-01-03", "target_realized_variance"
    ].iloc[0]

    assert short_day_value == 0.0
