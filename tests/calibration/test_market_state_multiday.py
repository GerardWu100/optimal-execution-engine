"""Regression tests for multi-day market-state calibration behavior."""

import pandas as pd

from optimal_execution_engine.calibration.market_state import calibrate_market_state


def test_calibration_uses_average_daily_volume_across_trade_dates() -> None:
    """ADV should be the mean of per-date volume totals, not a frame-wide sum."""
    bars = pd.DataFrame(
        {
            "ts": [
                "2026-01-02T14:30:00Z",
                "2026-01-02T14:35:00Z",
                "2026-01-03T14:30:00Z",
                "2026-01-03T14:35:00Z",
            ],
            "close": [100.0, 100.2, 101.0, 101.3],
            "volume": [100.0, 300.0, 200.0, 400.0],
        }
    )

    market_state = calibrate_market_state(bars=bars)

    # day 1 volume = 400, day 2 volume = 600, ADV = mean([400, 600]) = 500
    assert market_state.average_daily_volume == 500.0
