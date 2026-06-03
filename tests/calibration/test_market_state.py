"""Tests for market-state calibration from bar data."""

import pandas as pd

from optimal_execution_engine.calibration.market_state import calibrate_market_state


def test_calibration_returns_positive_volatility_and_adv() -> None:
    """Calibrated statistics should be positive for non-constant data."""
    bars = pd.DataFrame(
        {
            "close": [100.0, 100.5, 100.2, 100.9],
            "volume": [1000, 1200, 900, 1100],
        }
    )

    market_state = calibrate_market_state(bars=bars)

    assert market_state.daily_volatility > 0.0
    assert market_state.average_daily_volume > 0.0


def test_calibration_allows_forecast_volatility_override() -> None:
    """Forecast override should replace bar-estimated daily volatility."""
    bars = pd.DataFrame(
        {
            "close": [100.0, 100.5, 100.2, 100.9],
            "volume": [1000, 1200, 900, 1100],
        }
    )

    market_state = calibrate_market_state(bars=bars, override_daily_volatility=0.031)

    assert market_state.daily_volatility == 0.031
