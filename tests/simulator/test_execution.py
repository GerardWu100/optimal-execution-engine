"""Tests for execution-cost simulation."""

import pandas as pd
import pytest

from optimal_execution_engine.simulator.execution import simulate_schedule


def test_simulation_returns_non_negative_cost_columns() -> None:
    """Simulation output should expose cost columns for each slice."""
    schedule = pd.DataFrame({"slice_index": [0, 1], "shares": [500, 500]})
    bars = pd.DataFrame({"close": [100.0, 100.2], "volume": [2000, 2000]})

    result = simulate_schedule(
        schedule=schedule, bars=bars, arrival_price=100.0, side="BUY"
    )

    assert {"fill_price", "cost_dollars", "cost_bps"}.issubset(result.columns)
    assert float(result["cost_dollars"].sum()) >= 0.0


def test_halted_bar_with_shares_raises() -> None:
    """A zero-volume bar the schedule trades into is rejected, not priced."""
    schedule = pd.DataFrame({"slice_index": [0, 1], "shares": [500, 500]})
    bars = pd.DataFrame({"close": [100.0, 100.2], "volume": [2000, 0]})

    with pytest.raises(ValueError, match="positive volume"):
        simulate_schedule(
            schedule=schedule, bars=bars, arrival_price=100.0, side="BUY"
        )


def test_missing_bar_volume_raises() -> None:
    """A NaN volume is treated the same as a halt rather than producing NaN costs."""
    schedule = pd.DataFrame({"slice_index": [0, 1], "shares": [500, 500]})
    bars = pd.DataFrame({"close": [100.0, 100.2], "volume": [2000, float("nan")]})

    with pytest.raises(ValueError, match="positive volume"):
        simulate_schedule(
            schedule=schedule, bars=bars, arrival_price=100.0, side="BUY"
        )


def test_zero_share_slice_in_halted_bar_costs_nothing() -> None:
    """Routing no shares into a halted bar is valid and adds no cost."""
    schedule = pd.DataFrame({"slice_index": [0, 1], "shares": [1000, 0]})
    bars = pd.DataFrame({"close": [100.0, 100.2], "volume": [2000, 0]})

    result = simulate_schedule(
        schedule=schedule, bars=bars, arrival_price=100.0, side="BUY"
    )

    assert result["cost_dollars"].notna().all()
    assert float(result.loc[1, "cost_dollars"]) == 0.0
