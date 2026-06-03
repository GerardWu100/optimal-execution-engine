"""Tests for execution-cost simulation."""

import pandas as pd

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
