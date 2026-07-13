"""Tests for slice-to-order execution summaries."""

import numpy as np
import pandas as pd

from optimal_execution_engine.reporting.summary import summarize_execution


def test_summary_uses_arrival_notional_and_share_weighted_fill() -> None:
    """Order metrics should use standard implementation-shortfall weighting."""
    simulation = pd.DataFrame(
        {
            "shares": [100, 300],
            "fill_price": [101.0, 103.0],
            "cost_dollars": [100.0, 900.0],
            "cost_bps": [100.0, 300.0],
            "arrival_notional": [10_000.0, 30_000.0],
        }
    )

    summary = summarize_execution(simulation)

    assert np.isclose(summary["total_cost_bps"], 250.0)
    assert np.isclose(summary["average_fill_price"], 102.5)
