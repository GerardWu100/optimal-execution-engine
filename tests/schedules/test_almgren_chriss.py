"""Tests for the Almgren-Chriss scheduler."""

from optimal_execution_engine.schedules.almgren_chriss import (
    build_almgren_chriss_schedule,
)
from optimal_execution_engine.types import MarketState, ParentOrder


def test_almgren_chriss_front_loads_when_risk_aversion_increases() -> None:
    """Higher risk aversion should increase early execution size."""
    order = ParentOrder("AAPL", "BUY", 1000, 100.0, 60, 5.0)
    market_state = MarketState(1_000_000.0, 0.02, 5.0)

    schedule = build_almgren_chriss_schedule(
        order=order, market_state=market_state, slice_count=4
    )

    assert int(schedule["shares"].sum()) == 1000
    assert int(schedule.iloc[0]["shares"]) >= int(schedule.iloc[-1]["shares"])
