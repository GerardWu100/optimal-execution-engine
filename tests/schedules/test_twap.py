"""Tests for the TWAP scheduler."""

from optimal_execution_engine.schedules.twap import build_twap_schedule
from optimal_execution_engine.types import ParentOrder


def test_twap_schedule_sums_to_parent_order() -> None:
    """TWAP slices should add back to the full order size."""
    order = ParentOrder("AAPL", "BUY", 1000, 100.0, 60, 1.0)

    schedule = build_twap_schedule(order=order, slice_count=4)

    assert int(schedule["shares"].sum()) == 1000
    assert len(schedule) == 4
