"""Tests for the VWAP-style scheduler."""

import pandas as pd

from optimal_execution_engine.schedules.vwap import build_vwap_schedule
from optimal_execution_engine.types import ParentOrder


def test_vwap_schedule_allocates_more_to_heavier_volume_buckets() -> None:
    """A heavier expected volume bucket should receive more shares."""
    order = ParentOrder("AAPL", "BUY", 1000, 100.0, 60, 1.0)
    volume_profile = pd.DataFrame({"bucket": [0, 1, 2], "weight": [0.2, 0.3, 0.5]})

    schedule = build_vwap_schedule(
        order=order, volume_profile=volume_profile, bucket_column="bucket"
    )

    assert int(schedule["shares"].sum()) == 1000
    assert int(schedule.iloc[-1]["shares"]) > int(schedule.iloc[0]["shares"])
