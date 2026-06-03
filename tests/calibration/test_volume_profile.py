"""Tests for estimating an intraday volume curve."""

import pandas as pd

from optimal_execution_engine.calibration.volume_profile import estimate_volume_profile


def test_volume_profile_weights_sum_to_one() -> None:
    """Normalized intraday volume weights should sum to one."""
    bars = pd.DataFrame(
        {
            "minute_bucket": [0, 1, 2, 0, 1, 2],
            "volume": [100.0, 200.0, 300.0, 150.0, 150.0, 300.0],
        }
    )

    profile = estimate_volume_profile(bars=bars, bucket_column="minute_bucket")

    assert abs(float(profile["weight"].sum()) - 1.0) < 1e-9
