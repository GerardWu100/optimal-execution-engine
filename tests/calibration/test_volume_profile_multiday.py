"""Regression tests for multi-day volume-profile construction."""

import pandas as pd

from optimal_execution_engine.calibration.volume_profile import estimate_volume_profile


def test_volume_profile_averages_bucket_volume_across_trade_dates() -> None:
    """Bucket weights should come from mean per-bucket volume across days."""
    bars = pd.DataFrame(
        {
            "trade_date": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-03",
                "2026-01-03",
            ],
            "bucket": [0, 1, 0, 1],
            "volume": [100.0, 300.0, 300.0, 100.0],
        }
    )

    profile = estimate_volume_profile(bars=bars, bucket_column="bucket")

    bucket_zero_weight = float(profile.loc[profile["bucket"] == 0, "weight"].iloc[0])
    bucket_one_weight = float(profile.loc[profile["bucket"] == 1, "weight"].iloc[0])

    # mean volumes are equal: bucket0 mean=200, bucket1 mean=200
    assert abs(bucket_zero_weight - 0.5) < 1e-9
    assert abs(bucket_one_weight - 0.5) < 1e-9
