"""Tests for research feature-engineering helpers."""

import numpy as np
import pandas as pd

from optimal_execution_engine.research.dataset import build_modeling_dataset
from optimal_execution_engine.research.features import (
    build_feature_table,
    build_opening_feature_table,
)


def _build_daily_target_frame() -> pd.DataFrame:
    """Construct deterministic daily target values for one symbol."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 12,
            "trade_date": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-12",
                "2026-01-13",
                "2026-01-14",
                "2026-01-15",
                "2026-01-16",
                "2026-01-20",
            ],
            "target_remaining_realized_variance": [
                0.0010,
                0.0011,
                0.0012,
                0.0013,
                0.0014,
                0.0015,
                0.0016,
                0.0017,
                0.0018,
                0.0019,
                0.0020,
                0.0021,
            ],
        }
    )


def _build_opening_feature_frame() -> pd.DataFrame:
    """Construct opening-window features aligned by symbol/date."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 12,
            "trade_date": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-12",
                "2026-01-13",
                "2026-01-14",
                "2026-01-15",
                "2026-01-16",
                "2026-01-20",
            ],
            "opening_realized_variance": [
                0.0004,
                0.00045,
                0.0005,
                0.00055,
                0.0006,
                0.00065,
                0.0007,
                0.00075,
                0.0008,
                0.00085,
                0.0009,
                0.00095,
            ],
            "opening_return": [
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.010,
                0.011,
                0.012,
            ],
            "opening_range": [
                0.003,
                0.0031,
                0.0032,
                0.0033,
                0.0034,
                0.0035,
                0.0036,
                0.0037,
                0.0038,
                0.0039,
                0.004,
                0.0041,
            ],
            "opening_log_volume": [
                0.16,
                0.161,
                0.162,
                0.163,
                0.164,
                0.165,
                0.166,
                0.167,
                0.168,
                0.169,
                0.17,
                0.171,
            ],
        }
    )


def test_feature_table_contains_lags_and_rolling_features_without_lookahead() -> None:
    """Feature builder should use only lagged target information."""
    daily_targets = _build_daily_target_frame()
    opening_features = _build_opening_feature_frame()

    features = build_feature_table(
        daily_targets=daily_targets,
        opening_features=opening_features,
    )

    row = features.loc[features["trade_date"] == "2026-01-16"].iloc[0]

    assert np.isclose(row["lag_1_remaining_realized_variance"], 0.0019)
    assert np.isclose(row["rolling_5d_remaining_realized_variance"], 0.0017)
    assert np.isclose(row["rolling_10d_remaining_realized_variance"], 0.00145)


def test_feature_table_keeps_symbol_and_date_alignment() -> None:
    """Feature table rows should stay aligned by symbol/date."""
    daily_targets = _build_daily_target_frame()
    opening_features = _build_opening_feature_frame()

    features = build_feature_table(
        daily_targets=daily_targets,
        opening_features=opening_features,
    )

    assert set(features["symbol"]) == {"AAPL"}
    assert features["trade_date"].is_monotonic_increasing


def test_feature_table_drops_start_rows_without_required_history() -> None:
    """Rows lacking required lag/rolling history should be dropped."""
    daily_targets = _build_daily_target_frame()
    opening_features = _build_opening_feature_frame()

    features = build_feature_table(
        daily_targets=daily_targets,
        opening_features=opening_features,
    )

    assert features["trade_date"].min() == "2026-01-16"


def test_build_modeling_dataset_merges_targets_and_features() -> None:
    """Dataset helper should return a modeling-ready table with required columns."""
    daily_targets = _build_daily_target_frame()
    opening_features = _build_opening_feature_frame()

    dataset = build_modeling_dataset(
        daily_targets=daily_targets,
        opening_features=opening_features,
    )

    expected_columns = {
        "symbol",
        "trade_date",
        "target_remaining_realized_variance",
        "opening_realized_variance",
        "opening_return",
        "opening_range",
        "opening_log_volume",
        "lag_1_remaining_realized_variance",
        "rolling_5d_remaining_realized_variance",
        "rolling_10d_remaining_realized_variance",
    }
    assert expected_columns.issubset(dataset.columns)


def test_opening_features_do_not_depend_on_post_cutoff_bars() -> None:
    """Changing future prices and volume must not alter forecast-time features."""
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "ts": pd.date_range(
                "2026-01-02 14:30",
                periods=4,
                freq="5min",
                tz="UTC",
            ),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000, 1_100, 1_200, 1_300],
        }
    )
    altered_future = bars.copy()
    altered_future.loc[2:, ["open", "high", "low", "close"]] *= 2.0
    altered_future.loc[2:, "volume"] *= 10

    original = build_opening_feature_table(bars=bars, opening_window_bars=2)
    altered = build_opening_feature_table(
        bars=altered_future,
        opening_window_bars=2,
    )

    pd.testing.assert_frame_equal(original, altered)
