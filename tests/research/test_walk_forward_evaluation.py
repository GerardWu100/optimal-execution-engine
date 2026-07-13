"""Tests for walk-forward evaluation and forecast metrics."""

import numpy as np
import pandas as pd

from optimal_execution_engine.research.evaluation import (
    compute_mean_absolute_error,
    compute_qlike,
    compute_root_mean_squared_error,
    build_walk_forward_splits,
)


def _modeling_frame() -> pd.DataFrame:
    """Create deterministic multi-date rows for split tests."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 8,
            "trade_date": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-12",
                "2026-01-13",
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
            ],
        }
    )


def test_walk_forward_splits_are_time_ordered_and_non_overlapping() -> None:
    """Walk-forward split helper should preserve chronology without leakage."""
    frame = _modeling_frame()

    splits = build_walk_forward_splits(
        frame=frame,
        train_window_size=4,
        test_window_size=2,
        step_size=2,
    )

    assert len(splits) == 2

    first_split = splits[0]
    assert max(first_split.train_indices) < min(first_split.test_indices)
    assert (
        len(set(first_split.train_indices).intersection(first_split.test_indices)) == 0
    )


def test_mae_rmse_and_qlike_match_manual_values() -> None:
    """Metric helpers should match manual calculations on toy arrays."""
    actual = np.array([0.010, 0.020, 0.015])
    predicted = np.array([0.011, 0.018, 0.012])

    mae = compute_mean_absolute_error(actual=actual, predicted=predicted)
    rmse = compute_root_mean_squared_error(actual=actual, predicted=predicted)
    qlike = compute_qlike(actual=actual, predicted=predicted)

    expected_mae = np.mean(np.abs(actual - predicted))
    expected_rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    expected_qlike = np.mean(np.log(predicted) + (actual / predicted))

    assert np.isclose(mae, expected_mae)
    assert np.isclose(rmse, expected_rmse)
    assert np.isclose(qlike, expected_qlike)
