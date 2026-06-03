"""Tests for simple volatility-forecast modeling baselines."""

import numpy as np
import pandas as pd

from optimal_execution_engine.research.modeling import (
    fit_linear_model,
    predict_with_linear_model,
    predict_with_persistence,
    predict_with_rolling_mean,
)


def _feature_frame() -> pd.DataFrame:
    """Build deterministic features and target for modeling tests."""
    return pd.DataFrame(
        {
            "lag_1_realized_variance": [0.0010, 0.0012, 0.0014, 0.0016],
            "rolling_5d_realized_variance": [0.0011, 0.0013, 0.0015, 0.0017],
            "opening_realized_variance": [0.0004, 0.0005, 0.0006, 0.0007],
            "opening_return": [0.0010, 0.0015, 0.0020, 0.0025],
            "opening_range": [0.0030, 0.0032, 0.0034, 0.0036],
            "opening_volume_share": [0.16, 0.162, 0.164, 0.166],
            "target_realized_variance": [0.00115, 0.00135, 0.00155, 0.00175],
        }
    )


def test_persistence_baseline_equals_lagged_target() -> None:
    """Persistence baseline should return lag-1 realized variance directly."""
    frame = _feature_frame()

    predictions = predict_with_persistence(frame=frame)

    assert np.allclose(predictions, frame["lag_1_realized_variance"].to_numpy())


def test_rolling_mean_baseline_equals_requested_column() -> None:
    """Rolling-mean baseline should mirror the selected rolling feature."""
    frame = _feature_frame()

    predictions = predict_with_rolling_mean(
        frame=frame,
        rolling_feature_name="rolling_5d_realized_variance",
    )

    assert np.allclose(predictions, frame["rolling_5d_realized_variance"].to_numpy())


def test_linear_model_fit_and_predict_returns_expected_shape() -> None:
    """Linear model should fit and produce one prediction per row."""
    frame = _feature_frame()
    feature_columns = [
        "opening_realized_variance",
        "opening_return",
        "opening_range",
        "opening_volume_share",
        "lag_1_realized_variance",
        "rolling_5d_realized_variance",
        "rolling_10d_realized_variance",
    ]
    frame["rolling_10d_realized_variance"] = frame["rolling_5d_realized_variance"]

    model = fit_linear_model(
        train_frame=frame,
        feature_columns=feature_columns,
        target_column="target_realized_variance",
    )
    predictions = predict_with_linear_model(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
    )

    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()
