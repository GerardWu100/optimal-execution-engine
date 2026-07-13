"""Walk-forward evaluation helpers and forecast error metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimal_execution_engine.research.modeling import (
    fit_linear_model,
    predict_with_linear_model,
    predict_with_persistence,
    predict_with_rolling_mean,
)


@dataclass(slots=True)
class WalkForwardSplit:
    """Container for one walk-forward train/test index split.

    Parameters
    ----------
    train_indices
        Positional row indices used for model fitting.
    test_indices
        Positional row indices used for forward evaluation.
    """

    train_indices: list[int]
    test_indices: list[int]


@dataclass(slots=True)
class SplitForecastResult:
    """Forecasts produced on one walk-forward test window.

    Parameters
    ----------
    actual
        Realized variance targets on the test window.
    persistence
        Persistence-baseline forecasts.
    rolling
        Rolling-mean baseline forecasts.
    linear
        Linear-model forecasts clipped to a small positive floor.
    test_trade_dates
        Trade dates aligned with each forecast row.
    """

    actual: np.ndarray
    persistence: np.ndarray
    rolling: np.ndarray
    linear: np.ndarray
    test_trade_dates: pd.Series


def build_walk_forward_splits(
    frame: pd.DataFrame,
    train_window_size: int,
    test_window_size: int,
    step_size: int,
) -> list[WalkForwardSplit]:
    """Build chronological walk-forward splits for leakage-safe evaluation.

    Parameters
    ----------
    frame
        Modeling frame sorted by ``trade_date``.
    train_window_size
        Number of rows included in each training window.
    test_window_size
        Number of rows included in each forward test window.
    step_size
        Number of rows to move the window start after each split.

    Returns
    -------
    list[WalkForwardSplit]
        Sequence of non-overlapping chronological train/test splits.
    """
    if train_window_size <= 0:
        raise ValueError("train_window_size must be positive.")
    if test_window_size <= 0:
        raise ValueError("test_window_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    total_rows = len(ordered)

    splits: list[WalkForwardSplit] = []
    train_start = 0

    while True:
        train_end = train_start + train_window_size
        test_end = train_end + test_window_size

        if test_end > total_rows:
            break

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(train_end, test_end))
        splits.append(
            WalkForwardSplit(
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )

        train_start += step_size

    return splits


def evaluate_walk_forward_split(
    modeling_frame: pd.DataFrame,
    split: WalkForwardSplit,
    feature_columns: list[str],
    target_column: str = "target_remaining_realized_variance",
    rolling_feature_name: str = "rolling_5d_remaining_realized_variance",
) -> SplitForecastResult:
    """Fit models on one train window and forecast the paired test window.

    Parameters
    ----------
    modeling_frame
        Full modeling table sorted by trade date.
    split
        One walk-forward index split.
    feature_columns
        Ordered predictor columns for the linear model.
    target_column
        Realized-variance target column name.
    rolling_feature_name
        Rolling feature used by the rolling-mean baseline.

    Returns
    -------
    SplitForecastResult
        Actual targets and baseline/linear forecasts for the test window.
    """
    train_frame = modeling_frame.iloc[split.train_indices].reset_index(drop=True)
    test_frame = modeling_frame.iloc[split.test_indices].reset_index(drop=True)

    linear_model = fit_linear_model(
        train_frame=train_frame,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    actual = test_frame[target_column].to_numpy(dtype=float)
    persistence = predict_with_persistence(frame=test_frame)
    rolling = predict_with_rolling_mean(
        frame=test_frame,
        rolling_feature_name=rolling_feature_name,
    )
    linear = predict_with_linear_model(model=linear_model, frame=test_frame)
    linear = linear.clip(min=1e-12)

    return SplitForecastResult(
        actual=actual,
        persistence=persistence,
        rolling=rolling,
        linear=linear,
        test_trade_dates=test_frame["trade_date"],
    )


def compute_mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE) for realized-variance forecasts.

    Parameters
    ----------
    actual
        Array of realized target values.
    predicted
        Array of forecast values.

    Returns
    -------
    float
        Mean absolute error value.
    """
    return float(np.mean(np.abs(actual - predicted)))


def compute_root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) for forecasts.

    Parameters
    ----------
    actual
        Array of realized target values.
    predicted
        Array of forecast values.

    Returns
    -------
    float
        Root mean squared error value.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute QLIKE loss for realized-variance forecast evaluation.

    Parameters
    ----------
    actual
        Array of realized variance targets.
    predicted
        Array of predicted variance values.

    Returns
    -------
    float
        Mean QLIKE value, where lower values are better.
    """
    clipped_predictions = np.clip(predicted, 1e-12, None)
    return float(np.mean(np.log(clipped_predictions) + (actual / clipped_predictions)))
