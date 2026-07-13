"""Small, interpretable volatility-forecast modeling helpers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LinearVarianceModel:
    """Container for linear-model coefficients and intercept.

    Parameters
    ----------
    feature_columns
        Ordered list of feature column names used by the model.
    coefficients
        Linear coefficients aligned with ``feature_columns``.
    intercept
        Additive intercept term.
    """

    feature_columns: list[str]
    coefficients: np.ndarray
    intercept: float


def predict_with_persistence(frame: pd.DataFrame) -> np.ndarray:
    """Use lag-1 realized variance as a persistence forecast.

    Parameters
    ----------
    frame
        Modeling table containing ``lag_1_remaining_realized_variance``.

    Returns
    -------
    np.ndarray
        Predicted realized-variance array.
    """
    return frame["lag_1_remaining_realized_variance"].to_numpy(dtype=float)


def predict_with_rolling_mean(
    frame: pd.DataFrame,
    rolling_feature_name: str,
) -> np.ndarray:
    """Use a rolling realized-variance feature as the forecast.

    Parameters
    ----------
    frame
        Modeling table containing the requested rolling feature column.
    rolling_feature_name
        Name of the rolling feature column used for prediction.

    Returns
    -------
    np.ndarray
        Predicted realized-variance array.
    """
    return frame[rolling_feature_name].to_numpy(dtype=float)


def fit_linear_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> LinearVarianceModel:
    """Fit a small linear model with an explicit least-squares solution.

    Parameters
    ----------
    train_frame
        Training subset of the modeling table.
    feature_columns
        Ordered predictor columns used in the linear model.
    target_column
        Name of the realized-variance target column.

    Returns
    -------
    LinearVarianceModel
        Fitted linear-model container with intercept and coefficients.
    """
    feature_matrix = train_frame[feature_columns].to_numpy(dtype=float)
    target_vector = train_frame[target_column].to_numpy(dtype=float)

    ones_column = np.ones((feature_matrix.shape[0], 1), dtype=float)
    design_matrix = np.hstack([ones_column, feature_matrix])

    # Solve min ||X beta - y||_2 using least squares for stability.
    fitted_parameters, _, _, _ = np.linalg.lstsq(
        design_matrix, target_vector, rcond=None
    )

    intercept = float(fitted_parameters[0])
    coefficients = fitted_parameters[1:]

    return LinearVarianceModel(
        feature_columns=list(feature_columns),
        coefficients=coefficients,
        intercept=intercept,
    )


def predict_with_linear_model(
    model: LinearVarianceModel,
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Generate predictions from a fitted linear realized-variance model.

    Parameters
    ----------
    model
        Fitted linear-model container.
    frame
        Input frame with the model feature columns.
    feature_columns
        Optional override for predictor order. Defaults to ``model.feature_columns``.

    Returns
    -------
    np.ndarray
        Predicted realized variance per row.
    """
    columns = model.feature_columns if feature_columns is None else feature_columns
    if list(columns) != model.feature_columns:
        raise ValueError("feature_columns must match fitted model feature order.")

    feature_matrix = frame[columns].to_numpy(dtype=float)
    return model.intercept + feature_matrix @ model.coefficients
