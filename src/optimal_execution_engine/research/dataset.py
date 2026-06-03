"""Dataset assembly helpers for offline realized-variance research workflows."""

import pandas as pd

from optimal_execution_engine.research.features import (
    build_feature_table,
    build_opening_feature_table,
)
from optimal_execution_engine.research.realized_variance import (
    compute_daily_realized_variance,
    compute_opening_window_realized_variance,
)


LINEAR_MODEL_FEATURE_COLUMNS: list[str] = [
    "opening_realized_variance",
    "opening_return",
    "opening_range",
    "opening_volume_share",
    "lag_1_realized_variance",
    "rolling_5d_realized_variance",
    "rolling_10d_realized_variance",
]

MODELING_OUTPUT_COLUMNS: list[str] = [
    "symbol",
    "trade_date",
    "target_realized_variance",
    *LINEAR_MODEL_FEATURE_COLUMNS,
]


def build_modeling_dataset(
    daily_targets: pd.DataFrame,
    opening_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join target and feature inputs into a final modeling table.

    Parameters
    ----------
    daily_targets
        Daily target frame with ``symbol``, ``trade_date``, and
        ``target_realized_variance``.
    opening_features
        Daily opening-feature frame with ``symbol`` and ``trade_date`` plus
        opening predictors.

    Returns
    -------
    pd.DataFrame
        Modeling-ready frame with target and all required predictors.
    """
    feature_frame = build_feature_table(
        daily_targets=daily_targets,
        opening_features=opening_features,
    )

    return (
        feature_frame[MODELING_OUTPUT_COLUMNS]
        .sort_values(["symbol", "trade_date"])
        .reset_index(drop=True)
    )


def build_modeling_dataset_from_bars(
    bars: pd.DataFrame,
    opening_window_bars: int,
) -> pd.DataFrame:
    """Create a modeling dataset directly from raw intraday bars.

    Parameters
    ----------
    bars
        Intraday bars with OHLCV columns and symbol/timestamp keys.
    opening_window_bars
        Number of opening bars used in opening-window feature construction.

    Returns
    -------
    pd.DataFrame
        Daily modeling table ready for forecast modeling and evaluation.
    """
    daily_targets = compute_daily_realized_variance(bars=bars)
    opening_target = compute_opening_window_realized_variance(
        bars=bars,
        opening_window_bars=opening_window_bars,
    )
    opening_features = build_opening_feature_table(
        bars=bars,
        opening_window_bars=opening_window_bars,
    )

    # Attach opening-window realized variance before the final feature merge.
    opening_enriched = opening_features.merge(
        opening_target,
        on=["symbol", "trade_date"],
        how="inner",
    )

    return build_modeling_dataset(
        daily_targets=daily_targets,
        opening_features=opening_enriched,
    )
