"""Feature-engineering helpers for daily realized-variance modeling tables."""

import numpy as np
import pandas as pd

from optimal_execution_engine.research.realized_variance import normalize_intraday_bars


def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a lagged rolling mean that excludes the current row.

    Parameters
    ----------
    series
        Chronologically ordered observations for one symbol.
    window
        Number of prior observations included in each mean.

    Returns
    -------
    pd.Series
        Rolling means aligned with the input index. The first ``window`` values
        are missing because insufficient prior observations exist.
    """
    return series.shift(1).rolling(window=window, min_periods=window).mean()


def build_opening_feature_table(
    bars: pd.DataFrame,
    opening_window_bars: int,
) -> pd.DataFrame:
    """Build opening-window predictors from raw intraday bars.

    Parameters
    ----------
    bars
        Intraday bars with ``symbol``, ``ts``, ``open``, ``high``, ``low``,
        ``close``, and ``volume``.
    opening_window_bars
        Number of opening bars used to build opening-window predictors.

    Returns
    -------
    pd.DataFrame
        Daily opening-feature table with one row per ``symbol``/``trade_date``.
    """
    if opening_window_bars <= 0:
        raise ValueError("opening_window_bars must be positive.")

    normalized = normalize_intraday_bars(bars=bars)
    normalized["bar_index"] = normalized.groupby(["symbol", "trade_date"]).cumcount()

    opening_window = normalized.loc[
        normalized["bar_index"] < opening_window_bars
    ].copy()

    opening_summary = (
        opening_window.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            opening_open_price=("open", "first"),
            opening_close_price=("close", "last"),
            opening_high_price=("high", "max"),
            opening_low_price=("low", "min"),
            opening_volume=("volume", "sum"),
        )
        .reset_index(drop=True)
    )

    opening_features = opening_summary.copy()

    opening_features["opening_return"] = np.log(
        opening_features["opening_close_price"].astype(float)
        / opening_features["opening_open_price"].astype(float)
    )

    opening_features["opening_range"] = (
        opening_features["opening_high_price"].astype(float)
        - opening_features["opening_low_price"].astype(float)
    ) / opening_features["opening_open_price"].astype(float)

    # Total-day volume is unknown at the forecast cutoff. Log opening volume is
    # fully observable then and reduces the scale disparity in linear fitting.
    opening_features["opening_log_volume"] = np.log1p(
        opening_features["opening_volume"].astype(float)
    )

    return opening_features[
        [
            "symbol",
            "trade_date",
            "opening_return",
            "opening_range",
            "opening_log_volume",
        ]
    ]


def build_feature_table(
    daily_targets: pd.DataFrame,
    opening_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a daily modeling feature table with explicit lagging.

    Parameters
    ----------
    daily_targets
        Daily target frame with ``symbol``, ``trade_date``, and
        ``target_remaining_realized_variance``.
    opening_features
        Daily opening features with ``symbol`` and ``trade_date`` keys and
        opening-window predictors.

    Returns
    -------
    pd.DataFrame
        Modeling frame with target, opening predictors, lagged realized
        variance, and rolling realized-variance means.
    """
    base = daily_targets.copy()
    base["trade_date"] = base["trade_date"].astype(str)
    base = base.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    merged = base.merge(
        opening_features,
        on=["symbol", "trade_date"],
        how="inner",
    )
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Lag and rolling features are computed per symbol to avoid cross-ticker leakage.
    merged["lag_1_remaining_realized_variance"] = merged.groupby("symbol")[
        "target_remaining_realized_variance"
    ].shift(1)

    merged["rolling_5d_remaining_realized_variance"] = merged.groupby("symbol")[
        "target_remaining_realized_variance"
    ].transform(lambda values: _lagged_rolling_mean(values, window=5))

    merged["rolling_10d_remaining_realized_variance"] = merged.groupby("symbol")[
        "target_remaining_realized_variance"
    ].transform(lambda values: _lagged_rolling_mean(values, window=10))

    required_columns = [
        "opening_return",
        "opening_range",
        "opening_log_volume",
        "lag_1_remaining_realized_variance",
        "rolling_5d_remaining_realized_variance",
        "rolling_10d_remaining_realized_variance",
    ]

    modeling_frame = merged.dropna(subset=required_columns).reset_index(drop=True)
    return modeling_frame
