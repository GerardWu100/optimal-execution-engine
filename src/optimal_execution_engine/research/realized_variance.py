"""Utilities to construct realized-variance targets from intraday bars."""

import numpy as np
import pandas as pd


def normalize_intraday_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize bars to sorted UTC timestamps with explicit trade-date labels.

    Parameters
    ----------
    bars
        Intraday bars with at least ``symbol``, ``ts``, and ``close`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of input bars sorted by ``symbol`` then ``ts`` with ``trade_date``.
    """
    normalized = bars.copy()
    normalized["ts"] = pd.to_datetime(normalized["ts"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["ts"]).reset_index(drop=True)
    normalized = normalized.sort_values(["symbol", "ts"]).reset_index(drop=True)
    normalized["trade_date"] = normalized["ts"].dt.date.astype(str)
    return normalized


def compute_log_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday log returns per symbol and trade date.

    Parameters
    ----------
    bars
        Intraday bars with ``symbol``, ``ts``, and ``close`` columns.

    Returns
    -------
    pd.DataFrame
        Sorted frame including ``symbol``, ``ts``, ``trade_date``, ``close``,
        and ``log_return`` where ``log_return = log(P_t / P_{t-1})``.
    """
    normalized = normalize_intraday_bars(bars=bars)
    normalized["close"] = normalized["close"].astype(float)

    # First bar per day has no prior price, so shift leaves NaN until fillna below.
    normalized["log_return"] = normalized.groupby(["symbol", "trade_date"])[
        "close"
    ].transform(lambda prices: np.log(prices / prices.shift(1)))

    return normalized[["symbol", "ts", "trade_date", "close", "log_return"]]


def compute_daily_realized_variance(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday squared log returns into daily realized variance.

    Parameters
    ----------
    bars
        Intraday bars with ``symbol``, ``ts``, and ``close`` columns.

    Returns
    -------
    pd.DataFrame
        Daily target table with ``symbol``, ``trade_date``, and
        ``target_realized_variance`` where
        ``target_realized_variance = sum_t (log_return_t ** 2)``.
    """
    returns_frame = compute_log_returns(bars=bars)

    # Treat missing first-bar returns as zero contribution to the daily sum.
    returns_frame["squared_log_return"] = returns_frame["log_return"].fillna(0.0) ** 2

    daily_target = (
        returns_frame.groupby(["symbol", "trade_date"], as_index=False)[
            "squared_log_return"
        ]
        .sum()
        .rename(columns={"squared_log_return": "target_realized_variance"})
    )
    return daily_target


def compute_opening_window_realized_variance(
    bars: pd.DataFrame,
    opening_window_bars: int,
) -> pd.DataFrame:
    """Compute realized variance over the opening intraday window only.

    Parameters
    ----------
    bars
        Intraday bars with ``symbol``, ``ts``, and ``close`` columns.
    opening_window_bars
        Number of bars from the session open included in the opening window.

    Returns
    -------
    pd.DataFrame
        Daily opening-window table with ``symbol``, ``trade_date``, and
        ``opening_realized_variance``.
    """
    if opening_window_bars <= 0:
        raise ValueError("opening_window_bars must be positive.")

    returns_frame = compute_log_returns(bars=bars)
    returns_frame["bar_index"] = returns_frame.groupby(
        ["symbol", "trade_date"]
    ).cumcount()

    opening_frame = returns_frame.loc[
        returns_frame["bar_index"] < opening_window_bars
    ].copy()

    # Opening-window RV uses the same first-bar exclusion rule as the daily target.
    opening_frame["squared_log_return"] = opening_frame["log_return"].fillna(0.0) ** 2

    opening_target = (
        opening_frame.groupby(["symbol", "trade_date"], as_index=False)[
            "squared_log_return"
        ]
        .sum()
        .rename(columns={"squared_log_return": "opening_realized_variance"})
    )
    return opening_target
