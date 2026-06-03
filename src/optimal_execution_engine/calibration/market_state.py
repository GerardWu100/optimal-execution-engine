"""Estimate basic execution inputs from intraday bars."""

import numpy as np
import pandas as pd

from optimal_execution_engine.types import MarketState


MINUTES_PER_TRADING_DAY: float = 390.0
DEFAULT_SPREAD_BPS: float = 5.0


def _estimate_realized_bar_volatility(close_series: pd.Series) -> float:
    """Estimate bar-level realized volatility from close-price log returns.

    Parameters
    ----------
    close_series
        Close-price series ordered in event time.

    Returns
    -------
    float
        Realized volatility at the bar frequency in decimal units.
    """
    # log_return_t = log(price_t / price_(t-1))
    log_returns = np.log(close_series / close_series.shift(1)).dropna()

    if log_returns.empty:
        return 0.0

    # Sample std is preferred; one return falls back to absolute move magnitude.
    if len(log_returns) == 1:
        return float(abs(log_returns.iloc[0]))

    return float(log_returns.std(ddof=1))


def _build_trade_date_labels(bars: pd.DataFrame) -> pd.Series:
    """Construct trade-date labels used for ADV aggregation.

    Parameters
    ----------
    bars
        Input bars used for market-state calibration.

    Returns
    -------
    pd.Series
        String trade-date labels aligned with ``bars`` rows.
    """
    if "trade_date" in bars.columns:
        return bars["trade_date"].astype(str)
    if "ts" in bars.columns:
        return pd.to_datetime(bars["ts"], utc=True).dt.date.astype(str)

    # Fallback preserves historical behavior for toy inputs with no timestamps.
    return pd.Series(["single_day"] * len(bars), index=bars.index)


def calibrate_market_state(
    bars: pd.DataFrame,
    override_daily_volatility: float | None = None,
) -> MarketState:
    """Estimate daily volume, volatility, and spread proxy from bar data.

    Parameters
    ----------
    bars
        Intraday bars with at least `close` and `volume` columns.
    override_daily_volatility
        Optional externally forecast daily volatility in decimal units. When
        provided, this value replaces the volatility estimated from bars.

    Returns
    -------
    MarketState
        Estimated average daily volume, daily volatility, and spread proxy.
    """
    close_series = bars["close"].astype(float)
    volume_series = bars["volume"].astype(float)

    # Estimate bar-frequency volatility first, then scale to one trading day.
    realized_bar_volatility = _estimate_realized_bar_volatility(close_series)

    # daily_volatility = realized_bar_volatility * sqrt(minutes_per_day)
    estimated_daily_volatility = realized_bar_volatility * np.sqrt(
        MINUTES_PER_TRADING_DAY
    )

    daily_volatility = float(estimated_daily_volatility)
    if override_daily_volatility is not None:
        daily_volatility = float(max(override_daily_volatility, 0.0))

    # Compute per-day volume totals, then average across observed trading dates.
    trade_date_labels = _build_trade_date_labels(bars)
    volume_by_trade_date = (
        pd.DataFrame({"trade_date": trade_date_labels, "volume": volume_series})
        .groupby("trade_date", as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "daily_volume"})
    )
    average_daily_volume = float(volume_by_trade_date["daily_volume"].mean())

    # Spread is a fixed demo proxy rather than an estimated microstructure input.
    return MarketState(
        average_daily_volume=average_daily_volume,
        daily_volatility=daily_volatility,
        spread_bps=DEFAULT_SPREAD_BPS,
    )
