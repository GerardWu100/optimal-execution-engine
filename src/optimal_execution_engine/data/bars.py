"""Bar normalization helpers shared by CLI, notebook, and execution demos."""

import pandas as pd


def prepare_intraday_bars(raw_bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize bars for scheduling with deterministic day and bucket labels.

    Parameters
    ----------
    raw_bars
        Market bars with at least ``ts`` and price/volume columns.

    Returns
    -------
    pd.DataFrame
        Sorted bars enriched with ``trade_date`` and per-day ``bucket`` indices.
    """
    bars = raw_bars.copy()

    # UTC timestamps keep day boundaries stable across local demo environments.
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars = bars.sort_values("ts").reset_index(drop=True)

    # Bucket counters are consumed by VWAP profile estimation and daily slicing.
    bars["trade_date"] = bars["ts"].dt.date.astype(str)
    bars["bucket"] = bars.groupby("trade_date").cumcount()
    return bars
