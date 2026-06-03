"""Estimate normalized intraday volume weights from historical bars."""

import pandas as pd


def estimate_volume_profile(bars: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    """Aggregate volume by intraday bucket and normalize to one.

    Parameters
    ----------
    bars
        Historical intraday bars containing `volume` and bucket columns.
    bucket_column
        Column name representing the intraday bucket identifier.

    Returns
    -------
    pd.DataFrame
        Frame with `bucket_column` and normalized `weight` columns.
    """
    # Prefer multi-day historical averaging by bucket when trade_date is present.
    if "trade_date" in bars.columns:
        per_day_bucket_volume = (
            bars.groupby(["trade_date", bucket_column], as_index=False)["volume"]
            .sum()
            .rename(columns={"volume": "daily_bucket_volume"})
        )
        grouped = (
            per_day_bucket_volume.groupby(bucket_column, as_index=False)[
                "daily_bucket_volume"
            ]
            .mean()
            .rename(columns={"daily_bucket_volume": "volume"})
        )
    else:
        grouped = bars.groupby(bucket_column, as_index=False)["volume"].mean()

    total_volume = float(grouped["volume"].sum())

    if total_volume <= 0.0:
        raise ValueError("Total grouped volume must be positive to build a profile.")

    grouped["weight"] = grouped["volume"] / total_volume
    return grouped[[bucket_column, "weight"]]
