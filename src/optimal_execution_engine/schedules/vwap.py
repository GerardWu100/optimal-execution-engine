"""Volume-weighted scheduling baseline."""

import numpy as np
import pandas as pd

from optimal_execution_engine.schedules.shares import reconcile_integer_shares
from optimal_execution_engine.types import ParentOrder


def build_vwap_schedule(
    order: ParentOrder, volume_profile: pd.DataFrame, bucket_column: str
) -> pd.DataFrame:
    """Allocate shares in proportion to expected intraday volume weights.

    Parameters
    ----------
    order
        Parent order definition.
    volume_profile
        Frame containing the `bucket_column` and normalized `weight` values.
    bucket_column
        Name of the intraday bucket column.

    Returns
    -------
    pd.DataFrame
        Bucket-level schedule with allocated shares.
    """
    if volume_profile.empty:
        raise ValueError("volume_profile must not be empty.")

    schedule = volume_profile.copy().reset_index(drop=True)
    schedule["shares"] = np.round(
        schedule["weight"].astype(float) * order.shares
    ).astype(int)

    # Residual shares land on the final bucket so earlier weights stay unchanged.
    schedule["shares"] = reconcile_integer_shares(
        raw_shares=schedule["shares"].to_numpy(),
        target_total=order.shares,
    )

    return schedule[[bucket_column, "shares"]]
