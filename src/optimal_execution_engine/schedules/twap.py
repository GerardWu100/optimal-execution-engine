"""Time-weighted baseline scheduler."""

import numpy as np
import pandas as pd

from optimal_execution_engine.schedules.shares import reconcile_integer_shares
from optimal_execution_engine.types import ParentOrder


def build_twap_schedule(order: ParentOrder, slice_count: int) -> pd.DataFrame:
    """Allocate shares evenly across execution slices.

    Parameters
    ----------
    order
        Parent order definition.
    slice_count
        Number of equal time slices over the execution horizon.

    Returns
    -------
    pd.DataFrame
        Slice-level schedule with `slice_index` and `shares` columns.
    """
    if slice_count <= 0:
        raise ValueError("slice_count must be positive.")

    base_slice_shares = order.shares // slice_count
    slice_shares = np.full(slice_count, base_slice_shares, dtype=int)
    reconciled_shares = reconcile_integer_shares(
        raw_shares=slice_shares,
        target_total=order.shares,
    )

    return pd.DataFrame(
        {
            "slice_index": np.arange(slice_count),
            "shares": reconciled_shares,
        }
    )
