"""Discrete Almgren-Chriss execution schedule."""

import numpy as np
import pandas as pd

from optimal_execution_engine.schedules.shares import reconcile_integer_shares
from optimal_execution_engine.types import MarketState, ParentOrder


MIN_URGENCY: float = 1e-6


def build_almgren_chriss_schedule(
    order: ParentOrder, market_state: MarketState, slice_count: int
) -> pd.DataFrame:
    """Create a front-loaded schedule when inventory risk matters.

    Parameters
    ----------
    order
        Parent order definition.
    market_state
        Calibrated market statistics.
    slice_count
        Number of execution slices.

    Returns
    -------
    pd.DataFrame
        Slice-level schedule with integer share allocations.
    """
    if slice_count <= 0:
        raise ValueError("slice_count must be positive.")

    normalized_time = np.linspace(0.0, 1.0, slice_count + 1)

    # urgency = lambda * sigma where lambda is risk aversion and sigma is daily vol.
    urgency = max(order.risk_aversion * market_state.daily_volatility, MIN_URGENCY)

    # x(t) = sinh(urgency * (1 - t)) / sinh(urgency)
    inventory_curve = np.sinh(urgency * (1.0 - normalized_time)) / np.sinh(urgency)

    inventory_shares = order.shares * inventory_curve
    raw_slice_shares = -np.diff(inventory_shares)

    rounded_slice_shares = np.round(raw_slice_shares).astype(int)
    reconciled_shares = reconcile_integer_shares(
        raw_shares=rounded_slice_shares,
        target_total=order.shares,
    )

    return pd.DataFrame(
        {"slice_index": np.arange(slice_count), "shares": reconciled_shares}
    )
