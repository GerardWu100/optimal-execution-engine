"""Slice-level execution simulation."""

import pandas as pd

from optimal_execution_engine.types import ParentOrder


BASE_IMPACT_BPS: float = 2.0
VOLUME_SHARE_IMPACT_COEFFICIENT_BPS: float = 25.0


def simulate_schedule(
    schedule: pd.DataFrame, bars: pd.DataFrame, arrival_price: float, side: str
) -> pd.DataFrame:
    """Apply a simple volume-share impact model to each schedule slice.

    Parameters
    ----------
    schedule
        Schedule DataFrame with `shares` and ordering column.
    bars
        Market bars with `close` and `volume` columns, ordered in the same
        sequence as ``schedule`` rows.
    arrival_price
        Benchmark arrival price.
    side
        Trade direction, expected values are "BUY" or "SELL".

    Returns
    -------
    pd.DataFrame
        Slice-level simulation with fill and cost metrics.
    """
    if len(bars) < len(schedule):
        raise ValueError("bars must contain at least as many rows as schedule slices.")

    normalized_side = side.upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL.")

    side_multiplier = 1.0 if normalized_side == "BUY" else -1.0

    simulation_frame = schedule.copy().reset_index(drop=True)
    aligned_bars = bars.head(len(schedule)).reset_index(drop=True)

    simulation_frame["mid_price"] = aligned_bars["close"].astype(float)
    simulation_frame["bar_volume"] = aligned_bars["volume"].astype(float)

    # volume_share = executed_shares / bar_volume
    simulation_frame["volume_share"] = (
        simulation_frame["shares"].astype(float) / simulation_frame["bar_volume"]
    )

    # impact_bps = base_impact_bps + coefficient_bps * volume_share
    simulation_frame["impact_bps"] = (
        BASE_IMPACT_BPS
        + VOLUME_SHARE_IMPACT_COEFFICIENT_BPS * simulation_frame["volume_share"]
    )

    # For buys we cross up from mid, for sells we cross down from mid.
    simulation_frame["fill_price"] = simulation_frame["mid_price"] * (
        1.0 + side_multiplier * simulation_frame["impact_bps"] / 10_000.0
    )

    signed_cost = side_multiplier * (
        simulation_frame["fill_price"] - float(arrival_price)
    )
    simulation_frame["cost_dollars"] = (
        signed_cost * simulation_frame["shares"].astype(float)
    )
    simulation_frame["cost_bps"] = (
        simulation_frame["cost_dollars"]
        / (float(arrival_price) * simulation_frame["shares"].astype(float))
        * 10_000.0
    )

    return simulation_frame
