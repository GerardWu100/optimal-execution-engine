"""Slice-level execution simulation."""

import pandas as pd

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

    Raises
    ------
    ValueError
        Raised when `bars` has fewer rows than `schedule`, when `side` is
        neither "BUY" nor "SELL", or when a slice asks to trade shares in a bar
        whose volume is not a positive number (a halted or empty bar).
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

    # A bar with no positive volume is a halt (or an empty/missing bar). The
    # volume-share model divides by that volume, so without a guard the slice
    # gets an infinite impact and one bad bar poisons every aggregate cost
    # number downstream.
    #
    # Neither obvious rescue is acceptable, so this raises instead:
    #   - "zero impact" would price a halted bar as the cheapest bar of the day,
    #     the opposite of reality, and would bias schedule comparisons toward
    #     schedules that dump shares into halts;
    #   - "skip the bar" would drop that slice's shares, so the simulated order
    #     would be smaller than the requested order and the reported cost would
    #     describe a trade that was never asked for.
    # Both answers look plausible and are silently wrong. A zero-volume bar
    # means the schedule is infeasible as written, so the caller must re-plan
    # over tradeable bars.
    #
    # The test is "not greater than zero" rather than "<= 0" so that a NaN
    # volume from missing market data is caught too. Slices with zero shares are
    # allowed: a schedule that already routes nothing into a halted bar is fine.
    untradeable_slices = simulation_frame["shares"].astype(float).gt(0.0) & ~(
        simulation_frame["bar_volume"].gt(0.0)
    )
    if untradeable_slices.any():
        untradeable_positions = untradeable_slices.to_numpy().nonzero()[0].tolist()
        raise ValueError(
            "bars must have positive volume wherever the schedule trades shares; "
            f"halted or empty bars at slice positions {untradeable_positions}."
        )

    # volume_share = executed_shares / bar_volume
    # The guard above leaves only one division-by-zero case: zero shares in a
    # zero-volume bar, where 0/0 is NaN. Nothing trades in that slice, so its
    # volume share is 0 and it contributes no cost, rather than a NaN that would
    # spread through the aggregates.
    simulation_frame["volume_share"] = (
        simulation_frame["shares"]
        .astype(float)
        .div(simulation_frame["bar_volume"])
        .where(simulation_frame["shares"].astype(float).gt(0.0), 0.0)
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
    simulation_frame["cost_dollars"] = signed_cost * simulation_frame["shares"].astype(
        float
    )
    simulation_frame["arrival_price"] = float(arrival_price)
    simulation_frame["arrival_notional"] = float(arrival_price) * simulation_frame[
        "shares"
    ].astype(float)
    simulation_frame["cost_bps"] = (
        simulation_frame["cost_dollars"]
        / simulation_frame["arrival_notional"]
        * 10_000.0
    )

    return simulation_frame
