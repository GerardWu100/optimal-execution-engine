"""Aggregate schedule comparison metrics."""

import pandas as pd


def summarize_execution(simulation_frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate total dollars, total basis points, and average fill price.

    Parameters
    ----------
    simulation_frame
        Slice-level execution simulation output.

    Returns
    -------
    dict[str, float]
        Aggregated statistics for comparing schedules.
    """
    total_notional = float(
        (simulation_frame["shares"] * simulation_frame["mid_price"]).sum()
    )
    total_cost_dollars = float(simulation_frame["cost_dollars"].sum())
    total_cost_bps = (
        (total_cost_dollars / total_notional * 10_000.0)
        if total_notional > 0.0
        else 0.0
    )

    return {
        "total_cost_dollars": total_cost_dollars,
        "total_cost_bps": float(total_cost_bps),
        "mean_cost_bps": float(simulation_frame["cost_bps"].mean()),
        "average_fill_price": float(simulation_frame["fill_price"].mean()),
    }
