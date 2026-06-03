"""Aggregate metrics across many execution experiments."""

import pandas as pd


def _compute_win_rate_vs_twap(results: pd.DataFrame) -> dict[str, float]:
    """Compute date-matched win rates against TWAP for each schedule.

    Parameters
    ----------
    results
        Long-form results with ``trade_date``, ``schedule_name``, and ``cost_bps``.

    Returns
    -------
    dict[str, float]
        Mapping from schedule name to fraction of dates with lower cost than TWAP.
    """
    required_columns = {"trade_date", "schedule_name", "cost_bps"}
    if not required_columns.issubset(results.columns):
        return {}

    # Pivot once so each date has one column per schedule for like-for-like comparison.
    pivoted_costs = results.pivot_table(
        index="trade_date",
        columns="schedule_name",
        values="cost_bps",
        aggfunc="mean",
    )
    if "twap" not in pivoted_costs.columns:
        return {}

    twap_costs = pivoted_costs["twap"]
    comparable_costs = pivoted_costs.drop(columns=["twap"]).join(twap_costs, how="inner")
    comparable_costs = comparable_costs.dropna()

    if comparable_costs.empty:
        return {"twap": 0.0}

    win_rates: dict[str, float] = {"twap": 0.0}
    for schedule_name in comparable_costs.columns:
        if schedule_name == "twap":
            continue

        win_rates[schedule_name] = float(
            (comparable_costs[schedule_name] < comparable_costs["twap"]).mean()
        )

    return win_rates


def summarize_experiment_batch(results: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-day central, dispersion, and benchmark-relative metrics.

    Parameters
    ----------
    results
        DataFrame with `schedule_name` and `cost_bps` columns.

    Returns
    -------
    pd.DataFrame
        Grouped summary including mean, median, dispersion, tail, and win rate.
    """
    # Aggregate core distribution metrics at the schedule level.
    grouped = results.groupby("schedule_name")["cost_bps"]
    summary = grouped.agg(
        mean_cost_bps="mean",
        median_cost_bps="median",
        std_cost_bps="std",
        p90_cost_bps=lambda series: float(series.quantile(0.9)),
        evaluation_days="count",
    ).reset_index()

    summary["std_cost_bps"] = summary["std_cost_bps"].fillna(0.0)

    # Benchmark-relative metric is optional when date columns are unavailable.
    win_rate_by_schedule = _compute_win_rate_vs_twap(results)
    summary["win_rate_vs_twap"] = summary["schedule_name"].map(
        lambda name: float(win_rate_by_schedule.get(name, 0.0))
    )
    return summary
