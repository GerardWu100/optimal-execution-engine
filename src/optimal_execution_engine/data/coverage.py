"""Shared timestamp-coverage metadata for cache and ClickHouse paths."""

from datetime import time

import pandas as pd


REGULAR_SESSION_START_UTC: time = time(hour=14, minute=30)
REGULAR_SESSION_END_UTC: time = time(hour=20, minute=0)


EMPTY_TIMESTAMP_COVERAGE: dict[str, object] = {
    "first_bar_utc": "",
    "last_bar_utc": "",
    "trade_dates": 0,
    "rows_per_day_min": 0,
    "rows_per_day_max": 0,
    "unique_bar_spacing_minutes": [],
}


def build_timestamp_coverage(
    frame: pd.DataFrame,
    include_session_check: bool = False,
) -> dict[str, object]:
    """Compute deterministic timestamp-coverage metadata for one bar frame.

    Parameters
    ----------
    frame
        Dataset frame with a ``ts`` column when coverage can be measured.
    include_session_check
        When ``True``, add ``regular_session_time_check_passed`` for extraction QA.

    Returns
    -------
    dict[str, object]
        Coverage dictionary used by cache sidecars and ClickHouse metadata.
    """
    coverage = dict(EMPTY_TIMESTAMP_COVERAGE)

    if frame.empty or "ts" not in frame.columns:
        if include_session_check:
            coverage["regular_session_time_check_passed"] = True
        return coverage

    # Normalize once so spacing and day counts are stable across writers.
    timestamps = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        if include_session_check:
            coverage["regular_session_time_check_passed"] = True
        return coverage

    trade_dates = timestamps.dt.date
    rows_per_day = trade_dates.value_counts()
    ordered_timestamps = timestamps.sort_values().reset_index(drop=True)

    # Bar spacing is measured on consecutive sorted timestamps in minutes.
    spacing_minutes = (
        ordered_timestamps.diff().dropna().dt.total_seconds().div(60.0).round(3)
    )

    coverage["first_bar_utc"] = ordered_timestamps.iloc[0].isoformat()
    coverage["last_bar_utc"] = ordered_timestamps.iloc[-1].isoformat()
    coverage["trade_dates"] = int(trade_dates.nunique())
    coverage["rows_per_day_min"] = int(rows_per_day.min())
    coverage["rows_per_day_max"] = int(rows_per_day.max())
    coverage["unique_bar_spacing_minutes"] = sorted(
        [float(value) for value in spacing_minutes.unique().tolist()]
    )

    if include_session_check:
        # Extraction QA checks that every bar falls inside regular-session UTC bounds.
        regular_hours_mask = (timestamps.dt.time >= REGULAR_SESSION_START_UTC) & (
            timestamps.dt.time <= REGULAR_SESSION_END_UTC
        )
        coverage["regular_session_time_check_passed"] = bool(regular_hours_mask.all())

    return coverage
