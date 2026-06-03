"""ClickHouse extraction helpers for market-data refreshes."""

from dataclasses import dataclass

import clickhouse_connect
import pandas as pd

from optimal_execution_engine.config import ClickHouseSettings
from optimal_execution_engine.data.contracts import PORTABLE_SOURCE_NOTE
from optimal_execution_engine.data.coverage import build_timestamp_coverage


def build_stock_bars_query(
    symbol: str,
    start_date: str,
    end_date: str,
    session_filter: str = "regular-hours",
) -> str:
    """Build a deterministic SQL query for stock-bar extraction.

    Parameters
    ----------
    symbol
        Ticker symbol to extract.
    start_date
        Inclusive start date in YYYY-MM-DD format.
    end_date
        Inclusive end date in YYYY-MM-DD format.
    session_filter
        Session scope label. ``regular-hours`` applies a time-of-day filter.

    Returns
    -------
    str
        SQL query string for the firstrate.stocks table.
    """
    # Pick one session clause so opening-window filters do not overwrite regular-hours bounds.
    if session_filter == "opening-window-only":
        session_clause = (
            "\n        AND toTime(ts) >= toTime('14:30:00')"
            "\n        AND toTime(ts) <= toTime('15:25:00')"
        )
    elif session_filter == "regular-hours":
        session_clause = (
            "\n        AND toTime(ts) >= toTime('14:30:00')"
            "\n        AND toTime(ts) <= toTime('20:00:00')"
        )
    else:
        session_clause = ""

    return f"""
    SELECT symbol, ts, open, high, low, close, volume
    FROM firstrate.stocks
    WHERE
        symbol = '{symbol}'
        AND toDate(ts) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
        {session_clause}
    ORDER BY ts
    """.strip()


def summarize_timestamp_coverage(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize timestamp coverage and spacing for extraction metadata.

    Parameters
    ----------
    frame
        Extracted bar frame from ClickHouse.

    Returns
    -------
    dict[str, object]
        Coverage dictionary with min/max timestamps and spacing diagnostics.
    """
    return build_timestamp_coverage(frame=frame, include_session_check=True)


@dataclass(slots=True)
class ClickHouseDataset:
    """Container for fetched market data and cache metadata fields."""

    frame: pd.DataFrame
    metadata: dict[str, object]


class ClickHouseMarketDataClient:
    """Small adapter around clickhouse-connect for reproducible extractions."""

    def __init__(self, settings: ClickHouseSettings) -> None:
        """Initialize client only when host configuration is present."""
        if not settings.host:
            raise ValueError("ClickHouse host is required to create a database client.")

        self._settings = settings
        self._client = clickhouse_connect.get_client(
            host=settings.host,
            port=settings.port,
            username=settings.user,
            password=settings.password,
            secure=settings.secure,
            verify=settings.verify,
        )

    def fetch_stock_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        session_filter: str = "regular-hours",
    ) -> ClickHouseDataset:
        """Fetch a stock-bar slice and return frame plus provenance metadata."""
        query = build_stock_bars_query(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            session_filter=session_filter,
        )
        frame = self._client.query_df(query)
        timestamp_coverage = summarize_timestamp_coverage(frame=frame)

        if not frame.empty:
            min_ts = str(frame["ts"].min())
            max_ts = str(frame["ts"].max())
        else:
            min_ts = ""
            max_ts = ""

        metadata: dict[str, object] = {
            "source_database": "firstrate",
            "source_table": "stocks",
            "query_filters": {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "session": session_filter,
            },
            "symbol_set": [symbol],
            "date_range": {"start_date": start_date, "end_date": end_date},
            "timezone": "UTC",
            "timestamp_bounds": {"min_ts": min_ts, "max_ts": max_ts},
            "timestamp_coverage": timestamp_coverage,
            "expected_session_scope": session_filter,
            "intended_use": "modeling_and_demonstration",
            "portable_parquet_source": True,
            "portable_source_note": PORTABLE_SOURCE_NOTE,
            "extraction_sql": query,
        }
        return ClickHouseDataset(frame=frame, metadata=metadata)
