"""Tests for loader fallback behavior."""

import json
from pathlib import Path

import pandas as pd
import pytest

from optimal_execution_engine.data.clickhouse import ClickHouseDataset
from optimal_execution_engine.data.datasets import get_dataset_spec
from optimal_execution_engine.data.loaders import load_market_data


EXPECTED_PROVENANCE_KEYS: set[str] = {
    "dataset_name",
    "row_count",
    "created_at_utc",
    "source_database",
    "source_table",
    "query_filters",
    "symbol_set",
    "date_range",
    "timezone",
    "timestamp_bounds",
    "dataset_description",
    "bar_frequency",
    "session_filter",
    "expected_session_scope",
    "timestamp_coverage",
    "intended_use",
    "portable_parquet_source",
    "portable_source_note",
    "extracted_symbol",
    "trading_date_count",
    "file_size_bytes",
}


class RecordingClient:
    """Record ClickHouse refresh calls for loader behavior assertions."""

    def __init__(self) -> None:
        """Initialize call recorder."""
        self.calls: list[tuple[str, str, str, str]] = []

    def fetch_stock_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        session_filter: str,
    ) -> ClickHouseDataset:
        """Return deterministic bars and metadata for refresh path tests."""
        self.calls.append((symbol, start_date, end_date, session_filter))

        frame = pd.DataFrame(
            {
                "symbol": [symbol, symbol],
                "ts": [
                    "2025-11-03T14:30:00Z",
                    "2025-11-04T14:30:00Z",
                ],
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.2, 101.1],
                "volume": [10_000, 12_000],
            }
        )

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
            "timestamp_bounds": {
                "min_ts": "2025-11-03T14:30:00Z",
                "max_ts": "2025-11-04T14:30:00Z",
            },
            "timestamp_coverage": {
                "first_bar_utc": "2025-11-03T14:30:00+00:00",
                "last_bar_utc": "2025-11-04T14:30:00+00:00",
                "trade_dates": 2,
                "rows_per_day_min": 1,
                "rows_per_day_max": 1,
                "unique_bar_spacing_minutes": [1440.0],
            },
            "expected_session_scope": session_filter,
            "intended_use": "modeling_and_demonstration",
            "portable_parquet_source": True,
            "portable_source_note": "Offline portable source Parquet.",
        }
        return ClickHouseDataset(frame=frame, metadata=metadata)


class FailingClient:
    """Raise if refresh is attempted, used for cache-bypass checks."""

    def fetch_stock_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        session_filter: str,
    ) -> ClickHouseDataset:
        """Fail loudly because cache reads should not call this path."""
        raise AssertionError(
            "ClickHouse refresh should not be called when cache is valid."
        )


def _valid_metadata() -> dict[str, object]:
    """Return valid cache metadata for loader tests."""
    return {
        "dataset_name": "bars",
        "row_count": 1,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source_database": "firstrate",
        "source_table": "stocks",
        "query_filters": {"symbol": "AAPL"},
        "symbol_set": ["AAPL"],
        "date_range": {"start_date": "2025-12-15", "end_date": "2026-01-05"},
        "timezone": "UTC",
        "timestamp_bounds": {
            "min_ts": "2025-12-15T14:30:00Z",
            "max_ts": "2026-01-05T20:00:00Z",
        },
        "dataset_description": "Cached bars for test fixtures.",
        "bar_frequency": "5min",
        "session_filter": "regular-hours",
        "expected_session_scope": "regular-hours",
        "timestamp_coverage": {
            "first_bar_utc": "2025-12-15T14:30:00+00:00",
            "last_bar_utc": "2026-01-05T20:00:00+00:00",
            "trade_dates": 1,
            "rows_per_day_min": 1,
            "rows_per_day_max": 1,
            "unique_bar_spacing_minutes": [5.0],
        },
        "intended_use": "modeling_and_demonstration",
        "portable_parquet_source": True,
        "portable_source_note": "Offline portable source Parquet.",
        "extracted_symbol": "AAPL",
        "trading_date_count": 1,
        "file_size_bytes": 1234,
    }


def test_dataset_catalog_resolves_known_dataset_name() -> None:
    """Known dataset IDs should resolve to typed symbol and date settings."""
    spec = get_dataset_spec(dataset_name="sample_intraday_bars_msft")

    assert spec.symbol == "MSFT"
    assert spec.start_date == "2025-11-03"
    assert spec.end_date == "2026-01-16"
    assert spec.expected_session_scope == "opening-window-only"


def test_loader_fails_for_unknown_dataset_name(tmp_path: Path) -> None:
    """Unknown dataset IDs should fail before any refresh behavior."""
    with pytest.raises(ValueError) as exc_info:
        load_market_data(
            cache_dir=tmp_path,
            dataset_name="unknown_dataset",
            clickhouse_client=RecordingClient(),
        )

    assert "Unknown dataset_name" in str(exc_info.value)


def test_loader_fails_clearly_when_cache_invalid_and_clickhouse_disabled(
    tmp_path: Path,
) -> None:
    """The error should name the missing files and directory."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_market_data(
            cache_dir=tmp_path,
            dataset_name="sample_intraday_bars",
            clickhouse_client=None,
        )

    message = str(exc_info.value)
    assert "Offline raw-data boundary invalid" in message
    assert "sample_intraday_bars.parquet" in message
    assert "sample_intraday_bars.meta.json" in message
    assert str(tmp_path) in message


def test_loader_uses_cache_without_db_call(tmp_path: Path) -> None:
    """A valid cache should bypass ClickHouse entirely."""
    parquet_path = tmp_path / "sample_intraday_bars.parquet"
    metadata_path = tmp_path / "sample_intraday_bars.meta.json"
    pd.DataFrame({"close": [100.0]}).to_parquet(parquet_path)
    metadata_path.write_text(json.dumps(_valid_metadata()), encoding="utf-8")

    frame = load_market_data(
        cache_dir=tmp_path,
        dataset_name="sample_intraday_bars",
        clickhouse_client=FailingClient(),
    )

    assert list(frame["close"]) == [100.0]


def test_loader_refresh_uses_dataset_spec(tmp_path: Path) -> None:
    """Refresh behavior should use the symbol and date range from catalog spec."""
    client = RecordingClient()

    frame = load_market_data(
        cache_dir=tmp_path,
        dataset_name="sample_intraday_bars_msft",
        clickhouse_client=client,
    )

    assert set(frame["symbol"]) == {"MSFT"}
    assert client.calls == [("MSFT", "2025-11-03", "2026-01-16", "opening-window-only")]


def test_loader_refresh_writes_enriched_metadata_sidecar(tmp_path: Path) -> None:
    """Refresh path should persist cache files with portable provenance fields."""
    load_market_data(
        cache_dir=tmp_path,
        dataset_name="sample_intraday_bars_nvda",
        clickhouse_client=RecordingClient(),
    )

    metadata_path = tmp_path / "sample_intraday_bars_nvda.meta.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert EXPECTED_PROVENANCE_KEYS.issubset(metadata)
    assert metadata["dataset_description"]
    assert metadata["trading_date_count"] == 2
