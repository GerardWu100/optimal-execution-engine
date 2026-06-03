"""Tests for offline-first cache validity rules."""

import json
from pathlib import Path

import pandas as pd

from optimal_execution_engine.data.cache import validate_cache_pair, write_cache_dataset


EXPECTED_METADATA_KEYS: set[str] = {
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


def _valid_metadata() -> dict[str, object]:
    """Return a minimal metadata payload satisfying required cache keys."""
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
        "dataset_description": "Cached bars for cache-validity tests.",
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
        "file_size_bytes": 2048,
    }


def test_cache_pair_requires_parquet_and_metadata(tmp_path: Path) -> None:
    """A Parquet file without metadata is invalid."""
    parquet_path = tmp_path / "bars.parquet"
    pd.DataFrame({"close": [100.0]}).to_parquet(parquet_path)

    result = validate_cache_pair(parquet_path=parquet_path)

    assert result.is_valid is False
    assert "bars.meta.json" in result.message


def test_cache_pair_rejects_missing_required_metadata_keys(tmp_path: Path) -> None:
    """Metadata without required keys should be invalid."""
    parquet_path = tmp_path / "bars.parquet"
    metadata_path = tmp_path / "bars.meta.json"

    pd.DataFrame({"close": [100.0]}).to_parquet(parquet_path)
    metadata_path.write_text('{"dataset_name": "bars"}', encoding="utf-8")

    result = validate_cache_pair(parquet_path=parquet_path)

    assert result.is_valid is False
    assert "Missing metadata keys" in result.message


def test_cache_pair_accepts_valid_parquet_and_metadata(tmp_path: Path) -> None:
    """A complete metadata sidecar should produce a valid cache result."""
    parquet_path = tmp_path / "bars.parquet"
    metadata_path = tmp_path / "bars.meta.json"

    pd.DataFrame({"close": [100.0]}).to_parquet(parquet_path)
    metadata_path.write_text(json.dumps(_valid_metadata()), encoding="utf-8")

    result = validate_cache_pair(parquet_path=parquet_path)

    assert result.is_valid is True


def test_write_cache_dataset_adds_enriched_provenance_fields(tmp_path: Path) -> None:
    """Write helper should enrich metadata with stable portability fields."""
    parquet_path = tmp_path / "bars.parquet"
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts": ["2025-12-15T14:30:00Z", "2025-12-16T14:30:00Z"],
            "close": [100.0, 101.0],
            "volume": [10_000, 11_000],
        }
    )

    metadata = {
        "source_database": "firstrate",
        "source_table": "stocks",
        "query_filters": {
            "symbol": "AAPL",
            "start_date": "2025-12-15",
            "end_date": "2025-12-16",
            "session": "regular-hours",
        },
        "symbol_set": ["AAPL"],
        "date_range": {
            "start_date": "2025-12-15",
            "end_date": "2025-12-16",
        },
        "timezone": "UTC",
        "timestamp_bounds": {
            "min_ts": "2025-12-15T14:30:00Z",
            "max_ts": "2025-12-16T14:30:00Z",
        },
        "dataset_description": "AAPL sample bars for metadata enrichment test.",
        "bar_frequency": "5min",
        "session_filter": "regular-hours",
        "expected_session_scope": "regular-hours",
        "extracted_symbol": "AAPL",
        "trading_date_count": 2,
        "timestamp_coverage": {
            "first_bar_utc": "2025-12-15T14:30:00+00:00",
            "last_bar_utc": "2025-12-16T14:30:00+00:00",
            "trade_dates": 2,
            "rows_per_day_min": 1,
            "rows_per_day_max": 1,
            "unique_bar_spacing_minutes": [1440.0],
        },
        "intended_use": "modeling_and_demonstration",
        "portable_parquet_source": True,
        "portable_source_note": "Offline portable source Parquet.",
    }

    write_cache_dataset(parquet_path=parquet_path, frame=frame, metadata=metadata)

    result = validate_cache_pair(parquet_path=parquet_path)
    assert result.is_valid is True

    metadata_path = parquet_path.with_suffix(".meta.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert EXPECTED_METADATA_KEYS.issubset(payload)
    assert payload["dataset_name"] == "bars"
    assert payload["row_count"] == 2
    assert payload["trading_date_count"] == 2
    assert payload["portable_parquet_source"] is True
    assert payload["intended_use"] == "modeling_and_demonstration"
    assert int(payload["file_size_bytes"]) > 0
