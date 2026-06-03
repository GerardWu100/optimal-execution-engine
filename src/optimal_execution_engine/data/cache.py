"""Offline-first Parquet cache validation and persistence helpers."""

from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd

from optimal_execution_engine.data.contracts import (
    PORTABLE_SOURCE_NOTE,
    CacheValidationResult,
)
from optimal_execution_engine.data.coverage import build_timestamp_coverage


REQUIRED_METADATA_KEYS: set[str] = {
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

ALLOWED_INTENDED_USE_VALUES: set[str] = {
    "modeling",
    "demonstration",
    "modeling_and_demonstration",
}


def _validation_failure(
    message: str,
    parquet_path: Path,
    metadata_path: Path,
) -> CacheValidationResult:
    """Build one failed cache-validation result with stable path fields."""
    return CacheValidationResult(False, message, parquet_path, metadata_path)


def validate_cache_pair(parquet_path: Path) -> CacheValidationResult:
    """Validate that a Parquet cache and JSON sidecar both exist and match schema.

    Parameters
    ----------
    parquet_path
        Path to the expected Parquet dataset.

    Returns
    -------
    CacheValidationResult
        Validation outcome with actionable details.
    """
    metadata_path = parquet_path.with_suffix(".meta.json")

    if not parquet_path.exists():
        return _validation_failure(
            f"Missing {parquet_path.name}", parquet_path, metadata_path
        )

    if not metadata_path.exists():
        return _validation_failure(
            f"Missing {metadata_path.name}", parquet_path, metadata_path
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _validation_failure(
            f"Invalid JSON in {metadata_path.name}", parquet_path, metadata_path
        )

    if not REQUIRED_METADATA_KEYS.issubset(metadata):
        return _validation_failure(
            f"Missing metadata keys in {metadata_path.name}", parquet_path, metadata_path
        )

    if not isinstance(metadata.get("row_count"), int) or metadata["row_count"] < 0:
        return _validation_failure(
            f"Invalid row_count in {metadata_path.name}", parquet_path, metadata_path
        )

    if (
        not isinstance(metadata.get("bar_frequency"), str)
        or not metadata["bar_frequency"].strip()
    ):
        return _validation_failure(
            f"Invalid bar_frequency in {metadata_path.name}", parquet_path, metadata_path
        )

    if (
        not isinstance(metadata.get("expected_session_scope"), str)
        or not metadata["expected_session_scope"].strip()
    ):
        return _validation_failure(
            f"Invalid expected_session_scope in {metadata_path.name}",
            parquet_path,
            metadata_path,
        )

    timestamp_coverage = metadata.get("timestamp_coverage")
    if not isinstance(timestamp_coverage, dict):
        return _validation_failure(
            f"Invalid timestamp_coverage in {metadata_path.name}",
            parquet_path,
            metadata_path,
        )

    coverage_keys = {
        "first_bar_utc",
        "last_bar_utc",
        "trade_dates",
        "rows_per_day_min",
        "rows_per_day_max",
        "unique_bar_spacing_minutes",
    }
    if not coverage_keys.issubset(timestamp_coverage):
        return _validation_failure(
            f"Missing timestamp_coverage fields in {metadata_path.name}",
            parquet_path,
            metadata_path,
        )

    intended_use = metadata.get("intended_use")
    if intended_use not in ALLOWED_INTENDED_USE_VALUES:
        return _validation_failure(
            f"Invalid intended_use in {metadata_path.name}", parquet_path, metadata_path
        )

    if metadata.get("portable_parquet_source") is not True:
        return _validation_failure(
            f"portable_parquet_source must be true in {metadata_path.name}",
            parquet_path,
            metadata_path,
        )

    if (
        not isinstance(metadata.get("portable_source_note"), str)
        or not metadata["portable_source_note"].strip()
    ):
        return _validation_failure(
            f"Invalid portable_source_note in {metadata_path.name}",
            parquet_path,
            metadata_path,
        )

    return CacheValidationResult(True, "Cache is valid.", parquet_path, metadata_path)


def write_cache_dataset(
    parquet_path: Path, frame: pd.DataFrame, metadata: dict[str, object]
) -> None:
    """Persist a refreshed dataset and metadata sidecar atomically enough for demos.

    Parameters
    ----------
    parquet_path
        Target Parquet file path.
    frame
        Dataset to persist.
    metadata
        Metadata payload, augmented with standard fields before write.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = parquet_path.with_suffix(".meta.json")

    frame.to_parquet(parquet_path, index=False)

    normalized_metadata = dict(metadata)
    normalized_metadata["dataset_name"] = parquet_path.stem
    normalized_metadata["row_count"] = int(len(frame))

    # Keep the description explicit so cache users understand dataset intent.
    normalized_metadata.setdefault("dataset_description", "")
    normalized_metadata.setdefault("bar_frequency", "unknown")
    normalized_metadata.setdefault("session_filter", "unknown")
    normalized_metadata.setdefault(
        "expected_session_scope",
        str(normalized_metadata.get("session_filter", "unknown")),
    )
    normalized_metadata.setdefault("intended_use", "modeling_and_demonstration")
    normalized_metadata["portable_parquet_source"] = True
    normalized_metadata["portable_source_note"] = PORTABLE_SOURCE_NOTE

    # Resolve the extracted symbol from metadata first, then frame fallback.
    symbol_set = normalized_metadata.get("symbol_set", [])
    extracted_symbol = ""
    if isinstance(symbol_set, list) and symbol_set:
        extracted_symbol = str(symbol_set[0])
    elif "symbol" in frame.columns and not frame.empty:
        extracted_symbol = str(frame["symbol"].iloc[0])
    normalized_metadata.setdefault("extracted_symbol", extracted_symbol)

    # Compute the exact number of distinct trading dates in the persisted frame.
    trading_date_count = 0
    if "ts" in frame.columns and not frame.empty:
        timestamp_series = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        trading_date_count = int(timestamp_series.dt.date.nunique())
    normalized_metadata["trading_date_count"] = int(trading_date_count)
    normalized_metadata["timestamp_coverage"] = build_timestamp_coverage(frame=frame)

    normalized_metadata.setdefault(
        "created_at_utc",
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )

    # Include byte size after write so cache portability is inspectable quickly.
    normalized_metadata["file_size_bytes"] = int(parquet_path.stat().st_size)

    metadata_path.write_text(
        json.dumps(normalized_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
