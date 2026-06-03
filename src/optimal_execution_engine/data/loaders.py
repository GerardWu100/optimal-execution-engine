"""Market data loading with offline-first cache semantics."""

from pathlib import Path

import pandas as pd

from optimal_execution_engine.data.cache import validate_cache_pair, write_cache_dataset
from optimal_execution_engine.data.clickhouse import ClickHouseMarketDataClient
from optimal_execution_engine.data.contracts import PORTABLE_SOURCE_NOTE
from optimal_execution_engine.data.datasets import get_dataset_spec


def load_market_data(
    cache_dir: Path,
    dataset_name: str,
    clickhouse_client: ClickHouseMarketDataClient | None,
) -> pd.DataFrame:
    """Load from cache first, then refresh from ClickHouse if available.

    Parameters
    ----------
    cache_dir
        Directory where `<dataset>.parquet` and `<dataset>.meta.json` live.
    dataset_name
        Cache dataset identifier.
    clickhouse_client
        Optional ClickHouse client used only when cache validation fails.

    Returns
    -------
    pd.DataFrame
        Market data frame for calibration and simulation.
    """
    # Resolve catalog entry first so both cache and refresh paths share one source of truth.
    dataset_spec = get_dataset_spec(dataset_name=dataset_name)

    parquet_path = cache_dir / f"{dataset_name}.parquet"
    validation = validate_cache_pair(parquet_path=parquet_path)

    # Fast path: valid cache returns immediately with no database dependency.
    if validation.is_valid:
        return pd.read_parquet(validation.parquet_path)

    # No refresh client means the caller explicitly requested offline-only behavior.
    if clickhouse_client is None:
        raise FileNotFoundError(
            f"Offline raw-data boundary invalid for dataset '{dataset_name}'. "
            f"Required files: {validation.parquet_path.name}, "
            f"{validation.metadata_path.name}. Expected directory: {cache_dir}"
        )

    # Refresh path: query the exact symbol/date/session tuple from the dataset catalog.
    fetched_dataset = clickhouse_client.fetch_stock_bars(
        symbol=dataset_spec.symbol,
        start_date=dataset_spec.start_date,
        end_date=dataset_spec.end_date,
        session_filter=dataset_spec.session_filter,
    )

    refreshed_metadata = dict(fetched_dataset.metadata)
    refreshed_metadata.update(
        {
            "dataset_description": dataset_spec.description,
            "bar_frequency": dataset_spec.bar_frequency,
            "session_filter": dataset_spec.session_filter,
            "expected_session_scope": dataset_spec.expected_session_scope,
            "intended_use": dataset_spec.intended_use,
            "portable_parquet_source": True,
            "portable_source_note": PORTABLE_SOURCE_NOTE,
        }
    )

    write_cache_dataset(
        parquet_path=parquet_path,
        frame=fetched_dataset.frame,
        metadata=refreshed_metadata,
    )
    return fetched_dataset.frame
