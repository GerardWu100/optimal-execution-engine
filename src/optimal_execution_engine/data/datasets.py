"""Typed dataset catalog for offline cache and optional refresh boundaries.

The catalog defines supported demo datasets. A loader can resolve one entry and
then fetch or validate exactly that dataset without hard-coded conditionals.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Declarative dataset definition for cache and refresh operations.

    Parameters
    ----------
    dataset_name
        Stable identifier used as the cache filename stem.
    symbol
        Ticker symbol expected for this dataset.
    start_date
        Inclusive extraction start date in YYYY-MM-DD format.
    end_date
        Inclusive extraction end date in YYYY-MM-DD format.
    description
        Human-readable explanation of why this dataset exists.
    bar_frequency
        Bar sampling frequency label for metadata provenance.
    session_filter
        Session scope label, for example ``regular-hours``.
    expected_session_scope
        Human-readable session coverage label used in metadata sidecars.
    intended_use
        Whether the dataset is for modeling, demonstration, or both.
    """

    dataset_name: str
    symbol: str
    start_date: str
    end_date: str
    description: str
    bar_frequency: str
    session_filter: str
    expected_session_scope: str
    intended_use: str


def _opening_window_dataset(
    dataset_name: str,
    symbol: str,
    description: str,
) -> DatasetSpec:
    """Build one opening-window dataset spec with shared demo date boundaries."""
    return DatasetSpec(
        dataset_name=dataset_name,
        symbol=symbol,
        start_date="2025-11-03",
        end_date="2026-01-16",
        description=description,
        bar_frequency="5min",
        session_filter="opening-window-only",
        expected_session_scope="opening-window-only",
        intended_use="modeling_and_demonstration",
    )


DATASET_SPECS: dict[str, DatasetSpec] = {
    "sample_intraday_bars": _opening_window_dataset(
        dataset_name="sample_intraday_bars",
        symbol="AAPL",
        description="AAPL regular-hours intraday bars for the default interview demo.",
    ),
    "sample_intraday_bars_msft": _opening_window_dataset(
        dataset_name="sample_intraday_bars_msft",
        symbol="MSFT",
        description="MSFT regular-hours intraday bars for cross-symbol comparison.",
    ),
    "sample_intraday_bars_nvda": _opening_window_dataset(
        dataset_name="sample_intraday_bars_nvda",
        symbol="NVDA",
        description="NVDA regular-hours intraday bars for cross-symbol comparison.",
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Resolve one dataset spec or raise a clear error for unknown names.

    Parameters
    ----------
    dataset_name
        Requested dataset identifier.

    Returns
    -------
    DatasetSpec
        Typed catalog entry for the requested dataset.
    """
    if dataset_name not in DATASET_SPECS:
        known_names = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. Known datasets: {known_names}."
        )

    return DATASET_SPECS[dataset_name]
