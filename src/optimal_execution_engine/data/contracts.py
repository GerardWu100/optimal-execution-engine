"""Shared data contracts used by loaders and cache validators."""

from dataclasses import dataclass
from pathlib import Path


PORTABLE_SOURCE_NOTE: str = (
    "This Parquet file is the portable offline source for local research and "
    "CLI demonstrations."
)


@dataclass(slots=True)
class CacheValidationResult:
    """Outcome of validating a cached dataset and metadata sidecar.

    Parameters
    ----------
    is_valid
        Whether both files and required metadata checks passed.
    message
        Human-readable validation outcome details.
    parquet_path
        Absolute or relative path to the dataset Parquet file.
    metadata_path
        Absolute or relative path to the metadata JSON sidecar.
    """

    is_valid: bool
    message: str
    parquet_path: Path
    metadata_path: Path
