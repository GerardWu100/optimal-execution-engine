# GUIDE_data.md

## Part 1: Conceptual explanation

This folder defines the offline raw-data boundary for the project.

Default behavior:

- load from local `data/raw/` Parquet + metadata sidecar,
- fail clearly when raw files are invalid and no refresh client is provided,
- only use ClickHouse when explicitly requested for one-time refresh.

Metadata validation now checks provenance plus portability and modeling intent:

- bar-frequency label,
- expected session-scope label,
- timestamp-coverage diagnostics,
- intended-use classification,
- explicit portable-Parquet source note.

## Part 2: Code reference

- `datasets.py`: typed dataset catalog (`DatasetSpec`) including session scope and
  intended-use fields.
- `loaders.py`: `load_market_data` cache-first decision tree and metadata
  enrichment.
- `cache.py`: `validate_cache_pair` and `write_cache_dataset` with tightened
  metadata contract.
- `clickhouse.py`: extraction query builder, session filtering, and timestamp
  coverage diagnostics.
- `contracts.py`: shared data contracts (`CacheValidationResult`).
- `__init__.py`: package marker docstring.

Recommended reading order:

1. `loaders.py`
2. `cache.py`
3. `datasets.py`
4. `clickhouse.py`

## Part 3: Short journal

- 2026-04-20: Updated data guide for the `data/raw/` boundary and tightened
  metadata checks used to defend offline portability claims.
