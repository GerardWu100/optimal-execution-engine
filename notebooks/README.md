# Notebooks Guide

## Purpose

`offline_research_pipeline.ipynb` is the main teaching notebook for this project.
It demonstrates the full offline path from local raw Parquet files to realized
variance modeling, walk-forward evaluation, and a compact execution bridge.

## Deterministic Build

Regenerate the notebook from Python to keep deterministic cell ordering:

```bash
uv run python scripts/build_offline_research_pipeline.py
```

The notebook names the project-specific `optimal-execution-engine` kernel. If it
is absent after setup, register the current virtual environment once:

```bash
uv run python -m ipykernel install --user --name optimal-execution-engine --display-name "Python (optimal-execution-engine)"
```

## Execute Top-To-Bottom

Use module execution for reliable environments:

```bash
uv run python -m nbconvert --to notebook --execute notebooks/offline_research_pipeline.ipynb --output /tmp/offline_research_pipeline.executed.ipynb
```

## Offline Contract

- The notebook reads only `data/raw/*.parquet` files.
- The notebook does not instantiate `ClickHouseMarketDataClient`.
- ClickHouse is only for optional out-of-band raw data refresh.

## Allowed Notebook Outputs

- CSV summaries in `outputs/offline_research/`
- Parquet prediction artifacts in `outputs/offline_research/`
- PNG figures in `outputs/offline_research/`

No HTML report generation is part of this notebook workflow.
