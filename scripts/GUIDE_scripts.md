# GUIDE_scripts.md

## Part 1: Conceptual explanation

This folder holds thin, runnable entrypoints that orchestrate package code but
are not part of the importable library. Scripts should parse inputs, call
`src/optimal_execution_engine/`, and write artifacts to the appropriate
non-source folders (`notebooks/`, `outputs/`, `data/raw/`).

The primary script here rebuilds the offline teaching notebook deterministically
so cell order and IDs stay stable across regenerations.

## Part 2: Code reference

- `build_offline_research_pipeline.py`: assembles `notebooks/offline_research_pipeline.ipynb`
  from Python cell definitions; imports only third-party `nbformat` plus strings
  that reference package modules inside generated notebook cells.

Recommended command:

```bash
uv run python scripts/build_offline_research_pipeline.py
```

## Part 3: Short journal

- 2026-05-20: Moved notebook builder from `notebooks/` to `scripts/` per project layout.
