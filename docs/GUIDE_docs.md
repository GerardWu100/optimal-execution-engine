# GUIDE_docs.md

## Part 1: Conceptual explanation

This folder stores non-code documentation that supports offline reproducibility
and interview communication.

Main roles:

- **reference**: data-source and extraction contract references,
- **user**: demo and interview walkthrough materials.

## Part 2: Code reference

- `reference/clickhouse_exploration_queries.sql`: read-only queries for source
  schema checks, session filtering, bar spacing, and extraction slices.
- `user/demo_walkthrough.md`: concise script for explaining offline notebook,
  model evaluation, and execution bridge in interviews.

Recommended reading order:

1. `user/demo_walkthrough.md`
2. `reference/clickhouse_exploration_queries.sql`

## Part 3: Short journal

- 2026-04-20: Updated docs guide for the offline research-first architecture,
  notebook replacement, and `data/raw/` boundary.
- 2026-05-20: Removed historical `superpowers/plans/` implementation notes.
