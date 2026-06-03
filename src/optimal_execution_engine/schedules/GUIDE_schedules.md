# GUIDE_schedules.md

## Part 1: Conceptual explanation

This folder converts one parent order into executable share slices.

Definitions:

- $Q$: total parent-order shares,
- $N$: number of execution slices,
- $x_i$: shares in slice $i$, with $\sum_{i=1}^{N} x_i = Q$.

Implemented schedules:

- **TWAP**: equal-size slices plus final residual reconciliation.
- **VWAP-style**: shares proportional to expected volume weights.
- **Almgren-Chriss**: risk-aware inventory trajectory based on
  `risk_aversion * daily_volatility`.

When CLI passes forecast volatility through calibration, Almgren-Chriss urgency
changes through `daily_volatility` while schedule code remains unchanged.

## Part 2: Code reference

- `twap.py`: `build_twap_schedule(order, slice_count)`.
- `vwap.py`: `build_vwap_schedule(order, volume_profile, bucket_column)`.
- `almgren_chriss.py`:
  `build_almgren_chriss_schedule(order, market_state, slice_count)`.
- `__init__.py`: package marker docstring.

Cross-folder dependencies:

- all schedules consume `ParentOrder` from
  `src/optimal_execution_engine/types.py`.
- Almgren-Chriss consumes `MarketState` from
  `src/optimal_execution_engine/types.py`.
- VWAP consumes bucket weights from
  `src/optimal_execution_engine/calibration/volume_profile.py`.

## Part 3: Short journal

- 2026-04-20: Updated guide paths for namespace package and clarified that
  volatility forecast influence enters through calibration rather than schedule
  API changes.
