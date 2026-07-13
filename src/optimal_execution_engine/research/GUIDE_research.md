# GUIDE_research.md

## Part 1: Conceptual explanation

This folder is the research backbone of the project.

Pipeline stages:

1. compute intraday log returns,
2. engineer predictors from a six-bar information window,
3. aggregate the remaining bars into a non-overlapping variance target,
4. build modeling tables,
5. fit simple baselines and linear models,
6. evaluate with walk-forward splits and forecast metrics.

The design is intentionally simple and explainable for interview settings.

## Part 2: Code reference

- `realized_variance.py`
  - `compute_log_returns`
  - `compute_daily_realized_variance`
  - `compute_opening_window_realized_variance`
  - `compute_remaining_window_realized_variance`
- `features.py`
  - opening-feature construction
  - lag/rolling feature table construction
- `dataset.py`
  - assembly of final modeling dataset from bars or precomputed inputs
- `modeling.py`
  - persistence and rolling baselines
  - explicit least-squares linear model fit/predict
- `evaluation.py`
  - walk-forward split construction
  - MAE, RMSE, and QLIKE metrics

## Part 3: Short journal

- 2026-04-20: Added research backbone modules and tests to make the project
  narrative centered on offline realized-variance forecasting.
- 2026-07-13: Changed the model target to post-cutoff variance and removed the
  future total-volume denominator from same-day features.
