# GUIDE_calibration.md

## Part 1: Conceptual explanation

This folder transforms intraday bars into scheduling inputs.

Outputs:

1. **Market state**
   - Average daily volume (shares)
   - Daily volatility (decimal)
   - Spread proxy (basis points)
2. **Volume profile**
   - Intraday bucket weights that sum to 1

`calibrate_market_state` now supports one bridge hook:

- `override_daily_volatility`: optional externally forecast daily volatility
  passed in from the research pipeline.

If override is not provided, volatility is estimated from bar-level log returns.

## Part 2: Code reference

- `market_state.py`
  - constants: `MINUTES_PER_TRADING_DAY`, `DEFAULT_SPREAD_BPS`
  - entrypoint: `calibrate_market_state`
- `volume_profile.py`
  - entrypoint: `estimate_volume_profile`
- `__init__.py`: package marker docstring.

Cross-folder dependencies:

- writes `MarketState` from `src/optimal_execution_engine/types.py`.
- volume-profile output is consumed by
  `src/optimal_execution_engine/schedules/vwap.py`.

## Part 3: Short journal

- 2026-04-20: Added documented volatility override path so forecast research can
  influence Almgren-Chriss urgency without changing baseline calibration APIs.
