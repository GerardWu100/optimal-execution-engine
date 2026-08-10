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

If override is not provided, volatility is estimated from bar-level log returns
and scaled to one trading day by the square root of the number of bars in a day:

    daily_volatility = bar_volatility * sqrt(390 / bar_duration_minutes)

The bar length is inferred from the median gap between consecutive `ts` values,
so a multi-day frame is not skewed by overnight gaps. Frames without usable
timestamps fall back to `DEFAULT_BAR_DURATION_MINUTES`, which mirrors
`execution.bar_duration_minutes` in `config.toml`.

## Part 2: Code reference

- `market_state.py`
  - constants: `MINUTES_PER_TRADING_DAY`, `DEFAULT_SPREAD_BPS`,
    `DEFAULT_BAR_DURATION_MINUTES`
  - helpers: `_estimate_realized_bar_volatility`, `_infer_bar_duration_minutes`,
    `_build_trade_date_labels`
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
- 2026-08-10: Corrected the bar-to-day volatility scaling. It previously used
  sqrt(minutes per day), which is only right for one-minute bars and overstated
  daily volatility by sqrt(5) on the tracked five-minute payload.
