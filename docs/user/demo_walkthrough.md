# Interview Demo Walkthrough

Use this script to present the project in a clean research-first flow.

## Fastest CLI Demo

```bash
uv run optimal-execution
```

What to highlight:

- one offline research summary line (walk-forward metrics),
- one execution-bridge interpretation line,
- one single-order comparison section,
- one cross-day schedule comparison section with explicit units.

## Fastest Notebook Demo

Build and execute deterministically:

```bash
uv run python scripts/build_offline_research_pipeline.py
uv run python -m nbconvert --to notebook --execute notebooks/offline_research_pipeline.ipynb --output /tmp/offline_research_pipeline.executed.ipynb
```

Then open the notebook and walk through:

1. offline contract (`data/raw/` only),
2. schema/session coverage checks,
3. non-overlapping opening features and remaining-window target construction,
4. feature engineering and leakage-safe lagging,
5. walk-forward model evaluation,
6. forecast-to-execution bridge.

## How To Explain The Core Choices

- **Why realized variance?** It is directly observable from intraday returns and
  aligns with volatility-aware execution urgency.
- **Why simple models?** Persistence, rolling means, and linear regression are
  easy to defend, debug, and interpret in interviews.
- **Why walk-forward validation?** It preserves chronological order and avoids
  lookahead leakage.
- **Why bridge to execution?** The square root of forecast variance has the
  volatility units expected by Almgren-Chriss. Execution starts only after the
  forecast becomes available.

## Honest Limitations To Acknowledge

- tracked payload covers only the first hour, not full regular-hours coverage,
- deterministic sample construction makes forecast results unrealistically smooth,
- the VWAP comparison uses realized future volume as an oracle benchmark,
- impact model remains deliberately simple,
- no order-book dynamics or venue routing,
- no hyperparameter search or deep-learning models.

## Key Interview Message

This project emphasizes research discipline and software reproducibility:
offline-first raw data, transparent modeling choices, leakage-safe evaluation,
and one clear path from forecast insight to execution interpretation.
