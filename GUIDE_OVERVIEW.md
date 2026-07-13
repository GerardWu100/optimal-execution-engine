# GUIDE_OVERVIEW.md

## Project Snapshot

```text
optimal-execution-engine/
|- config.toml
|- data/
|  `- raw/
|- docs/
|  |- reference/
|  `- user/
|- logs/
|- notebooks/
|  |- offline_research_pipeline.ipynb
|  `- README.md
|- outputs/
|- scripts/
|  `- build_offline_research_pipeline.py
|- src/
|  `- optimal_execution_engine/
|     |- calibration/
|     |- data/
|     |- reporting/
|     |- research/
|     |- schedules/
|     |- simulator/
|     |- cli.py
|     |- config.py
|     |- types.py
|     `- version.py
`- tests/
```

## What This System Does

This system is an offline research-first volatility forecasting and execution
demo. It starts with tracked local raw Parquet files, builds realized-variance
targets and features, evaluates simple forecasting models with walk-forward
splits, and then shows one small bridge into execution urgency.

## End-to-End Data Flow

1. **Load raw bars** from `data/raw/` only.
2. **Engineer features** from the first six bars, all known by 09:55.
3. **Build the target** from squared log returns ending from 10:00 onward.
4. **Fit/evaluate models** with persistence, rolling-mean, and linear baselines
   under walk-forward splits.
5. **Bridge to execution** by square-rooting forecast variance, passing the
   resulting volatility into Almgren-Chriss, and using only post-cutoff bars.
6. **Present outputs** in the offline notebook and concise CLI text output.

## Domain Logic and Conventions

- **Log return**: $r_t = \log(P_t / P_{t-1})$ where $P_t$ is close price.
- **Opening variance**: $RV_d^{open}=\sum_{t<m}r_{d,t}^2$, where $m=6$.
- **Forecast target**: $RV_d^{remaining}=\sum_{t\ge m}r_{d,t}^2$.
- **Unit conversion**: $\widehat{\sigma}_d=\sqrt{\widehat{RV}_d^{remaining}}$.
- **Execution impact**: $impact_{bps} = 2.0 + 25.0 \times volume\_share$.
- **Cost convention**: implementation shortfall in dollars and basis points.
- **Validation protocol**: walk-forward splits to avoid lookahead leakage.

## Architecture Choices and Tradeoffs

- **Offline portability over live dependency**: default runtime never requires
  ClickHouse.
- **Interpretability over complexity**: simple baselines and explicit linear
  model instead of heavy machine-learning frameworks.
- **Compact execution bridge**: one forecast input influences one execution
  parameter, avoiding platform scope creep.

## Boundaries and Limitations

- Tracked demo payload covers only the first hour, not a full regular-hours day.
- Its deterministic construction makes opening and later returns almost
  perfectly collinear, so model errors do not measure real market skill.
- The VWAP schedule uses realized later-window volumes and is an oracle benchmark.
- No order-book microstructure or queue-position modeling.
- No multi-asset coupling or portfolio-level execution optimization.
- No hyperparameter search or deep-learning models.

## Where To Drill Down Next

- Root map and reading order: `GUIDE_ROOT.md`
- Detailed repository guide: `GUIDE_PROJECT.md`
- Package guide: `src/GUIDE_src.md`
- Scripts guide: `scripts/GUIDE_scripts.md`
- Notebook workflow: `notebooks/README.md`
- Docs map: `docs/GUIDE_docs.md`
- Test strategy: `tests/GUIDE_tests.md`
