# GUIDE_simulator.md

## Part 1: Conceptual explanation

This folder applies a simple deterministic execution-cost model to schedules.

Per slice:

- volume share: $s_i = q_i / V_i$,
- impact: $impact_i = 2.0 + 25.0 \cdot s_i$ basis points,
- buy fill price: $P^{fill}_i = P^{mid}_i (1 + impact_i / 10000)$,
- sell fill price uses the symmetric downward adjustment.

Symbols:

- $q_i$: executed shares in slice $i$,
- $V_i$: bar volume,
- $P^{mid}_i$: bar close proxy,
- $P^{arrival}$: arrival benchmark.

## Part 2: Code reference

- `execution.py`
  - constants: `BASE_IMPACT_BPS`, `VOLUME_SHARE_IMPACT_COEFFICIENT_BPS`
  - entrypoint: `simulate_schedule(schedule, bars, arrival_price, side)`
- `__init__.py`: package marker docstring.

Important behavior:

- raises when bars are fewer than schedule rows,
- aligns bars to the first `len(schedule)` rows,
- returns mid price, volume share, impact, fill price, arrival notional, dollar
  cost, and basis-point cost.

## Part 3: Short journal

- 2026-04-20: Updated package paths after namespace refactor; simulator logic and
  intent remain intentionally simple and transparent.
