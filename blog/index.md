---
title: "When a Perfect Volatility Forecast Is a Warning"
description: "An audit of an offline realized-variance pipeline, its walk-forward test, and the small execution model that consumes the forecast."
date: 2026-07-13
image: images/cover-optimal-execution.png
categories: ["Quantitative Finance", "Optimal Execution"]
---

A linear volatility model with a Mean Absolute Error (MAE) of $1.99 \times 10^{-14}$ looks extraordinary. On this dataset, it is also a reason to stop.

I built this project as an offline research chain: tracked five-minute bars become a realized-variance target, a small feature set feeds three transparent models, and walk-forward tests pass the forecast into a simplified execution schedule. The plumbing is useful. The perfect-looking forecast is not evidence of skill. It comes from a feature that equals the target on every modeled day.

That distinction matters beyond this project. A chronological split can prevent the training set from seeing future rows while still allowing a test-row feature to contain the answer for that same row.

## The experiment on paper

The tracked AAPL sample contains 55 trading days. Each day has 12 five-minute bars from 09:30 through 10:25 Eastern Time. After the ten-day rolling feature removes the first ten observations, the modeling table has 45 daily rows.

For day $d$ and intraday bar $t$, let $P_{d,t}$ be the close price in dollars. The log return is

$$
r_{d,t}=\log\left(\frac{P_{d,t}}{P_{d,t-1}}\right).
$$

The ratio $P_{d,t}/P_{d,t-1}$ is unitless, so $r_{d,t}$ is also unitless. Squaring and summing the available intraday returns gives the realized-variance proxy for day $d$:

$$
RV_d=\sum_{t=1}^{T_d}r_{d,t}^2,
$$

where $T_d$ is the number of bars available on that day. The first bar has no within-day prior close, so the implementation assigns it a zero contribution. In the tracked sample, $T_d=12$ on every day. This is an opening-hour variance proxy, not full-session daily variance.

The feature table contains the opening-window realized variance, opening return, opening high-low range, opening volume share, yesterday's realized variance, and five-day and ten-day lagged means. The rolling features are shifted before averaging:

```python
merged["lag_1_realized_variance"] = merged.groupby("symbol")[
    "target_realized_variance"
].shift(1)

merged["rolling_5d_realized_variance"] = merged.groupby("symbol")[
    "target_realized_variance"
].transform(lambda values: values.shift(1).rolling(5, min_periods=5).mean())
```

That code handles historical information correctly. Each lagged value is known before the current target. The problem sits in a different feature.

## The feature that contains the answer

Let $m$ be the number of bars in the opening feature window. Its realized variance is

$$
RV_d^{(m)}=\sum_{t=1}^{m}r_{d,t}^2.
$$

The experiment sets $m=12$. The data also contain exactly $T_d=12$ bars per day. Substitution gives

$$
RV_d^{(12)}=\sum_{t=1}^{12}r_{d,t}^2
$$

and

$$
RV_d=\sum_{t=1}^{T_d}r_{d,t}^2=\sum_{t=1}^{12}r_{d,t}^2.
$$

Both sums have the same terms, so

$$
RV_d^{(12)}=RV_d.
$$

The audit confirms exact equality across all 45 modeling rows. The maximum absolute gap is $0.0$ variance units.

![Opening realized variance equals target realized variance on every modeled day](images/01_feature_target_identity.png)

Every point lies on the identity line. A linear regression does not need to discover a stable relationship here. It can copy the target through `opening_realized_variance`.

Suppose $x_{d,j}$ is feature $j$ on day $d$, $\beta_0$ is an intercept, and $\beta_j$ is the coefficient on feature $j$. The fitted model is

$$
\widehat{RV}_d=\beta_0+\sum_{j=1}^{p}\beta_j x_{d,j},
$$

where $p=7$ features and $\widehat{RV}_d$ is predicted realized variance. One feature, say $x_{d,1}$, is already $RV_d$. The least-squares solution can set $\beta_1$ near one, the other coefficients near zero, and reproduce the target up to floating-point error.

The walk-forward procedure still keeps each five-day test block after its 20-day training block. It prevents future-row leakage. It cannot fix contemporaneous leakage inside $x_{d,1}$. These are separate controls:

| Check | What it prevents | Status here |
|---|---|---|
| Chronological train/test ordering | Training on future dates | Correct |
| Lagging historical target features | Using the current target in rolling inputs | Correct |
| Feature availability at forecast time | Using same-day information that spans the target window | Fails |

## What the metrics really say

For $N$ forecast observations, let $y_i$ be actual realized variance and $\widehat{y}_i$ be its forecast. Mean Absolute Error is

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}\left|y_i-\widehat{y}_i\right|.
$$

Root Mean Squared Error (RMSE) is derived by squaring each error, averaging the squares, and taking the square root:

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(y_i-\widehat{y}_i\right)^2}.
$$

The project also computes QLIKE, a loss commonly used for variance forecasts. For strictly positive forecast $\widehat{y}_i$, its implemented form is

$$
QLIKE=\frac{1}{N}\sum_{i=1}^{N}\left[\log(\widehat{y}_i)+\frac{y_i}{\widehat{y}_i}\right].
$$

Lower values are better for all three measures when comparing forecasts on the same target scale.

| Model | Mean MAE | Mean RMSE | Mean QLIKE |
|---|---:|---:|---:|
| Linear | $1.9865 \times 10^{-14}$ | $2.1812 \times 10^{-14}$ | -10.599912 |
| Persistence | $1.4148 \times 10^{-8}$ | $1.4148 \times 10^{-8}$ | -10.599911 |
| Rolling five-day mean | $4.2574 \times 10^{-8}$ | $4.2574 \times 10^{-8}$ | -10.599901 |

![Walk-forward model errors on a logarithmic scale](images/02_model_error_comparison.png)

The logarithmic axis makes the enormous gap visible. It should not be read as an enormous forecasting gain. The linear error is numerical residue from reconstructing the target. The persistence and rolling results are the only honest forecast baselines in this table, and the sample remains too narrow and smooth to support a broad performance claim.

There is a smaller reporting trap too. The command-line interface prints variance errors with six digits after the decimal point. Values near $10^{-8}$ appear as `0.000000`, which makes all three models look exact. Scientific notation is the safer display for quantities on this scale.

## The execution bridge and its units

The second half of the project turns a volatility input into an execution schedule. A parent order of 10,000 shares is divided into six slices. Time-Weighted Average Price (TWAP) allocates shares evenly. The volume-weighted schedule follows the observed bar-volume profile. The simplified Almgren-Chriss schedule uses a hyperbolic inventory curve.

Let $\lambda$ be the order's risk-aversion coefficient and $\sigma$ be daily volatility in decimal units. The code defines urgency as

$$
u=\lambda\sigma.
$$

For normalized time $t$ between zero and one, the remaining-inventory fraction is

$$
x(t)=\frac{\sinh(u(1-t))}{\sinh(u)}.
$$

At $t=0$, the numerator and denominator are both $\sinh(u)$, so $x(0)=1$. At $t=1$, the numerator is $\sinh(0)=0$, so $x(1)=0$. Larger $u$ bends the curve toward earlier execution.

```python
if override_daily_volatility is not None:
    daily_volatility = float(max(override_daily_volatility, 0.0))

urgency = max(order.risk_aversion * market_state.daily_volatility, MIN_URGENCY)
```

The interface names the override `daily_volatility`, but the research pipeline supplies $\widehat{RV}_d$, a variance forecast. Variance and volatility are not interchangeable. If returns are in decimal units, the required conversion is

$$
\widehat{\sigma}_d=\sqrt{\widehat{RV}_d}.
$$

The current bridge omits that square root. This is a unit mismatch at the module boundary. A production implementation would also calibrate the full Almgren-Chriss parameters in consistent time and cost units instead of treating $\lambda\sigma$ as a complete urgency parameter.

The simulator applies an explicit participation-rate impact rule. Let $q_k$ be shares executed in slice $k$, and let $V_k$ be market volume in the matching bar. The participation rate is $q_k/V_k$, and simulated impact in basis points is

$$
I_k=2+25\frac{q_k}{V_k}.
$$

One basis point is $0.01\%$, or $10^{-4}$ in decimal form. Because cost increases directly with participation, a schedule that concentrates shares in low-volume bars is penalized.

![Mean simulated execution cost by schedule](images/03_execution_cost_comparison.png)

Almgren-Chriss and TWAP are visually indistinguishable at this scale, while the VWAP-style schedule is much costlier under the simulator's participation-rate penalty. The exact values below show how small the first difference is.

| Schedule | Mean cost | Standard deviation | 90th percentile | Days |
|---|---:|---:|---:|---:|
| Almgren-Chriss | 32.794536 bps | 0.457265 bps | 33.417538 bps | 55 |
| TWAP | 32.794760 bps | 0.457282 bps | 33.417782 bps | 55 |
| VWAP-style | 48.038066 bps | 0.613094 bps | 48.871924 bps | 55 |

Almgren-Chriss beats TWAP by about $0.000224$ basis points on mean cost, an economically negligible difference. Its recorded 100% win rate against TWAP sounds stronger than the magnitude warrants. The VWAP-style schedule costs about $15.24$ basis points more than TWAP under this simulator, but that result describes the chosen volume profile and linear impact rule, not a general ranking of execution algorithms.

## A defensible next experiment

The pipeline can become a useful forecasting study without adding a complicated model. The data contract needs to change first.

1. Use full-session bars so the target spans a later period than the opening features.
2. Pick an opening cutoff that is strictly earlier than the target endpoint. For example, construct features through 10:30 and forecast variance from 10:30 to 16:00.
3. Record an availability timestamp for every feature. A feature is valid only if it exists when the forecast is issued.
4. Re-run persistence and rolling baselines before fitting the linear model. Complexity earns its place only after the simple baselines are credible.
5. Convert predicted variance to volatility with $\widehat{\sigma}_d=\sqrt{\widehat{RV}_d}$ before crossing the execution interface.
6. Test schedule sensitivity across risk aversion, order size, horizon, and impact coefficients. Report economic differences, not win rates alone.

The codebase already has a good offline boundary, small functions, and repeatable walk-forward machinery. The useful lesson from the current data is more specific: a clean split is necessary, but forecast-time availability is the real definition of a usable feature. When an error metric looks perfect, the first job is to trace the target, not celebrate the model.
