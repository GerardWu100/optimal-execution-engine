"""Core typed objects for orders, market state, and schedules."""

from dataclasses import dataclass


@dataclass(slots=True)
class ParentOrder:
    """Inputs for a parent execution problem.

    Parameters
    ----------
    symbol
        Ticker symbol for the instrument to execute.
    side
        Trade direction, expected values are "BUY" or "SELL".
    shares
        Total shares in the parent order.
    arrival_price
        Benchmark price at the decision time.
    horizon_minutes
        Execution horizon measured in minutes.
    risk_aversion
        Non-negative urgency coefficient for inventory risk.
    """

    symbol: str
    side: str
    shares: int
    arrival_price: float
    horizon_minutes: int
    risk_aversion: float


@dataclass(slots=True)
class MarketState:
    """Calibrated market statistics used by scheduling models.

    Parameters
    ----------
    average_daily_volume
        Average daily trading volume in shares.
    daily_volatility
        Daily return volatility in decimal units.
    spread_bps
        Effective spread proxy in basis points.
    """

    average_daily_volume: float
    daily_volatility: float
    spread_bps: float
