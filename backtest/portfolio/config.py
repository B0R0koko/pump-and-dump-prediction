from dataclasses import dataclass
from datetime import timedelta


@dataclass
class PortfolioExecutionConfig:
    """
    Holds portfolio construction and execution settings.
    """

    portfolio_size: int
    buy_before: timedelta = timedelta(minutes=15)
    sell_after: timedelta = timedelta(minutes=1)
    use_price_impact: bool = False
    order_notional_quote: float = 0.0
    order_notional_usdt: float = 1.0
    impact_lookback_days: int = 30
    impact_liquidity_quantile: float = 0.9

    def __post_init__(self) -> None:
        if self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        if self.impact_lookback_days <= 0:
            raise ValueError("impact_lookback_days must be positive")
        if not (0 < self.impact_liquidity_quantile <= 1):
            raise ValueError("impact_liquidity_quantile must be in (0, 1]")
