"""
Portfolio construction, execution simulation, and PnL evaluation components.
"""

from pumps_and_dumps.backtest.portfolio.base import (
    Portfolio,
    PortfolioStats,
    Transaction,
)
from pumps_and_dumps.backtest.portfolio.price_impact import (
    PriceImpactModel,
    fit_price_impact_model,
    fit_price_impact_model_from_klines,
)
from pumps_and_dumps.backtest.portfolio.topk import (
    TOPKPortfolio,
    evaluate_topk_pnl_for_quantities,
    portfolio_pnl_objective,
)
from pumps_and_dumps.backtest.portfolio.config import PortfolioExecutionConfig
from pumps_and_dumps.backtest.portfolio.impact_provider import (
    LookbackImpactModelProvider,
)
from pumps_and_dumps.backtest.portfolio.models import ExecutionResult, OrderIntent
from pumps_and_dumps.backtest.portfolio.manipulated_impact_provider import (
    ManipulatedImpactModelProvider,
)
from pumps_and_dumps.backtest.portfolio.pnl import PnLCalculator, USDTPnLCalculator
from pumps_and_dumps.backtest.portfolio.selector import TopKPortfolioSelector
from pumps_and_dumps.backtest.portfolio.sizing import NotionalSizer
from pumps_and_dumps.backtest.portfolio.vwap_estimator import VWAPEstimator

__all__ = [
    "Portfolio",
    "PortfolioExecutionConfig",
    "PortfolioStats",
    "TOPKPortfolio",
    "TopKPortfolioSelector",
    "Transaction",
    "OrderIntent",
    "ExecutionResult",
    "PriceImpactModel",
    "LookbackImpactModelProvider",
    "ManipulatedImpactModelProvider",
    "NotionalSizer",
    "PnLCalculator",
    "USDTPnLCalculator",
    "VWAPEstimator",
    "fit_price_impact_model",
    "fit_price_impact_model_from_klines",
    "evaluate_topk_pnl_for_quantities",
    "portfolio_pnl_objective",
]
