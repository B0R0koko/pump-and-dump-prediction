from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.core.time_utils import Bounds

if TYPE_CHECKING:
    from pumps_and_dumps.backtest.datasets.sample import Dataset
    from pumps_and_dumps.backtest.pipelines.base_model import ImplementsRank
    from pumps_and_dumps.backtest.portfolio.base import Portfolio, PortfolioStats
    from pumps_and_dumps.backtest.portfolio.models import ExecutionResult, OrderIntent


class TradeSource(Protocol):
    """Read the trades needed by a simulation, independently of their storage."""

    def load_trades(
        self, bounds: Bounds, currency_pair: CurrencyPair
    ) -> pd.DataFrame: ...


class PriceProvider(Protocol):
    """Look up a past-only price without requiring currency-conversion methods."""

    def get_indicative_price(self, symbol: str, ts: datetime) -> float | None: ...


class PortfolioSelector(Protocol):
    """Choose assets and weights from a ranked cross-section."""

    def select_portfolio(
        self, model: ImplementsRank, cross_section: Dataset
    ) -> Portfolio: ...


class PortfolioEvaluator(Protocol):
    """Expose event evaluation without requiring a particular portfolio strategy."""

    def evaluate_for_pump(
        self, dataset: Dataset, pump: PumpEvent
    ) -> PortfolioStats: ...


class QuoteToUSDTProvider(Protocol):
    """Maps a quote asset and timestamp to an indicative USDT conversion rate."""

    def get_quote_to_usdt_indicative_price(
        self, quote_asset: str, ts: datetime
    ) -> float: ...


class ExecutionImpactModel(Protocol):
    """Execution-impact abstraction used by the simulation engine."""

    def estimate_vwap_price(
        self, base_price: float, side: int, notional_quote: float
    ) -> tuple[float, float]: ...


class FittedImpactModel(ExecutionImpactModel, Protocol):
    """Add the fit diagnostics needed to select and report execution regimes."""

    @property
    def num_samples(self) -> int: ...


class ImpactModelProvider(Protocol):
    """Loads or constructs execution-impact models for a pump/currency context."""

    def get_impact_model(
        self,
        pump: PumpEvent,
        currency_pair: CurrencyPair,
        end_exclusive: datetime | None = None,
    ) -> FittedImpactModel: ...


class ExecutionEstimator(Protocol):
    """Simulate an order without owning selection or market-data loading."""

    def estimate(
        self,
        intent: OrderIntent,
        use_price_impact: bool,
        entry_impact_model: ExecutionImpactModel | None,
        exit_impact_model: ExecutionImpactModel | None,
    ) -> ExecutionResult: ...
