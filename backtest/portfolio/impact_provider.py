from datetime import datetime, timedelta
from typing import Callable, Dict, Tuple

import pandas as pd

from backtest.portfolio.PriceImpact import PriceImpactModel, fit_price_impact_model
from backtest.portfolio.interfaces import ImpactModelProvider
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent
from core.time_utils import Bounds

LoadTradesFn = Callable[[Bounds, CurrencyPair], pd.DataFrame]


class LookbackImpactModelProvider(ImpactModelProvider):
    """
    Caches and provides impact models fit on a fixed lookback window per (asset, pump time).
    """

    def __init__(self, load_trades: LoadTradesFn, lookback_days: int, liquidity_quantile: float):
        """
        Build provider that fits models from trailing trade history.

        Args:
            load_trades: Function used to fetch historical trades for bounds/pair.
            lookback_days: Number of trailing days before pump time to fit on.
            liquidity_quantile: Quantile used to estimate executable notional cap.
        """
        self._load_trades: LoadTradesFn = load_trades
        self.lookback_days: int = lookback_days
        self.liquidity_quantile: float = liquidity_quantile
        self._cache: Dict[Tuple[str, datetime], PriceImpactModel] = {}

    def get_impact_model(self, pump: PumpEvent, currency_pair: CurrencyPair) -> PriceImpactModel:
        """
        Return cached or newly-fitted impact model for `(currency_pair, pump.time)`.
        """
        cache_key: Tuple[str, datetime] = (currency_pair.name, pump.time)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bounds = Bounds(
            start_inclusive=pump.time - timedelta(days=self.lookback_days),
            end_exclusive=pump.time,
        )
        trades_lookback = self._load_trades(bounds=bounds, currency_pair=currency_pair)
        model = fit_price_impact_model(trades=trades_lookback, liquidity_quantile=self.liquidity_quantile)
        self._cache[cache_key] = model
        return model
