from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from backtest.portfolio.PriceImpact import (
    PriceImpactModel,
    fit_price_impact_model,
)
from backtest.portfolio.interfaces import QuoteToUSDTProvider
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent
from core.time_utils import Bounds

LoadTradesFn = Callable[[Bounds, CurrencyPair], pd.DataFrame]


class ManipulatedImpactModelProvider:
    """
    Fit impact models on the realized manipulation window using trade-level data,
    with notionals normalised to USDT.
    """

    def __init__(
        self,
        load_trades: LoadTradesFn,
        liquidity_quantile: float,
        indicative_price_provider: Optional[QuoteToUSDTProvider] = None,
    ):
        self._load_trades: LoadTradesFn = load_trades
        self.liquidity_quantile: float = liquidity_quantile
        self._indicative_price_provider: Optional[QuoteToUSDTProvider] = indicative_price_provider
        self._cache: Dict[Tuple[str, datetime, datetime], PriceImpactModel] = {}

    def _get_quote_to_usdt(self, currency_pair: CurrencyPair, ts: datetime) -> float:
        if self._indicative_price_provider is None:
            return 1.0
        try:
            return self._indicative_price_provider.get_quote_to_usdt_indicative_price(
                quote_asset=currency_pair.term, ts=ts,
            )
        except Exception:
            return 1.0

    def get_impact_model(
        self,
        pump: PumpEvent,
        currency_pair: CurrencyPair,
        end_exclusive: datetime,
    ) -> PriceImpactModel:
        """
        Fit an impact model on data observed during the manipulation window.
        """
        cache_key: Tuple[str, datetime, datetime] = (currency_pair.name, pump.time, end_exclusive)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bounds = Bounds(
            start_inclusive=pump.time,
            end_exclusive=end_exclusive,
        )

        manipulated_trades = self._load_trades(bounds, currency_pair)
        quote_to_usdt = self._get_quote_to_usdt(currency_pair=currency_pair, ts=pump.time)
        model = fit_price_impact_model(
            trades=manipulated_trades,
            liquidity_quantile=self.liquidity_quantile,
            quote_to_usdt=quote_to_usdt,
        )

        self._cache[cache_key] = model
        return model
