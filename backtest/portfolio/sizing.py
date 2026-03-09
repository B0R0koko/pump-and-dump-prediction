from datetime import datetime

from backtest.portfolio.interfaces import QuoteToUSDTProvider
from core.currency_pair import CurrencyPair


class NotionalSizer:
    """
    Resolves target notional in quote units for an asset transaction.
    """

    def __init__(self, indicative_price_provider: QuoteToUSDTProvider):
        self.indicative_price_provider: QuoteToUSDTProvider = indicative_price_provider

    def resolve_intended_notional_quote(
        self,
        currency_pair: CurrencyPair,
        entry_ts: datetime,
        order_notional_quote: float,
        order_notional_usdt: float,
    ) -> float:
        """
        Resolve intended notional in quote currency.

        Priority:
        1. Explicit quote notional if positive.
        2. Otherwise convert USDT notional into quote units using entry-time indicative price.
        """
        if order_notional_quote > 0:
            return float(order_notional_quote)
        if order_notional_usdt > 0:
            quote_to_usdt: float = self.indicative_price_provider.get_quote_to_usdt_indicative_price(
                quote_asset=currency_pair.term,
                ts=entry_ts,
            )
            return float(order_notional_usdt) / quote_to_usdt
        return 0.0
