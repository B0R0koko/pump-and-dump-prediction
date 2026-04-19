from backtest.portfolio.interfaces import ExecutionImpactModel, QuoteToUSDTProvider
from backtest.portfolio.models import ExecutionResult, OrderIntent


class ExecutionEngine:
    """
    Simulates execution for an order intent, optionally with market impact.
    """

    def __init__(self, indicative_price_provider: QuoteToUSDTProvider):
        self.indicative_price_provider: QuoteToUSDTProvider = indicative_price_provider

    def execute(
        self,
        intent: OrderIntent,
        use_price_impact: bool,
        impact_model: ExecutionImpactModel | None,
    ) -> ExecutionResult:
        """
        Execute order intent and return impacted prices in quote and USDT units.

        Orders are assumed to be fully executable at any size; the price-impact
        model captures the resulting slippage through the VWAP price adjustment.
        """
        quote_to_usdt_entry: float = self.indicative_price_provider.get_quote_to_usdt_indicative_price(
            quote_asset=intent.currency_pair.term,
            ts=intent.entry_ts,
        )
        quote_to_usdt_exit: float = self.indicative_price_provider.get_quote_to_usdt_indicative_price(
            quote_asset=intent.currency_pair.term,
            ts=intent.exit_ts,
        )

        if not use_price_impact:
            filled_notional_quote: float = max(intent.intended_notional_quote, 0.0)
            return ExecutionResult(
                entry_price=intent.entry_price,
                exit_price=intent.exit_price,
                filled_notional_quote=filled_notional_quote,
                filled_notional_usdt_entry=filled_notional_quote * quote_to_usdt_entry,
                filled_notional_usdt_exit=filled_notional_quote * quote_to_usdt_exit,
                entry_impact_bps=0.0,
                exit_impact_bps=0.0,
            )

        if impact_model is None:
            raise ValueError("impact_model is required when use_price_impact=True")

        filled_notional_quote = max(intent.intended_notional_quote, 0.0)
        impacted_entry_price, entry_impact_bps = impact_model.estimate_vwap_price(
            base_price=intent.entry_price,
            side=1,
            notional_quote=filled_notional_quote,
        )
        impacted_exit_price, exit_impact_bps = impact_model.estimate_vwap_price(
            base_price=intent.exit_price,
            side=-1,
            notional_quote=filled_notional_quote,
        )

        return ExecutionResult(
            entry_price=impacted_entry_price,
            exit_price=impacted_exit_price,
            filled_notional_quote=filled_notional_quote,
            filled_notional_usdt_entry=filled_notional_quote * quote_to_usdt_entry,
            filled_notional_usdt_exit=filled_notional_quote * quote_to_usdt_exit,
            entry_impact_bps=entry_impact_bps,
            exit_impact_bps=exit_impact_bps,
        )
