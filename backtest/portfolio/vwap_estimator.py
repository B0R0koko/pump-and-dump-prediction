from backtest.portfolio.interfaces import ExecutionImpactModel, QuoteToUSDTProvider
from backtest.portfolio.models import ExecutionResult, OrderIntent


class VWAPEstimator:
    """
    Estimate round-trip execution prices from side-aware market-impact models.

    Entry and exit can use different impact regimes, although the default
    backtest path applies one pre-pump model to both legs.
    """

    def __init__(self, indicative_price_provider: QuoteToUSDTProvider):
        self.indicative_price_provider: QuoteToUSDTProvider = indicative_price_provider

    def estimate(
        self,
        intent: OrderIntent,
        use_price_impact: bool,
        entry_impact_model: ExecutionImpactModel | None,
        exit_impact_model: ExecutionImpactModel | None,
    ) -> ExecutionResult:
        """
        Estimate executed notionals and impacted prices in quote and USDT units.

        Orders are assumed to be fully executable at any size; the impact models
        only adjust the entry and exit VWAPs to reflect slippage.
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

        if entry_impact_model is None:
            raise ValueError("entry_impact_model is required when use_price_impact=True")
        if exit_impact_model is None:
            raise ValueError("exit_impact_model is required when use_price_impact=True")

        filled_notional_quote = max(intent.intended_notional_quote, 0.0)
        impacted_entry_price, entry_impact_bps = entry_impact_model.estimate_vwap_price(
            base_price=intent.entry_price,
            side=1,
            notional_quote=filled_notional_quote,
        )
        impacted_exit_price, exit_impact_bps = exit_impact_model.estimate_vwap_price(
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
