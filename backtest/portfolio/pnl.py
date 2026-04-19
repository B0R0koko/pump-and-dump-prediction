from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from backtest.portfolio.BasePortfolio import Portfolio, Transaction


class PnLCalculator(ABC):
    @abstractmethod
    def calculate_transaction_pnl(self, tx: "Transaction") -> float: ...

    def calculate_portfolio_pnl(self, portfolio: "Portfolio", txs: List["Transaction"]) -> float:
        """Aggregate weighted transaction PnL for non-empty legs in a portfolio."""
        pnl: float = 0.0
        for tx in txs:
            if tx.is_empty():
                continue
            pnl += self.calculate_transaction_pnl(tx) * portfolio.get_weight(tx.currency_pair)
        return pnl


class USDTPnLCalculator(PnLCalculator):
    """
    Computes transaction PnL in USDT when conversion metadata is available.
    """

    def calculate_transaction_pnl(self, tx: "Transaction") -> float:
        """
        Compute transaction PnL in USDT.

        If quote/USDT metadata is present we convert quote notional using exit-time
        conversion; otherwise fall back to raw fractional return.
        """
        if (
            tx.exit_filled_notional_quote is not None
            and tx.exit_filled_notional_usdt is not None
            and tx.exit_filled_notional_quote > 0
        ):
            quote_to_usdt_exit: float = tx.exit_filled_notional_usdt / tx.exit_filled_notional_quote
            return tx.transaction_return * tx.exit_filled_notional_quote * quote_to_usdt_exit
        if (
            tx.intended_notional_quote is not None
            and tx.exit_filled_notional_usdt is not None
            and tx.intended_notional_quote > 0
        ):
            quote_to_usdt_exit = tx.exit_filled_notional_usdt / tx.intended_notional_quote
            return tx.transaction_return * tx.intended_notional_quote * quote_to_usdt_exit
        return tx.transaction_return
