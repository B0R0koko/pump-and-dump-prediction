import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from optuna import Trial

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.BasePortfolio import ImplementsPortfolio, Portfolio, Transaction
from backtest.portfolio.PriceImpact import PriceImpactModel
from backtest.portfolio.config import PortfolioExecutionConfig
from backtest.portfolio.execution_engine import ExecutionEngine
from backtest.portfolio.impact_provider import LookbackImpactModelProvider
from backtest.portfolio.interfaces import QuoteToUSDTProvider
from backtest.portfolio.models import OrderIntent
from backtest.portfolio.selector import TopKPortfolioSelector
from backtest.portfolio.sizing import NotionalSizer
from backtest.utils.IndicativePriceProvider import IndicativePriceProvider
from backtest.utils.sample import Dataset, DatasetType, Sample
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent


class TOPKPortfolio(ImplementsPortfolio):
    """
    Orchestrates top-k portfolio construction, execution simulation, and PnL evaluation.
    """

    def __init__(
        self,
        model: ImplementsRank,
        config: Optional[PortfolioExecutionConfig] = None,
        *,
        portfolio_size: int = 1,
        buy_before: timedelta = timedelta(minutes=15),
        sell_after: timedelta = timedelta(minutes=1),
        use_price_impact: bool = False,
        order_notional_quote: float = 0.0,
        order_notional_usdt: float = 1.0,
        impact_lookback_days: int = 30,
        impact_liquidity_quantile: float = 0.9,
        indicative_price_provider: Optional[QuoteToUSDTProvider] = None,
    ):
        effective_config = config or PortfolioExecutionConfig(
            portfolio_size=portfolio_size,
            buy_before=buy_before,
            sell_after=sell_after,
            use_price_impact=use_price_impact,
            order_notional_quote=order_notional_quote,
            order_notional_usdt=order_notional_usdt,
            impact_lookback_days=impact_lookback_days,
            impact_liquidity_quantile=impact_liquidity_quantile,
        )
        super().__init__(
            model=model,
            buy_before=effective_config.buy_before,
            sell_after=effective_config.sell_after,
        )
        self.config: PortfolioExecutionConfig = effective_config

        self._selector = TopKPortfolioSelector(portfolio_size=self.config.portfolio_size)
        self._indicative_price_provider: QuoteToUSDTProvider = indicative_price_provider or IndicativePriceProvider()
        self._notional_sizer = NotionalSizer(indicative_price_provider=self._indicative_price_provider)
        self._impact_model_provider = LookbackImpactModelProvider(
            load_trades=self.load_trades,
            lookback_days=self.config.impact_lookback_days,
            liquidity_quantile=self.config.impact_liquidity_quantile,
        )
        self._execution_engine = ExecutionEngine(indicative_price_provider=self._indicative_price_provider)

    @property
    def portfolio_size(self) -> int:
        return self.config.portfolio_size

    @property
    def use_price_impact(self) -> bool:
        return self.config.use_price_impact

    @use_price_impact.setter
    def use_price_impact(self, value: bool) -> None:
        self.config.use_price_impact = bool(value)

    @property
    def order_notional_quote(self) -> float:
        return self.config.order_notional_quote

    @order_notional_quote.setter
    def order_notional_quote(self, value: float) -> None:
        self.config.order_notional_quote = float(value)

    @property
    def order_notional_usdt(self) -> float:
        return self.config.order_notional_usdt

    @order_notional_usdt.setter
    def order_notional_usdt(self, value: float) -> None:
        self.config.order_notional_usdt = float(value)

    @property
    def impact_lookback_days(self) -> int:
        return self.config.impact_lookback_days

    def create_portfolio(self, cross_section: Dataset) -> Portfolio:
        return self._selector.select_portfolio(model=self.model, cross_section=cross_section)

    def _get_impact_model(self, pump: PumpEvent, cp: CurrencyPair) -> PriceImpactModel:
        return self._impact_model_provider.get_impact_model(pump=pump, currency_pair=cp)

    def _build_order_intent(
        self,
        cp: CurrencyPair,
        pump: PumpEvent,
        entry_price: float,
        exit_price: float,
        entry_ts: datetime,
        exit_ts: datetime,
    ) -> OrderIntent:
        """
        Build execution intent for one asset leg.

        This resolves intended notional in quote currency using either a direct quote
        notional target or USDT target converted at entry-time indicative price.
        """
        intended_notional_quote: float = self._notional_sizer.resolve_intended_notional_quote(
            currency_pair=cp,
            entry_ts=entry_ts,
            order_notional_quote=self.config.order_notional_quote,
            order_notional_usdt=self.config.order_notional_usdt,
        )
        return OrderIntent(
            currency_pair=cp,
            pump=pump,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            intended_notional_quote=intended_notional_quote,
        )

    def _create_transaction_from_intent(self, intent: OrderIntent) -> Transaction:
        """
        Convert order intent into a transaction with simulated fills and execution prices.

        If the intended notional is zero, the transaction is returned with only raw prices.
        Otherwise, execution is simulated with or without impact according to config.
        """
        if intent.intended_notional_quote <= 0:
            return Transaction(
                entry_price=intent.entry_price,
                exit_price=intent.exit_price,
                currency_pair=intent.currency_pair,
                entry_ts=intent.entry_ts,
                exit_ts=intent.exit_ts,
            )

        impact_model: Optional[PriceImpactModel] = None
        if self.config.use_price_impact:
            impact_model = self._get_impact_model(pump=intent.pump, cp=intent.currency_pair)

        execution = self._execution_engine.execute(
            intent=intent,
            use_price_impact=self.config.use_price_impact,
            impact_model=impact_model,
        )
        return Transaction(
            entry_price=execution.entry_price,
            exit_price=execution.exit_price,
            currency_pair=intent.currency_pair,
            entry_ts=intent.entry_ts,
            exit_ts=intent.exit_ts,
            intended_notional_quote=intent.intended_notional_quote,
            entry_filled_notional_quote=execution.filled_notional_quote,
            exit_filled_notional_quote=execution.filled_notional_quote,
            entry_filled_notional_usdt=execution.filled_notional_usdt_entry,
            exit_filled_notional_usdt=execution.filled_notional_usdt_exit,
            entry_impact_bps=execution.entry_impact_bps,
            exit_impact_bps=execution.exit_impact_bps,
            fill_ratio=execution.fill_ratio,
        )

    def regular_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        """
        Simulate transaction for a non-manipulated asset around the pump window.

        Entry is the latest price at or before `pump.time - buy_before`.
        Exit is the earliest price at or after `pump.time`.
        """
        assert ts_price.index.is_monotonic_increasing
        entry_series: pd.Series = ts_price[ts_price.index <= pump.time - self._buy_before]
        exit_series: pd.Series = ts_price[ts_price.index >= pump.time]

        if entry_series.empty or exit_series.empty:
            logging.info("No data to get prices for %s", cp.name)
            return Transaction.empty(currency_pair=cp)

        entry_price, entry_ts = entry_series.iloc[-1], entry_series.index[-1]
        exit_price, exit_ts = exit_series.iloc[0], exit_series.index[0]
        intent = self._build_order_intent(
            cp=cp,
            pump=pump,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
        )
        return self._create_transaction_from_intent(intent=intent)

    def pumped_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        """
        Simulate transaction for the manipulated asset itself.

        Entry still occurs before the pump, while exit is delayed by `sell_after`
        to avoid unrealistically selling inside the pump candle.
        """
        assert ts_price.index.is_monotonic_increasing
        entry: pd.Series = ts_price[ts_price.index <= pump.time - self._buy_before]
        exit: pd.Series = ts_price[ts_price.index >= pump.time + self._sell_after]

        if entry.empty or exit.empty:
            logging.info("No data to get prices for %s", cp.name)
            return Transaction.empty(currency_pair=cp)

        entry_price, entry_ts = entry.iloc[-1], entry.index[-1]
        exit_price, exit_ts = exit.iloc[0], exit.index[0]
        intent = self._build_order_intent(
            cp=cp,
            pump=pump,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
        )
        return self._create_transaction_from_intent(intent=intent)

    def evaluate_pnl_for_quantities(self, dataset: Dataset, quantities_usdt: List[float]) -> pd.DataFrame:
        """
        Sweep order notionals in USDT and re-run full backtest with impact enabled.

        Returns aggregate portfolio PnL metrics for each tested investment size.
        """
        rows: List[Dict[str, float]] = []
        prev_order_notional_quote: float = self.config.order_notional_quote
        prev_order_notional_usdt: float = self.config.order_notional_usdt
        prev_use_price_impact: bool = self.config.use_price_impact

        try:
            self.config.use_price_impact = True
            self.config.order_notional_quote = 0.0
            for quantity_usdt in quantities_usdt:
                self.config.order_notional_usdt = float(quantity_usdt)
                pnls: List[float] = []

                for pump in dataset.get_pumps():
                    stats = self.evaluate_for_pump(dataset=dataset, pump=pump)
                    pnls.append(stats.pnl)
                rows.append(
                    {
                        "quantity_usdt": float(quantity_usdt),
                        "overall_pnl": float(np.sum(pnls)),
                        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
                    }
                )
        finally:
            self.config.order_notional_quote = prev_order_notional_quote
            self.config.order_notional_usdt = prev_order_notional_usdt
            self.config.use_price_impact = prev_use_price_impact

        return pd.DataFrame(rows).sort_values("quantity_usdt").reset_index(drop=True)


def evaluate_topk_pnl_for_quantities(
    model: ImplementsRank,
    dataset: Dataset,
    portfolio_size: int,
    quantities_usdt: Optional[List[float]] = None,
    quantities_quote: Optional[List[float]] = None,
    buy_before: timedelta = timedelta(minutes=15),
    sell_after: timedelta = timedelta(minutes=1),
    impact_lookback_days: int = 30,
    impact_liquidity_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    Backtest top-k strategy over a sweep of order sizes expressed in USDT.

    `quantities_quote` is kept only for backward compatibility and is interpreted
    as USDT quantities when `quantities_usdt` is not provided.
    """
    resolved_quantities_usdt: List[float] = quantities_usdt if quantities_usdt is not None else (quantities_quote or [])
    config = PortfolioExecutionConfig(
        portfolio_size=portfolio_size,
        buy_before=buy_before,
        sell_after=sell_after,
        use_price_impact=True,
        order_notional_quote=0.0,
        order_notional_usdt=resolved_quantities_usdt[0] if resolved_quantities_usdt else 0.0,
        impact_lookback_days=impact_lookback_days,
        impact_liquidity_quantile=impact_liquidity_quantile,
    )
    manager = TOPKPortfolio(model=model, config=config)
    return manager.evaluate_pnl_for_quantities(dataset=dataset, quantities_usdt=resolved_quantities_usdt)


def portfolio_pnl_objective(trial: Trial, model: ImplementsRank, sample: Sample) -> float:
    """
    Optuna objective for tuning portfolio timing and top-k size parameters.
    """
    buy_before_minutes: int = trial.suggest_categorical("buy_before", choices=[1, 2, 3, 5, 10, 15, 30, 60])
    sell_after_minutes: int = trial.suggest_categorical("sell_after", choices=[1, 2, 3, 5, 10, 15])
    portfolio_size: int = trial.suggest_categorical("portfolio_size", choices=[1, 2, 3, 5, 10, 15, 20, 30, 50])
    config = PortfolioExecutionConfig(
        portfolio_size=portfolio_size,
        buy_before=timedelta(minutes=buy_before_minutes),
        sell_after=timedelta(minutes=sell_after_minutes),
    )
    manager = TOPKPortfolio(model=model, config=config)
    return manager.compute_overall_pnl(dataset=sample.get_dataset(ds_type=DatasetType.TEST))
