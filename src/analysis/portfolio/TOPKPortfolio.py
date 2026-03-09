import logging
from datetime import timedelta, datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from optuna import Trial

from analysis.pipelines.BaseModel import ImplementsRank
from analysis.portfolio.BasePortfolio import ImplementsPortfolio, Portfolio, Transaction
from analysis.portfolio.PriceImpact import PriceImpactModel, fit_price_impact_model
from analysis.utils.columns import COL_PROBAS_PRED, COL_CURRENCY_PAIR
from analysis.utils.sample import Dataset, Sample, DatasetType
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent
from core.time_utils import Bounds


class TOPKPortfolio(ImplementsPortfolio):

    def __init__(
            self,
            model: ImplementsRank,
            portfolio_size: int,
            buy_before: timedelta = timedelta(minutes=15),
            sell_after: timedelta = timedelta(minutes=1),
            use_price_impact: bool = False,
            order_notional_quote: float = 0.0,
            impact_lookback_days: int = 30,
            impact_liquidity_quantile: float = 0.9,
    ):
        super().__init__(model=model, buy_before=buy_before, sell_after=sell_after)
        self.portfolio_size: int = portfolio_size
        self.use_price_impact: bool = use_price_impact
        self.order_notional_quote: float = order_notional_quote
        self.impact_lookback_days: int = impact_lookback_days
        self.impact_liquidity_quantile: float = impact_liquidity_quantile
        self._impact_model_cache: Dict[Tuple[str, datetime], PriceImpactModel] = {}

    def create_portfolio(self, cross_section: Dataset) -> Portfolio:
        ords: np.ndarray = self.model.rank(dataset=cross_section)

        _data = cross_section.all_data()
        _data[COL_PROBAS_PRED] = ords

        df_portfolio: pd.DataFrame = _data.sort_values(
            by=[COL_PROBAS_PRED], ascending=False
        ).iloc[
            : self.portfolio_size
        ]  # type: ignore

        weights: np.ndarray = np.full(
            shape=(self.portfolio_size,), fill_value=1 / self.portfolio_size
        )
        currency_pairs: List[CurrencyPair] = [
            CurrencyPair.from_string(symbol=symbol)
            for symbol in df_portfolio[COL_CURRENCY_PAIR]
        ]
        return Portfolio(currency_pairs=currency_pairs, weights=weights)  # type: ignore

    def _get_impact_model(self, pump: PumpEvent, cp: CurrencyPair) -> PriceImpactModel:
        cache_key: Tuple[str, datetime] = (cp.name, pump.time)
        if cache_key in self._impact_model_cache:
            return self._impact_model_cache[cache_key]

        lookback_bounds = Bounds(
            start_inclusive=pump.time - timedelta(days=self.impact_lookback_days),
            end_exclusive=pump.time,
        )
        trades_lookback: pd.DataFrame = self.load_trades(
            bounds=lookback_bounds, currency_pair=cp
        )
        model = fit_price_impact_model(
            trades=trades_lookback,
            liquidity_quantile=self.impact_liquidity_quantile,
        )
        self._impact_model_cache[cache_key] = model
        return model

    def _estimate_executed_notional_quote(
            self, impact_model: PriceImpactModel
    ) -> float:
        intended_notional_quote: float = max(self.order_notional_quote, 0.0)
        if intended_notional_quote <= 0:
            return 0.0

        entry_fillable_quote: float = impact_model.estimate_fill_notional(
            side=1, intended_notional_quote=intended_notional_quote
        )
        exit_fillable_quote: float = impact_model.estimate_fill_notional(
            side=-1, intended_notional_quote=intended_notional_quote
        )
        return max(
            0.0, min(intended_notional_quote, entry_fillable_quote, exit_fillable_quote)
        )

    def _estimate_execution_vwap(
            self,
            base_price: float,
            side: int,
            executed_notional_quote: float,
            impact_model: PriceImpactModel,
    ) -> tuple[float, float]:
        return impact_model.estimate_vwap_price(
            base_price=base_price, side=side, notional_quote=executed_notional_quote
        )

    def _create_transaction_with_impact(
            self,
            cp: CurrencyPair,
            pump: PumpEvent,
            entry_price: float,
            exit_price: float,
            entry_ts: datetime,
            exit_ts: datetime,
    ) -> Transaction:
        if not self.use_price_impact or self.order_notional_quote <= 0:
            return Transaction(
                entry_price=entry_price,
                exit_price=exit_price,
                currency_pair=cp,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
            )

        impact_model: PriceImpactModel = self._get_impact_model(pump=pump, cp=cp)
        executed_notional_quote: float = self._estimate_executed_notional_quote(
            impact_model=impact_model
        )
        impacted_entry_price, entry_impact_bps = self._estimate_execution_vwap(
            base_price=entry_price,
            side=1,
            executed_notional_quote=executed_notional_quote,
            impact_model=impact_model,
        )
        impacted_exit_price, exit_impact_bps = self._estimate_execution_vwap(
            base_price=exit_price,
            side=-1,
            executed_notional_quote=executed_notional_quote,
            impact_model=impact_model,
        )
        fill_ratio: float = 0.0
        if self.order_notional_quote > 0:
            fill_ratio = max(0.0, executed_notional_quote / self.order_notional_quote)

        return Transaction(
            entry_price=impacted_entry_price,
            exit_price=impacted_exit_price,
            currency_pair=cp,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            intended_notional_quote=self.order_notional_quote,
            entry_filled_notional_quote=executed_notional_quote,
            exit_filled_notional_quote=executed_notional_quote,
            entry_impact_bps=entry_impact_bps,
            exit_impact_bps=exit_impact_bps,
            fill_ratio=fill_ratio,
        )

    def regular_transaction(
            self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair
    ) -> Transaction:
        assert ts_price.index.is_monotonic_increasing
        entry_series: pd.Series = ts_price[
            ts_price.index <= pump.time - self._buy_before
            ]
        exit_series: pd.Series = ts_price[ts_price.index >= pump.time]

        if entry_series.empty or exit_series.empty:
            logging.info("No data to get prices for %s", cp.name)
            return Transaction.empty(currency_pair=cp)

        entry_price, entry_ts = entry_series.iloc[-1], entry_series.index[-1]
        exit_price, exit_ts = exit_series.iloc[0], exit_series.index[0]
        return self._create_transaction_with_impact(
            cp=cp,
            pump=pump,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
        )

    def pumped_transaction(
            self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair
    ) -> Transaction:
        assert ts_price.index.is_monotonic_increasing
        entry: pd.Series = ts_price[ts_price.index <= pump.time - self._buy_before]
        exit: pd.Series = ts_price[ts_price.index >= pump.time + self._sell_after]

        if entry.empty or exit.empty:
            logging.info("No data to get prices for %s", cp.name)
            return Transaction.empty(currency_pair=cp)

        entry_price, entry_ts = entry.iloc[-1], entry.index[-1]
        exit_price, exit_ts = exit.iloc[0], exit.index[0]
        return self._create_transaction_with_impact(
            cp=cp,
            pump=pump,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
        )

    def evaluate_pnl_for_quantities(
            self, dataset: Dataset, quantities_quote: List[float]
    ) -> pd.DataFrame:
        """
        Re-run backtest for multiple order nationals (quote currency units) with price impact enabled.
        """
        rows: List[Dict[str, float]] = []
        prev_order_notional: float = self.order_notional_quote
        prev_use_price_impact: bool = self.use_price_impact

        try:
            self.use_price_impact = True
            for quantity_quote in quantities_quote:
                self.order_notional_quote = quantity_quote
                pnls: List[float] = []
                fill_ratios: List[float] = []

                for pump in dataset.get_pumps():
                    stats = self.evaluate_for_pump(dataset=dataset, pump=pump)
                    pnls.append(stats.pnl)
                    fill_ratios.extend(
                        [tx.fill_ratio for tx in stats.txs if not tx.is_empty()]
                    )

                rows.append(
                    {
                        "quantity_quote": float(quantity_quote),
                        "overall_pnl": float(np.sum(pnls)),
                        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
                        "mean_fill_ratio": (
                            float(np.mean(fill_ratios)) if fill_ratios else 0.0
                        ),
                    }
                )
        finally:
            self.order_notional_quote = prev_order_notional
            self.use_price_impact = prev_use_price_impact

        return pd.DataFrame(rows).sort_values("quantity_quote").reset_index(drop=True)


def evaluate_topk_pnl_for_quantities(
        model: ImplementsRank,
        dataset: Dataset,
        portfolio_size: int,
        quantities_quote: List[float],
        buy_before: timedelta = timedelta(minutes=15),
        sell_after: timedelta = timedelta(minutes=1),
        impact_lookback_days: int = 30,
        impact_liquidity_quantile: float = 0.9,
) -> pd.DataFrame:
    manager = TOPKPortfolio(
        model=model,
        portfolio_size=portfolio_size,
        buy_before=buy_before,
        sell_after=sell_after,
        use_price_impact=True,
        order_notional_quote=quantities_quote[0] if quantities_quote else 0.0,
        impact_lookback_days=impact_lookback_days,
        impact_liquidity_quantile=impact_liquidity_quantile,
    )
    return manager.evaluate_pnl_for_quantities(
        dataset=dataset, quantities_quote=quantities_quote
    )


def portfolio_pnl_objective(
        trial: Trial, model: ImplementsRank, sample: Sample
) -> float:
    buy_before_minutes: int = trial.suggest_categorical(
        "buy_before", choices=[1, 2, 3, 5, 10, 15, 30, 60]
    )
    sell_after_minutes: int = trial.suggest_categorical(
        "sell_after", choices=[1, 2, 3, 5, 10, 15]
    )
    portfolio_size: int = trial.suggest_categorical(
        "portfolio_size", choices=[1, 2, 3, 5, 10, 15, 20, 30, 50]
    )
    manager = TOPKPortfolio(
        model=model,
        portfolio_size=portfolio_size,
        buy_before=timedelta(minutes=buy_before_minutes),
        sell_after=timedelta(minutes=sell_after_minutes),
    )
    return manager.compute_overall_pnl(
        dataset=sample.get_dataset(ds_type=DatasetType.TEST)
    )
