import logging
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from analysis.models.BaseModel import ImplementsRank
from analysis.portfolio.BasePortfolio import ImplementsPortfolio, Portfolio, Transaction
from analysis.utils.columns import COL_PROBAS_PRED, COL_CURRENCY_PAIR
from analysis.utils.sample import Dataset
from core.currency_pair import CurrencyPair
from core.pump_event import PumpEvent


class TOPKPortfolio(ImplementsPortfolio):

    def __init__(self, model: ImplementsRank, portfolio_size: int):
        super().__init__(model=model)
        self.portfolio_size: int = portfolio_size

    def create_portfolio(self, cross_section: Dataset) -> Portfolio:
        ords: np.ndarray = self.model.rank(dataset=cross_section)

        _data = cross_section.all_data()
        _data[COL_PROBAS_PRED] = ords

        df_portfolio: pd.DataFrame = (
            _data.sort_values(by=[COL_PROBAS_PRED], ascending=False)
            .iloc[:self.portfolio_size]  # type:ignore
        )

        weights: np.ndarray = np.full(shape=(self.portfolio_size,), fill_value=1 / self.portfolio_size)
        currency_pairs: List[CurrencyPair] = [
            CurrencyPair.from_string(symbol=symbol) for symbol in df_portfolio[COL_CURRENCY_PAIR]
        ]
        return Portfolio(currency_pairs=currency_pairs, weights=weights)  # type:ignore

    def regular_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        assert ts_price.index.is_monotonic_increasing
        entry: pd.Series = ts_price[ts_price.index <= pump.time - timedelta(minutes=15)]
        exit: pd.Series = ts_price[ts_price.index >= pump.time]

        if entry.empty or exit.empty:
            logging.info("No data to get prices for %s", cp.name)
            return Transaction.empty(currency_pair=cp)

        entry_price, entry_ts = entry.iloc[-1], entry.index[-1]
        exit_price, exit_ts = exit.iloc[0], exit.index[0]

        return Transaction(
            entry_price=entry_price,
            exit_price=exit_price,
            currency_pair=cp,
            entry_ts=entry_ts,
            exit_ts=exit_ts
        )

    def pumped_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        assert ts_price.index.is_monotonic_increasing
        entry: pd.Series = ts_price[ts_price.index <= pump.time - timedelta(minutes=15)]
        exit: pd.Series = ts_price[ts_price.index >= pump.time + timedelta(minutes=5)]

        if entry.empty or exit.empty:
            logging.info("No data to get prices for %s", cp.name)
            return Transaction.empty(currency_pair=cp)

        entry_price, entry_ts = entry.iloc[-1], entry.index[-1]
        exit_price, exit_ts = exit.iloc[0], exit.index[0]

        return Transaction(
            entry_price=entry_price,
            exit_price=exit_price,
            currency_pair=cp,
            entry_ts=entry_ts,
            exit_ts=exit_ts
        )
