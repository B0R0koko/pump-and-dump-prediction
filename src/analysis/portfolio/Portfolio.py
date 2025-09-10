from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import List

import numpy as np
import pandas as pd
import polars as pl

from analysis.models.BaseModel import ImplementsRank
from core.columns import SYMBOL, DATE, TRADE_TIME, PRICE
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.time_utils import Bounds


@dataclass
class Portfolio:
    currency_pairs: List[CurrencyPair]
    weights: np.ndarray[float]

    def __repr__(self) -> str:
        return f"Portfolio: {dict(zip(self.currency_pairs, self.weights.round(4)))}"

    def get_weight(self, cp: CurrencyPair) -> float:
        """Return weight of the asset in the portfolio"""
        return self.weights[self.currency_pairs.index(cp)]


@dataclass
class Transaction:
    entry_price: float
    exit_price: float
    currency_pair: CurrencyPair
    entry_ts: datetime
    exit_ts: datetime


class ImplementsPortfolio(ABC):

    def __init__(self, model: ImplementsRank):
        self.model = model
        self._hive: pl.LazyFrame = pl.scan_parquet(
            Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True
        )

    @abstractmethod
    def create_portfolio(self, cross_section: pd.DataFrame) -> Portfolio:
        ...

    def load_price_ts(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.Series:
        """Loads price time series for a given currency pair"""
        data: pd.DataFrame = (
            self._hive
            .filter(
                (pl.col(SYMBOL) == currency_pair.name) &
                (pl.col(DATE).is_between(bounds.day0, bounds.day1)) &
                (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
            )
            .collect()
            .sort(by=TRADE_TIME)
            .select([TRADE_TIME, PRICE])
            .to_pandas()
        )
        return pd.Series(data=data[PRICE], index=data[TRADE_TIME])

    @abstractmethod
    def regular_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        """
        Define when we entry and exit for the regular asset which is not manipulated
        """

    @abstractmethod
    def pumped_transaction(self, ts_price: pd.Series, pump: PumpEvent, cp: CurrencyPair) -> Transaction:
        """
        Define when we exit and when we can enter for the manipulated asset
        """

    def _create_cross_section_transactions(self, pump: PumpEvent, portfolio: Portfolio) -> List[Transaction]:
        """
        Returns a list of transactions for cross-section
        """
        bounds: Bounds = Bounds(pump.time - timedelta(days=1), pump.time + timedelta(days=1))
        txs: List[Transaction] = []

        for cp in portfolio.currency_pairs:
            ts_price: pd.Series = self.load_price_ts(bounds=bounds, currency_pair=cp)

            tx: Transaction
            if not pump.is_manipulated(cp):
                tx = self.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)
            else:
                tx = self.pumped_transaction(ts_price=ts_price, pump=pump, cp=cp)
            txs.append(tx)
        return txs

    def calculate_return(self, txs: List[Transaction], portfolio: Portfolio) -> float:
        """
        Define the return of the portfolio
        """
        portfolio_return: float = 0
        tx: Transaction

        for tx in txs:
            asset_return: float = (tx.exit_price - tx.entry_price) / tx.entry_price
            portfolio_return += asset_return * portfolio.get_weight(tx.currency_pair)

        return portfolio_return

    def evaluate_cross_section(self, cross_section: pd.DataFrame, pump: PumpEvent) -> float:
        portfolio: Portfolio = self.create_portfolio(cross_section=cross_section)
        txs: List[Transaction] = self._create_cross_section_transactions(pump=pump, portfolio=portfolio)
        portfolio_return: float = self.calculate_return(txs=txs, portfolio=portfolio)
        return portfolio_return
