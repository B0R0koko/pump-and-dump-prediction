from functools import cached_property
from pathlib import Path

import pandas as pd
import polars as pl

from pumps_and_dumps.core.columns import (
    DATE,
    IS_BUYER_MAKER,
    PRICE,
    QUANTITY,
    SYMBOL,
    TRADE_TIME,
)
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.time_utils import Bounds


class HiveTradeSource:
    """Read sorted trades from Hive partitions, opening storage only when needed."""

    def __init__(self, path: Path | None = None):
        self.path = path

    @cached_property
    def _hive(self) -> pl.LazyFrame:
        path = (
            self.path
            if self.path is not None
            else Exchange.BINANCE_SPOT.get_hive_location()
        )
        return pl.scan_parquet(path, hive_partitioning=True)

    def load_trades(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        return (
            self._hive.filter(
                (pl.col(SYMBOL) == currency_pair.name)
                & pl.col(DATE).is_between(bounds.day0, bounds.day1)
                & (pl.col(TRADE_TIME) >= bounds.start_inclusive)
                & (pl.col(TRADE_TIME) < bounds.end_exclusive)
            )
            .collect()
            .sort(by=TRADE_TIME)
            .select([TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER])
            .to_pandas()
        )
