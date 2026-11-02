from pathlib import Path

import pandas as pd

from pumps_and_dumps.core.columns import (
    IS_BUYER_MAKER,
    PRICE,
    QUANTITY,
    TRADE_TIME,
)
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.features.source import HiveTradeSource as PartitionedTrades


class HiveTradeSource:
    """Read sorted trades from Hive partitions, opening storage only when needed."""

    def __init__(self, path: Path | None = None):
        self.path = path

    def load_trades(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        path = self.path if self.path is not None else Exchange.BINANCE_SPOT.get_hive_location()
        return (
            PartitionedTrades(path)
            .load(bounds, currency_pair)
            .select([TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER])
            .to_pandas()
        )
