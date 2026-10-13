"""Trade access used by feature generation, independent of feature calculations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pumps_and_dumps.core.columns import (
    DATE,
    IS_BUYER_MAKER,
    PRICE,
    QUANTITY,
    SYMBOL,
    TRADE_TIME,
)
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.market_data.hive_catalog import get_cross_section_currencies


class TradeSource(Protocol):
    def currency_pairs(self, bounds: Bounds) -> list[CurrencyPair]: ...

    def load(self, bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame: ...


@dataclass(frozen=True)
class HiveTradeSource:
    root: Path

    def currency_pairs(self, bounds: Bounds) -> list[CurrencyPair]:
        return [pair for pair in get_cross_section_currencies(hive_dir=self.root, bounds=bounds) if pair.term == "BTC"]

    def load(self, bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame:
        # Resolve only this asset's requested partitions; scanning the whole
        # multi-year archive per candidate makes a feature rebuild needlessly slow.
        paths = [
            path
            for day in bounds.date_range()
            for path in sorted((self.root / f"date={day}" / f"symbol={currency_pair.name}").glob("*.parquet"))
        ]
        if not paths:
            return _empty_trades()
        try:
            return _read_trades(paths, bounds, currency_pair)
        except (pl.exceptions.ComputeError, OSError) as error:
            raise _read_error(paths, bounds, currency_pair, error) from error


def _read_trades(paths: list[Path], bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame:
    return (
        pl.scan_parquet(paths, hive_partitioning=True)
        .filter(
            (pl.col(SYMBOL) == currency_pair.name)
            & (pl.col(DATE).is_between(bounds.day0, bounds.day1))
            & pl.col(TRADE_TIME)
            .cast(pl.Datetime("ns"))
            .is_between(bounds.start_inclusive, bounds.end_exclusive, closed="left")
        )
        .collect()
        .sort(by=TRADE_TIME, maintain_order=True)
    )


def _read_error(paths: list[Path], bounds: Bounds, currency_pair: CurrencyPair, error: Exception) -> OSError:
    """Locate broken metadata only after a failed read, keeping normal loads fast."""
    for path in paths:
        try:
            pq.read_metadata(path)
        except (pa.ArrowInvalid, OSError) as metadata_error:
            return OSError(
                f"Invalid Hive Parquet file: {path}. Rebuild this date/symbol partition from its raw trade ZIP. "
                f"{metadata_error}"
            )
    return OSError(
        f"Cannot read Hive trades for {currency_pair.name} from {bounds.day0} through {bounds.day1}: {error}"
    )


def _empty_trades() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            TRADE_TIME: pl.Datetime("ns"),
            PRICE: pl.Float64,
            QUANTITY: pl.Float64,
            IS_BUYER_MAKER: pl.Boolean,
            DATE: pl.Date,
            SYMBOL: pl.String,
        }
    )
