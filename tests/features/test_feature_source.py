"""Hive loading preserves causal cutoffs and identifies damaged partitions."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from pumps_and_dumps.core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.features.source import HiveTradeSource
from pumps_and_dumps.features.writer import PumpsFeatureWriter

CUTOFF = datetime(2021, 6, 1, 12, 3, 17, 123456)


def test_partition_loader_identifies_corrupt_file(tmp_path) -> None:
    pair = CurrencyPair.from_string("ETH-BTC")
    partition = tmp_path / "date=2021-06-01" / "symbol=ETH-BTC"
    partition.mkdir(parents=True)
    path = partition / "broken.parquet"
    path.write_bytes(b"PAR1unfinished data")
    with pytest.raises(OSError, match="Invalid Hive Parquet file") as error:
        HiveTradeSource(tmp_path).load(Bounds.for_day(CUTOFF.date()), pair)
    assert str(path) in str(error.value)


def test_partition_loader_preserves_exact_cutoff_without_scanning_other_assets(tmp_path, monkeypatch) -> None:
    partition = tmp_path / "date=2021-06-01" / "symbol=AAA-BTC"
    partition.mkdir(parents=True)
    pl.DataFrame(
        {
            TRADE_TIME: [
                datetime(2021, 6, 1, 12, 3, 17, 123000),
                datetime(2021, 6, 1, 12, 3, 17, 124000),
            ],
            PRICE: [10.0, 10.0],
            QUANTITY: [1.0, 1.0],
            IS_BUYER_MAKER: [False, False],
        }
    ).with_columns(pl.col(TRADE_TIME).cast(pl.Datetime("ms"))).write_parquet(partition / "trades.parquet")
    unrelated = tmp_path / "date=2021-06-01" / "symbol=BBB-BTC"
    unrelated.mkdir()
    (unrelated / "unreadable.parquet").write_bytes(b"must not be scanned")
    monkeypatch.setattr(Exchange, "get_hive_location", lambda self: tmp_path)
    writer = PumpsFeatureWriter([], feature_dir=tmp_path / "features")
    bounds = Bounds(CUTOFF - timedelta(seconds=1), CUTOFF)
    loaded = writer.load_data_for_currency_pair(bounds, CurrencyPair.from_string("AAA-BTC"))
    assert loaded.height == 1
    assert loaded[TRADE_TIME][0] < CUTOFF
    assert writer.load_data_for_currency_pair(bounds, CurrencyPair.from_string("MISSING-BTC")).is_empty()
