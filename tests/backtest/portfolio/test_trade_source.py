"""The shared partition reader preserves the portfolio's half-open trade window."""

from datetime import datetime, timedelta

import pandas as pd
import polars as pl

from pumps_and_dumps.backtest.portfolio.trades import HiveTradeSource
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.time_utils import Bounds


def test_partition_reader_matches_full_scan_and_ignores_unrelated_assets(tmp_path):
    start = datetime(2021, 5, 1, 23, 59, 59)
    end = start + timedelta(seconds=3)
    for day, times in (
        ("2021-05-01", [start - timedelta(seconds=1), start]),
        ("2021-05-02", [end, start + timedelta(seconds=2), start + timedelta(seconds=1)]),
    ):
        folder = tmp_path / f"date={day}" / "symbol=ETH-BTC"
        folder.mkdir(parents=True)
        pl.DataFrame(
            {
                "time": times,
                "price": list(range(len(times))),
                "quantity": [1.0] * len(times),
                "is_buyer_maker": [False] * len(times),
            }
        ).rename({"time": "trade_time"}).write_parquet(folder / "trades.parquet")
    bounds = Bounds(start, end)
    # Baseline is the former full-archive query, including its ordering and columns.
    expected = (
        pl.scan_parquet(tmp_path, hive_partitioning=True)
        .filter(
            (pl.col("symbol") == "ETH-BTC")
            & pl.col("date").is_between(bounds.day0, bounds.day1)
            & (pl.col("trade_time") >= start)
            & (pl.col("trade_time") < end)
        )
        .collect()
        .sort("trade_time")
        .select("trade_time", "price", "quantity", "is_buyer_maker")
        .to_pandas()
    )
    actual = HiveTradeSource(tmp_path).load_trades(bounds, CurrencyPair.from_string("ETH-BTC"))
    pd.testing.assert_frame_equal(actual, expected)
    assert len(actual) == 3
    unrelated = tmp_path / "date=2021-05-02" / "symbol=BNB-BTC"
    unrelated.mkdir()
    (unrelated / "broken.parquet").write_text("unrelated invalid file")
    pd.testing.assert_frame_equal(
        HiveTradeSource(tmp_path).load_trades(bounds, CurrencyPair.from_string("ETH-BTC")), expected
    )
