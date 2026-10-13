"""Future data must not change predictors or historical candidate membership."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
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
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.features.writer import (
    DECISION_LAG,
    NORMALIZER_LOOKBACK,
    PumpsFeatureWriter,
)


@pytest.mark.parametrize(
    "announcement_offset, expected",
    [
        (timedelta(microseconds=-1), 1),
        (timedelta(0), 0),
        (timedelta(seconds=1), 0),
        (DECISION_LAG - timedelta(microseconds=1), 0),
        (DECISION_LAG, 0),
        (timedelta(days=1), 0),
    ],
)
def test_previous_pump_count_uses_prediction_cutoff(announcement_offset: timedelta, expected: int) -> None:
    event = PumpEvent(
        CurrencyPair.from_string("AAA-BTC"),
        datetime(2021, 6, 1, 12, 20),
        Exchange.BINANCE_SPOT,
    )
    cutoff = event.time - DECISION_LAG
    candidate = CurrencyPair.from_string("BBB-BTC")
    writer = object.__new__(PumpsFeatureWriter)
    writer._pump_times_by_currency = {candidate.name: [cutoff + announcement_offset]}

    assert writer._num_prev_pumps(candidate, event) == expected


def _tick(
    symbol: str,
    timestamp: datetime,
    price: float,
    quantity: float,
    seller: bool = False,
) -> dict:
    return {
        SYMBOL: symbol,
        DATE: timestamp.date(),
        TRADE_TIME: timestamp,
        PRICE: price,
        QUANTITY: quantity,
        IS_BUYER_MAKER: seller,
    }


def _write_hive(root: Path, rows: list[dict]) -> None:
    for (day, symbol), frame in pl.DataFrame(rows).partition_by([DATE, SYMBOL], as_dict=True).items():
        partition = root / f"date={day}" / f"symbol={symbol}"
        partition.mkdir(parents=True, exist_ok=True)
        frame.drop(DATE, SYMBOL).write_parquet(partition / "trades.parquet")


@pytest.mark.parametrize(
    "pump_time",
    [
        datetime(2021, 6, 1, 12, 20, 37, 123456),
        datetime(2021, 6, 1, 0, 10),  # The prediction cutoff is on the previous day.
        datetime(2021, 6, 1, 0, 15),  # The prediction cutoff is exactly midnight.
    ],
)
def test_full_cross_section_is_invariant_to_future_data(tmp_path: Path, monkeypatch, pump_time: datetime) -> None:
    """Exercise Hive discovery, loading, preprocessing and every default regressor.

    The second dataset includes extreme future trades, a future-only listing,
    and newly announced pumps. Only the intentionally forward-looking labels
    may differ from the dataset available at the prediction cutoff.
    """
    cutoff = pump_time - DECISION_LAG
    event = PumpEvent(CurrencyPair.from_string("AAA-BTC"), pump_time, Exchange.BINANCE_SPOT)
    history = []
    for symbol, base_price in (("AAA-BTC", 10.0), ("BBB-BTC", 20.0)):
        for hour in range(int(NORMALIZER_LOOKBACK.total_seconds() / 3600)):
            timestamp = cutoff - timedelta(hours=hour + 1) + timedelta(minutes=10)
            price = base_price + hour * 0.01
            # Multiple fills at one timestamp exercise the aggregation and lag path.
            history.extend(
                [
                    _tick(symbol, timestamp, price, 1.0 + hour % 7),
                    _tick(symbol, timestamp, price + 0.001, 0.5, seller=True),
                ]
            )
        for minute in range(1, 61):
            history.append(
                _tick(
                    symbol,
                    cutoff - timedelta(minutes=minute),
                    base_price + minute * 0.003,
                    2.0,
                )
            )
        history.append(_tick(symbol, cutoff - timedelta(microseconds=1), base_price, 3.0))
    history.extend(
        [
            _tick("STALE-BTC", cutoff - timedelta(days=1, microseconds=1), 1.0, 1.0),
            _tick("OTHER-USDT", cutoff - timedelta(seconds=1), 1.0, 1.0),
        ]
    )
    future = []
    for symbol in ("AAA-BTC", "BBB-BTC", "FUTURE-BTC", "STALE-BTC"):
        for index, timestamp in enumerate(
            (
                cutoff,
                cutoff + timedelta(microseconds=1),
                pump_time - timedelta(seconds=1),
                pump_time,
                pump_time + timedelta(seconds=30),
                pump_time + timedelta(minutes=4),
                pump_time + timedelta(minutes=5),
                pump_time + timedelta(days=1),
            )
        ):
            future.append(_tick(symbol, timestamp, 1e6 * (index + 1), 1e8))
    past_event = PumpEvent(event.currency_pair, cutoff - timedelta(days=40), Exchange.BINANCE_SPOT)
    future_events = [
        PumpEvent(event.currency_pair, cutoff, Exchange.BINANCE_SPOT),
        PumpEvent(
            CurrencyPair.from_string("BBB-BTC"),
            cutoff + timedelta(seconds=1),
            Exchange.BINANCE_SPOT,
        ),
        PumpEvent(event.currency_pair, pump_time + timedelta(days=1), Exchange.BINANCE_SPOT),
    ]
    results = []
    for name, rows, events in (
        ("past_only", history, [past_event, event]),
        ("with_future", history + future, [event, *future_events, past_event]),
    ):
        root = tmp_path / name
        _write_hive(root, rows)
        monkeypatch.setattr(Exchange, "get_hive_location", lambda self: root)
        writer = PumpsFeatureWriter(pump_events=events, feature_dir=tmp_path / name / "features")
        frame = writer.create_cross_section(event, position=0)
        assert frame is not None
        results.append(frame.sort("currency_pair"))

    past, with_future = results
    assert past["currency_pair"].to_list() == ["AAA-BTC", "BBB-BTC"]
    assert with_future["currency_pair"].to_list() == past["currency_pair"].to_list()
    regressors = FeatureSet.auto().regressors
    assert_frame_equal(
        past.select("currency_pair", *regressors),
        with_future.select("currency_pair", *regressors),
        check_exact=True,
    )
    assert past["num_prev_pump"].to_list() == [1, 0]
    # Ensure the future rows really reached the label calculation: the return
    # targets must change while every selected model input remains unchanged.
    assert past["target_return@1MIN"].null_count() == 2
    assert with_future["target_return@1MIN"].null_count() == 0
