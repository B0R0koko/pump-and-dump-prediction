"""Small, hand-checkable examples of the information available at decision time."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.core.time_utils import NamedTimeDelta
from pumps_and_dumps.features.calculation import compute_window_features, prepare_trades
from pumps_and_dumps.features.source import HiveTradeSource
from pumps_and_dumps.features.writer import DECISION_LAG, PumpsFeatureWriter


def _tick(time: datetime, price: float, quantity: float = 1.0, seller: bool = False) -> dict:
    return {"trade_time": time, "price": price, "quantity": quantity, "is_buyer_maker": seller}


def test_five_minute_features_exclude_cutoff_and_future_trades() -> None:
    """Compute literal five-minute features using only pre-cutoff information.

    A 12:00 cutoff has a prior price of 100 at 11:54 and in-window trades
    of 110 x 1 at 11:55 and 120 x 2 at 11:59. Expect two trades, a 1/2 buy
    share, return (120 / 100 - 1) * 10,000 = 2,000 basis points, and signed
    quote-volume imbalance (110 - 240) / (110 + 240) = -13/35.
    The left boundary is included; extreme trades at 12:00 and 12:00:01
    must be excluded. Compare every calculated feature before and after
    adding those future trades, including values using past normalizers.
    This catches right-boundary mistakes and future data affecting either
    the feature numerator or its reference price and normalization inputs.
    """
    cutoff = datetime(2021, 6, 1, 12)
    past = [
        _tick(cutoff - timedelta(minutes=6), 100.0),  # Previous price, outside the window.
        _tick(cutoff - timedelta(minutes=5), 110.0),  # Left boundary is included.
        _tick(cutoff - timedelta(minutes=1), 120.0, quantity=2.0, seller=True),
    ]
    future = [
        _tick(cutoff, 1_000_000.0, quantity=1_000_000.0),  # Right boundary is excluded.
        _tick(cutoff + timedelta(seconds=1), 2_000_000.0, quantity=2_000_000.0),
    ]
    outputs = []
    for rows in (past, past + future):
        trades = prepare_trades(pl.DataFrame(rows), side=1 - 2 * pl.col("is_buyer_maker"))
        features = compute_window_features(trades, cutoff, [NamedTimeDelta.FIVE_MINUTES])
        # Return: (120 / 100 - 1) * 10,000 = 2,000 basis points.
        # Two trades, one buy; signed quote volume = 110 - 240 out of 350.
        assert features["asset_return@5MIN"] == pytest.approx(2_000.0)
        assert features["num_trades@5MIN"] == 2
        assert features["share_of_long_trades@5MIN"] == 0.5
        assert features["flow_imbalance@5MIN"] == pytest.approx(-13 / 35)
        outputs.append(pl.DataFrame(features))

    # Include every calculated feature, including its past-only normalizer.
    assert_frame_equal(outputs[0], outputs[1], check_exact=True)


def test_cross_section_contains_only_assets_traded_before_cutoff(tmp_path: Path) -> None:
    """Build the eligible asset universe from the pre-decision trading window.

    Write tiny Hive partitions for an active asset AAA, EDGE exactly one day
    before cutoff, STALE one second earlier, and assets first traded at or
    after cutoff. Set the pump timestamp to cutoff + DECISION_LAG so the
    writer must recover the intended decision cutoff before reading trades.
    Expect exactly AAA and EDGE from the interval [cutoff - 1 day, cutoff).
    AAA's 100 -> 110 -> 120 prices give 2,000 basis points and two five-minute
    trades; its extreme cutoff trade is excluded and num_prev_pump is zero.
    This catches future listings entering the cross-section, incorrect
    eligibility boundaries, and counting the current pump as historical data.
    """
    cutoff = datetime(2021, 6, 1, 12)
    histories = {
        "AAA-BTC": [
            _tick(cutoff - timedelta(minutes=6), 100.0),
            _tick(cutoff - timedelta(minutes=5), 110.0),
            _tick(cutoff - timedelta(minutes=1), 120.0),
            _tick(cutoff, 1_000_000.0),
        ],
        "EDGE-BTC": [_tick(cutoff - timedelta(days=1), 10.0)],
        "STALE-BTC": [_tick(cutoff - timedelta(days=1, seconds=1), 10.0)],
        "ATCUTOFF-BTC": [_tick(cutoff, 1_000_000.0)],
        "FUTURE-BTC": [_tick(cutoff + timedelta(seconds=1), 1_000_000.0)],
    }
    for symbol, rows in histories.items():
        frame = pl.DataFrame(rows).with_columns(pl.col("trade_time").dt.date().alias("date"))
        for (day,), daily in frame.partition_by("date", as_dict=True).items():
            partition = tmp_path / f"date={day}" / f"symbol={symbol}"
            partition.mkdir(parents=True)
            daily.drop("date").write_parquet(partition / "trades.parquet")

    event = PumpEvent(CurrencyPair.from_string("AAA-BTC"), cutoff + DECISION_LAG, Exchange.BINANCE_SPOT)
    writer = PumpsFeatureWriter([event], data_source=HiveTradeSource(tmp_path), feature_dir=tmp_path / "features")
    output = writer.create_cross_section(event, position=0)

    assert output is not None
    # The eligibility window is [cutoff - 1 day, cutoff); a future listing
    # cannot enter the candidate universe merely because its partition exists.
    assert output["currency_pair"].sort().to_list() == ["AAA-BTC", "EDGE-BTC"]
    aaa = output.filter(pl.col("currency_pair") == "AAA-BTC").row(0, named=True)
    assert aaa["asset_return@5MIN"] == pytest.approx(2_000.0)
    assert aaa["num_trades@5MIN"] == 2
    assert aaa["num_prev_pump"] == 0  # The current event is still in the future.
