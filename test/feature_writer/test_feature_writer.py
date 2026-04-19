from datetime import datetime, timedelta

import numpy as np
import polars as pl

import features.FeatureWriter as feature_writer_module
from core.columns import TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.feature_type import FeatureType
from core.pump_event import PumpEvent
from core.time_utils import NamedTimeDelta
from features.FeatureWriter import PumpsFeatureWriter


def _build_writer(pump_events: list[PumpEvent]) -> PumpsFeatureWriter:
    writer = object.__new__(PumpsFeatureWriter)
    writer._pump_events = pump_events
    writer._pump_times_by_currency = {}
    for pump_event in sorted(pump_events, key=lambda event: event.time):
        writer.pump_times_by_currency.setdefault(pump_event.currency_pair.name, []).append(pump_event.time)
    return writer


def test_preprocess_data_for_currency_aggregates_ticks_into_trades() -> None:
    writer: PumpsFeatureWriter = _build_writer(pump_events=[])
    t1: datetime = datetime(2021, 1, 1, 8, 0, 0)
    t2: datetime = datetime(2021, 1, 1, 8, 1, 0)

    df_ticks: pl.DataFrame = pl.DataFrame(
        {
            TRADE_TIME: [t1, t1, t2, t2],
            PRICE: [10.0, 12.0, 20.0, 18.0],
            QUANTITY: [2.0, 1.0, 1.0, 3.0],
            IS_BUYER_MAKER: [False, True, True, True],
        }
    )

    df_trades: pl.DataFrame = writer.preprocess_data_for_currency(df=df_ticks)
    assert df_trades.shape[0] == 2

    row0 = df_trades.row(0, named=True)
    row1 = df_trades.row(1, named=True)

    assert np.isclose(row0["price_first"], 10.0)
    assert np.isclose(row0["price_last"], 12.0)
    assert np.isclose(row0["quote_abs"], 32.0)
    assert np.isclose(row0["quote_sign"], 8.0)
    assert np.isclose(row0["quantity_sign"], 1.0)
    assert bool(row0["is_long"]) is True
    assert np.isclose(row0["quote_slippage_abs"], 2.0)
    assert np.isclose(row0["quote_slippage_sign"], 2.0)

    assert np.isclose(row1["price_first"], 20.0)
    assert np.isclose(row1["price_last"], 18.0)
    assert np.isclose(row1["quote_abs"], 74.0)
    assert np.isclose(row1["quote_sign"], -74.0)
    assert np.isclose(row1["quantity_sign"], -4.0)
    assert bool(row1["is_long"]) is False
    assert np.isclose(row1["quote_slippage_abs"], 6.0)
    assert np.isclose(row1["quote_slippage_sign"], -6.0)
    assert np.isclose(row1["price_last_prev"], 12.0)
    assert row1["trade_time_prev"] == t1


def test_compute_features_matches_feature_definitions(monkeypatch) -> None:
    monkeypatch.setattr(feature_writer_module, "REGRESSOR_OFFSETS", [NamedTimeDelta.ONE_HOUR])
    monkeypatch.setattr(feature_writer_module, "DECAY_OFFSETS", [NamedTimeDelta.ONE_MINUTE])

    currency_pair: CurrencyPair = CurrencyPair.from_string("AAA-BTC")
    other_pair: CurrencyPair = CurrencyPair.from_string("BBB-BTC")

    pump_event: PumpEvent = PumpEvent(
        currency_pair=currency_pair,
        time=datetime(2021, 1, 1, 10, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    writer: PumpsFeatureWriter = _build_writer(
        pump_events=[
            PumpEvent(
                currency_pair=currency_pair,
                time=datetime(2020, 1, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            PumpEvent(
                currency_pair=other_pair,
                time=datetime(2020, 3, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            PumpEvent(
                currency_pair=currency_pair,
                time=datetime(2020, 6, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            PumpEvent(
                currency_pair=currency_pair,
                time=datetime(2021, 2, 1, 0, 0, 0),
                exchange=Exchange.BINANCE_SPOT,
            ),
            pump_event,
        ]
    )

    df: pl.DataFrame = pl.DataFrame(
        {
            TRADE_TIME: [
                datetime(2021, 1, 1, 7, 10, 0),
                datetime(2021, 1, 1, 7, 40, 0),
                datetime(2021, 1, 1, 7, 59, 0),  # excluded from 1H window before pump
                datetime(2021, 1, 1, 8, 0, 0),  # included lower boundary
                datetime(2021, 1, 1, 8, 30, 0),
                datetime(2021, 1, 1, 10, 0, 10),  # used for 1MIN target
                datetime(2021, 1, 1, 10, 0, 40),
            ],
            "price_first": [100.0, 101.0, 200.0, 102.0, 103.0, 110.0, 115.0],
            "price_last": [101.0, 102.0, 400.0, 103.0, 108.0, 115.0, 120.0],
            "price_last_prev": [99.0, 100.0, 1.0, 102.0, 103.0, 109.0, 114.0],
            "quote_abs": [10.0, 20.0, 1000.0, 30.0, 50.0, 60.0, 70.0],
            "quote_sign": [5.0, -5.0, 1000.0, 10.0, 20.0, 30.0, 35.0],
            "quote_slippage_abs": [1.0, 2.0, 100.0, 3.0, 5.0, 6.0, 7.0],
            "quote_slippage_sign": [1.0, -2.0, 100.0, 3.0, 2.0, 3.0, 4.0],
            "is_long": [True, False, True, True, True, True, True],
        }
    )

    features = writer.compute_features(df=df, currency_pair=currency_pair, pump_event=pump_event)
    window: NamedTimeDelta = NamedTimeDelta.ONE_HOUR
    rb: datetime = pump_event.time - timedelta(hours=1)
    lb: datetime = rb - window.get_td()

    df_window: pl.DataFrame = df.filter(pl.col(TRADE_TIME).is_between(lb, rb))
    assert df_window.shape[0] == 2

    expected_return = ((108.0 / 102.0) - 1.0) * 1e4
    expected_share_long = 1.0
    expected_powerlaw = 1.0 + 2.0 / np.log(50.0 / 30.0)
    expected_slippage_imbalance = (3.0 + 2.0) / (3.0 + 5.0)
    expected_flow_imbalance = (10.0 + 20.0) / (30.0 + 50.0)
    expected_num_trades = 2
    expected_target_return = ((120.0 / 109.0) - 1.0) * 1e4

    hourly = (
        df.group_by_dynamic(index_column=TRADE_TIME, period=timedelta(hours=1), every=timedelta(hours=1))
        .agg(
            asset_return_pips=(pl.col("price_last").last() / pl.col("price_first").first() - 1.0) * 1e4,
            quote_abs=pl.col("quote_abs").sum(),
        )
        .sort(TRADE_TIME)
    )
    asset_return_std = float(np.std(hourly["asset_return_pips"].to_numpy(), ddof=1))
    quote_abs_mean = float(np.mean(hourly["quote_abs"].to_numpy()))
    quote_abs_std = float(np.std(hourly["quote_abs"].to_numpy(), ddof=1))

    expected_asset_return_zscore = 588.2352941176471 / asset_return_std
    expected_quote_abs_zscore = (80.0 - quote_abs_mean) / quote_abs_std

    assert np.isclose(features[FeatureType.ASSET_RETURN.col_name(window)], expected_return)
    assert np.isclose(
        features[FeatureType.ASSET_RETURN_ZSCORE.col_name(window)],
        expected_asset_return_zscore,
    )
    assert np.isclose(
        features[FeatureType.QUOTE_ABS_ZSCORE.col_name(window)],
        expected_quote_abs_zscore,
    )
    assert np.isclose(features[FeatureType.SHARE_OF_LONG_TRADES.col_name(window)], expected_share_long)
    assert np.isclose(features[FeatureType.POWERLAW_ALPHA.col_name(window)], expected_powerlaw)
    assert np.isclose(
        features[FeatureType.SLIPPAGE_IMBALANCE.col_name(window)],
        expected_slippage_imbalance,
    )
    assert np.isclose(features[FeatureType.FLOW_IMBALANCE.col_name(window)], expected_flow_imbalance)
    assert np.isclose(features[FeatureType.NUM_TRADES.col_name(window)], expected_num_trades)
    assert features[FeatureType.NUM_PREV_PUMP.lower()] == 2
    assert np.isclose(features["target_return@1MIN"], expected_target_return)
