from datetime import datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_PUMP_HASH
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.portfolio.indicative_price import IndicativePriceProvider
from pumps_and_dumps.backtest.portfolio.outcomes import (
    HiveOutcomeTradeSource,
    collect_portfolio_outcomes,
    collect_validation_outcomes,
)
from pumps_and_dumps.backtest.portfolio.pnl import USDTPnLCalculator
from pumps_and_dumps.backtest.portfolio.topk import TOPKPortfolio
from pumps_and_dumps.core.columns import (
    BINANCE_KLINES_COLS,
    CLOSE_PRICE,
    OPEN_TIME,
    PRICE,
    QUOTE_ASSET_VOLUME,
    TRADE_TIME,
    VOLUME,
)
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.pump_event import PumpEvent


def _pump():
    return PumpEvent(CurrencyPair.from_string("AAA-BTC"), datetime(2021, 1, 2, 0, 5), Exchange.BINANCE_SPOT)


def _dataset(symbols, ds_type=DatasetType.VALIDATION):
    return Dataset(
        pd.DataFrame({COL_PUMP_HASH: [_pump().as_pump_hash()] * len(symbols), COL_CURRENCY_PAIR: symbols}),
        FeatureSet.empty(),
        ds_type,
    )


def _write_trades(root, symbol, prices):
    frame = pd.DataFrame({TRADE_TIME: prices.index, PRICE: prices.values})
    for day, rows in frame.groupby(frame[TRADE_TIME].dt.date):
        directory = root / f"date={day}" / f"symbol={symbol}"
        directory.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(rows).write_parquet(directory / "part-0.parquet")


def _write_fx(root):
    # The active entry/exit minute must not supply its not-yet-completed VWAP.
    times = pd.to_datetime(["2021-01-01 23:49", "2021-01-01 23:50", "2021-01-02 00:04", "2021-01-02 00:05"])
    frame = pd.DataFrame(0.0, index=range(4), columns=BINANCE_KLINES_COLS)
    frame[OPEN_TIME] = times.as_unit("ms").astype("int64")
    frame[VOLUME] = 1.0
    frame[QUOTE_ASSET_VOLUME] = [10_000.0, 999_999.0, 11_000.0, 12_000.0]
    frame[CLOSE_PRICE] = frame[QUOTE_ASSET_VOLUME]
    directory = root / "BTC-USDT"
    directory.mkdir(parents=True)
    for day, rows in frame.groupby(times.date):
        rows.to_csv(directory / f"klines@1m@{day}.zip", index=False, header=False, compression="zip")


@pytest.mark.parametrize("dataset_type", [DatasetType.VALIDATION, DatasetType.TEST])
def test_hive_outcomes_match_topk_timing_completed_fx_and_fee_and_preserve_cash_slots(tmp_path, dataset_type):
    pump = _pump()
    decision = pump.time - timedelta(minutes=15)
    prices = pd.Series(
        [9.0, 100.0, 999.0, 110.0, 130.0, 777.0],
        index=[
            decision - timedelta(seconds=1),
            decision,
            decision,
            pump.time,
            pump.time + timedelta(minutes=1),
            pump.time + timedelta(days=1),
        ],
    )
    hive = tmp_path / "hive"
    for symbol in ("AAA-BTC", "BBB-BTC"):
        _write_trades(hive, symbol, prices)
    _write_trades(hive, "NOENTRY-BTC", prices.iloc[3:])
    _write_trades(hive, "NOEXIT-BTC", prices.iloc[[1, 5]])
    _write_fx(tmp_path / "klines")
    provider = IndicativePriceProvider(tmp_path / "klines")
    symbols = ["AAA-BTC", "BBB-BTC", "NOENTRY-BTC", "NOEXIT-BTC", "MISSING-BTC"]
    progress = []
    outcomes = collect_portfolio_outcomes(
        _dataset(symbols, dataset_type),
        trade_source=HiveOutcomeTradeSource(hive),
        price_provider=provider,
        max_workers=2,
        progress=lambda completed, total: progress.append((completed, total)),
    ).set_index(COL_CURRENCY_PAIR)
    manager = TOPKPortfolio(model=None, indicative_price_provider=provider, order_notional_usdt=1.0)
    for symbol in ("AAA-BTC", "BBB-BTC"):
        pair = CurrencyPair.from_string(symbol)
        transaction_fn = manager.pumped_transaction if pump.is_manipulated(pair) else manager.regular_transaction
        transaction = transaction_fn(prices.iloc[:-1], pump, pair)
        row = outcomes.loc[symbol]
        assert row["entry_ts"] == transaction.entry_ts == decision
        assert row["exit_ts"] == transaction.exit_ts
        assert row["net_return"] == pytest.approx(USDTPnLCalculator().calculate_transaction_pnl(transaction))
        assert row["executed"]
    assert outcomes.loc["AAA-BTC", "net_return"] == pytest.approx(1.3 * 1.2 - 1 - 0.0025)
    assert outcomes.loc["BBB-BTC", "net_return"] == pytest.approx(1.1 * 1.1 - 1 - 0.0025)
    cash = outcomes.loc[["NOENTRY-BTC", "NOEXIT-BTC", "MISSING-BTC"]]
    assert cash["net_return"].eq(0).all()
    assert not cash["executed"].any()
    assert len(outcomes) == len(symbols)
    assert progress == [(1, 1)]


def test_corrupt_hive_partition_raises_with_asset_and_path(tmp_path):
    path = tmp_path / "date=2021-01-02" / "symbol=AAA-BTC" / "broken.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")
    with pytest.raises(OSError, match="AAA-BTC.*broken.parquet"):
        collect_validation_outcomes(_dataset(["AAA-BTC"]), trade_source=HiveOutcomeTradeSource(tmp_path), max_workers=1)


def test_missing_quote_conversion_raises_instead_of_dropping_candidate(tmp_path):
    pump = _pump()
    prices = pd.Series([100.0, 130.0], index=[pump.time - timedelta(minutes=15), pump.time + timedelta(minutes=1)])
    _write_trades(tmp_path / "hive", "AAA-BTC", prices)
    with pytest.raises(FileNotFoundError, match="quote asset BTC"):
        collect_validation_outcomes(
            _dataset(["AAA-BTC"]),
            trade_source=HiveOutcomeTradeSource(tmp_path / "hive"),
            price_provider=IndicativePriceProvider(tmp_path / "missing-klines"),
            max_workers=1,
        )


@pytest.mark.parametrize("ds_type", [DatasetType.TRAIN, DatasetType.TEST])
def test_outcomes_reject_nonvalidation_data(ds_type):
    with pytest.raises(ValueError, match="validation dataset"):
        collect_validation_outcomes(_dataset(["AAA-BTC"], ds_type))


def test_generic_outcomes_reject_train():
    with pytest.raises(ValueError, match="validation or test"):
        collect_portfolio_outcomes(_dataset(["AAA-BTC"], DatasetType.TRAIN))


def test_outcomes_reject_duplicate_candidates():
    with pytest.raises(ValueError, match="duplicate event/asset"):
        collect_validation_outcomes(_dataset(["AAA-BTC", "AAA-BTC"]))
