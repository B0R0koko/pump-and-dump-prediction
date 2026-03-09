from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from analysis.pipelines.BaseModel import ImplementsRank
from analysis.portfolio.PriceImpact import fit_price_impact_model, PriceImpactModel
from analysis.portfolio.TOPKPortfolio import TOPKPortfolio
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import Dataset, DatasetType
from core.columns import TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


class DummyModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> pd.Series:
        return pd.Series(np.ones(dataset.all_data().shape[0]))


def _build_synthetic_trades_for_impact() -> pd.DataFrame:
    rows = []
    start = datetime(2021, 1, 1, 0, 0, 0)
    notionals = np.linspace(100.0, 10_000.0, 40)

    for idx, notional in enumerate(notionals):
        base_price = 100.0
        qty = notional / (2 * base_price)
        impact_bps = 2.0 + 0.2 * np.sqrt(notional)

        # Two ticks with the exact same TRADE_TIME become one aggregated trade.
        # This mirrors feature calculation, which groups ticks by exact timestamp.
        buy_trade_time = start + timedelta(minutes=idx)
        sell_trade_time = buy_trade_time + timedelta(seconds=1)

        # Aggressor buy (is_buyer_maker=False), upward adverse move.
        vwap_buy = base_price * (1 + impact_bps / 1e4)
        second_buy_price = 2 * vwap_buy - base_price
        rows.append(
            {
                TRADE_TIME: buy_trade_time,
                PRICE: base_price,
                QUANTITY: qty,
                IS_BUYER_MAKER: False,
            }
        )
        rows.append(
            {
                TRADE_TIME: buy_trade_time,
                PRICE: second_buy_price,
                QUANTITY: qty,
                IS_BUYER_MAKER: False,
            }
        )

        # Aggressor sell (is_buyer_maker=True), downward adverse move.
        vwap_sell = base_price * (1 - impact_bps / 1e4)
        second_sell_price = 2 * vwap_sell - base_price
        rows.append(
            {
                TRADE_TIME: sell_trade_time,
                PRICE: base_price,
                QUANTITY: qty,
                IS_BUYER_MAKER: True,
            }
        )
        rows.append(
            {
                TRADE_TIME: sell_trade_time,
                PRICE: second_sell_price,
                QUANTITY: qty,
                IS_BUYER_MAKER: True,
            }
        )

    return pd.DataFrame(rows)


def test_fit_price_impact_model_produces_monotonic_impact_and_capacities() -> None:
    # Larger notionals should produce larger impact, and fills should be capped
    # by the side-specific historical capacity estimate.
    trades = _build_synthetic_trades_for_impact()
    model: PriceImpactModel = fit_price_impact_model(
        trades=trades, liquidity_quantile=0.8
    )

    assert model.num_bars == 80
    assert model.predict_impact_bps(
        side=1, notional_quote=5_000
    ) >= model.predict_impact_bps(side=1, notional_quote=200)
    assert model.predict_impact_bps(
        side=-1, notional_quote=5_000
    ) >= model.predict_impact_bps(side=-1, notional_quote=200)
    assert (
        model.estimate_fill_notional(side=1, intended_notional_quote=1_000_000)
        <= model.buy_capacity_quote + 1e-9
    )
    assert (
        model.estimate_fill_notional(side=-1, intended_notional_quote=1_000_000)
        <= model.sell_capacity_quote + 1e-9
    )


def test_topk_transaction_applies_price_impact_and_fill_ratio(monkeypatch) -> None:
    # Execution-aware pricing should move entry/exit away from the raw prices
    # and scale return by the matched fill ratio.
    cp = CurrencyPair.from_string("AAA-BTC")
    pump = PumpEvent(
        currency_pair=cp,
        time=datetime(2021, 1, 2, 12, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    dataset = Dataset(
        data=pd.DataFrame(
            {"currency_pair": ["AAA-BTC"], "pump_hash": [pump.as_pump_hash()]}
        ),
        feature_set=FeatureSet(
            numeric_features=[],
            categorical_features=[],
            target="",
            eval_fields=["currency_pair", "pump_hash"],
        ),
        ds_type=DatasetType.TEST,
    )
    manager = TOPKPortfolio(
        model=DummyModel(),
        portfolio_size=1,
        use_price_impact=True,
        order_notional_quote=100.0,
    )
    impact_model = PriceImpactModel(
        buy_intercept_bps=10.0,
        buy_slope_bps_per_sqrt_notional=0.0,
        sell_intercept_bps=10.0,
        sell_slope_bps_per_sqrt_notional=0.0,
        buy_capacity_quote=50.0,
        sell_capacity_quote=50.0,
        num_bars=100,
    )
    monkeypatch.setattr(
        TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model
    )

    ts_price = pd.Series(
        index=[pump.time - timedelta(minutes=20), pump.time + timedelta(minutes=1)],
        data=[100.0, 110.0],
    )
    tx = manager.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)

    assert tx.entry_price is not None and tx.exit_price is not None
    assert np.isclose(tx.entry_price, 100.1)
    assert np.isclose(tx.exit_price, 109.89)
    assert np.isclose(tx.entry_filled_notional_quote, 50.0)
    assert np.isclose(tx.exit_filled_notional_quote, 50.0)
    assert np.isclose(tx.fill_ratio, 0.5)
    expected_return = (109.89 / 100.1 - 1 - 0.002) * 0.5
    assert np.isclose(tx.transaction_return, expected_return)

    # Make sure dummy dataset object is still a valid input for rank interface usage.
    assert isinstance(DummyModel().rank(dataset=dataset), pd.Series)


def test_topk_transaction_uses_matched_executed_notional_for_both_vwaps(
    monkeypatch,
) -> None:
    # Both legs of the simulated round-trip must be priced off the same
    # executable notional, even when buy-side and sell-side liquidity differ.
    cp = CurrencyPair.from_string("AAA-BTC")
    pump = PumpEvent(
        currency_pair=cp,
        time=datetime(2021, 1, 2, 12, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    manager = TOPKPortfolio(
        model=DummyModel(),
        portfolio_size=1,
        use_price_impact=True,
        order_notional_quote=100.0,
    )
    impact_model = PriceImpactModel(
        buy_intercept_bps=0.0,
        buy_slope_bps_per_sqrt_notional=1.0,
        sell_intercept_bps=0.0,
        sell_slope_bps_per_sqrt_notional=2.0,
        buy_capacity_quote=80.0,
        sell_capacity_quote=50.0,
        num_bars=100,
    )
    monkeypatch.setattr(
        TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model
    )

    ts_price = pd.Series(
        index=[pump.time - timedelta(minutes=20), pump.time + timedelta(minutes=1)],
        data=[100.0, 110.0],
    )
    tx = manager.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)

    executed_notional_quote = 50.0
    expected_entry_price = 100.0 * (1.0 + np.sqrt(executed_notional_quote) / 1e4)
    expected_exit_price = 110.0 * (1.0 - 2.0 * np.sqrt(executed_notional_quote) / 1e4)

    assert np.isclose(tx.entry_filled_notional_quote, executed_notional_quote)
    assert np.isclose(tx.exit_filled_notional_quote, executed_notional_quote)
    assert np.isclose(tx.fill_ratio, 0.5)
    assert np.isclose(tx.entry_price, expected_entry_price)
    assert np.isclose(tx.exit_price, expected_exit_price)


def test_topk_portfolio_defaults_to_30_day_price_impact_lookback() -> None:
    # The execution-aware backtest should default to a 30-day pre-pump lookback.
    manager = TOPKPortfolio(model=DummyModel(), portfolio_size=1)
    assert manager.impact_lookback_days == 30
