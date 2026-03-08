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

        # Aggressor buy (is_buyer_maker=False), upward adverse move.
        vwap_buy = base_price * (1 + impact_bps / 1e4)
        second_buy_price = 2 * vwap_buy - base_price
        t0 = start + timedelta(minutes=idx)
        rows.append({TRADE_TIME: t0, PRICE: base_price, QUANTITY: qty, IS_BUYER_MAKER: False})
        rows.append({TRADE_TIME: t0 + timedelta(seconds=20), PRICE: second_buy_price, QUANTITY: qty, IS_BUYER_MAKER: False})

        # Aggressor sell (is_buyer_maker=True), downward adverse move.
        vwap_sell = base_price * (1 - impact_bps / 1e4)
        second_sell_price = 2 * vwap_sell - base_price
        rows.append({TRADE_TIME: t0 + timedelta(seconds=40), PRICE: base_price, QUANTITY: qty, IS_BUYER_MAKER: True})
        rows.append({TRADE_TIME: t0 + timedelta(seconds=50), PRICE: second_sell_price, QUANTITY: qty, IS_BUYER_MAKER: True})

    return pd.DataFrame(rows)


def test_fit_price_impact_model_produces_monotonic_impact_and_capacities() -> None:
    trades = _build_synthetic_trades_for_impact()
    model: PriceImpactModel = fit_price_impact_model(trades=trades, bar_minutes=1, liquidity_quantile=0.8)

    assert model.num_bars > 0
    assert model.predict_impact_bps(side=1, notional_quote=5_000) >= model.predict_impact_bps(side=1, notional_quote=200)
    assert model.predict_impact_bps(side=-1, notional_quote=5_000) >= model.predict_impact_bps(side=-1, notional_quote=200)
    assert model.estimate_fill_notional(side=1, intended_notional_quote=1_000_000) <= model.buy_capacity_quote + 1e-9
    assert model.estimate_fill_notional(side=-1, intended_notional_quote=1_000_000) <= model.sell_capacity_quote + 1e-9


def test_topk_transaction_applies_price_impact_and_fill_ratio(monkeypatch) -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump = PumpEvent(
        currency_pair=cp,
        time=datetime(2021, 1, 2, 12, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    dataset = Dataset(
        data=pd.DataFrame({"currency_pair": ["AAA-BTC"], "pump_hash": [pump.as_pump_hash()]}),
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
    monkeypatch.setattr(TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model)

    ts_price = pd.Series(
        index=[pump.time - timedelta(minutes=20), pump.time + timedelta(minutes=1)],
        data=[100.0, 110.0],
    )
    tx = manager.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)

    assert tx.entry_price is not None and tx.exit_price is not None
    assert np.isclose(tx.entry_price, 100.1)
    assert np.isclose(tx.exit_price, 109.89)
    assert np.isclose(tx.fill_ratio, 0.5)
    expected_return = (109.89 / 100.1 - 1 - 0.002) * 0.5
    assert np.isclose(tx.transaction_return, expected_return)

    # Make sure dummy dataset object is still a valid input for rank interface usage.
    assert isinstance(DummyModel().rank(dataset=dataset), pd.Series)
