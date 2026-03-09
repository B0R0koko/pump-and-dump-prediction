from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.BasePortfolio import Portfolio, PortfolioStats, Transaction
from backtest.portfolio.PriceImpact import fit_price_impact_model, PriceImpactModel
from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
from backtest.utils.feature_set import FeatureSet
from backtest.utils.sample import Dataset, DatasetType
from core.columns import TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


class DummyModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> pd.Series:
        return pd.Series(np.ones(dataset.all_data().shape[0]))


class DummyQuoteToUSDTRateProvider:

    def __init__(self, rate: float = 20_000.0):
        self.rate: float = rate

    def get_quote_to_usdt_indicative_price(self, quote_asset: str, ts: datetime) -> float:
        assert quote_asset == "BTC"
        assert isinstance(ts, datetime)
        return self.rate


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

        # Aggressor buy (is_buyer_maker=False), upward adverse close-to-open move.
        second_buy_price = base_price * (1 + impact_bps / 1e4)
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

        # Aggressor sell (is_buyer_maker=True), downward adverse close-to-open move.
        second_sell_price = base_price * (1 - impact_bps / 1e4)
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
    model: PriceImpactModel = fit_price_impact_model(trades=trades, liquidity_quantile=0.8)

    assert model.num_bars == 80
    assert model.predict_impact_bps(side=1, notional_quote=5_000) >= model.predict_impact_bps(
        side=1, notional_quote=200
    )
    assert model.predict_impact_bps(side=-1, notional_quote=5_000) >= model.predict_impact_bps(
        side=-1, notional_quote=200
    )
    assert model.estimate_fill_notional(side=1, intended_notional_quote=1_000_000) <= model.buy_capacity_quote + 1e-9
    assert model.estimate_fill_notional(side=-1, intended_notional_quote=1_000_000) <= model.sell_capacity_quote + 1e-9


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
        indicative_price_provider=DummyQuoteToUSDTRateProvider(),
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
        indicative_price_provider=DummyQuoteToUSDTRateProvider(),
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
    monkeypatch.setattr(TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model)

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


def test_topk_transaction_converts_usdt_to_quote_before_price_impact(
    monkeypatch,
) -> None:
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
        order_notional_quote=0.0,
        order_notional_usdt=100.0,
        indicative_price_provider=DummyQuoteToUSDTRateProvider(),
    )
    impact_model = PriceImpactModel(
        buy_intercept_bps=0.0,
        buy_slope_bps_per_sqrt_notional=0.0,
        sell_intercept_bps=0.0,
        sell_slope_bps_per_sqrt_notional=0.0,
        buy_capacity_quote=np.inf,
        sell_capacity_quote=np.inf,
        num_bars=10,
    )
    monkeypatch.setattr(TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model)

    ts_price = pd.Series(
        index=[pump.time - timedelta(minutes=20), pump.time + timedelta(minutes=1)],
        data=[100.0, 110.0],
    )
    tx = manager.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)

    expected_notional_quote = 100.0 / 20_000.0
    assert np.isclose(tx.intended_notional_quote, expected_notional_quote)
    assert np.isclose(tx.entry_filled_notional_quote, expected_notional_quote)
    assert np.isclose(tx.exit_filled_notional_quote, expected_notional_quote)
    assert np.isclose(tx.fill_ratio, 1.0)


def test_portfolio_stats_pnl_is_usdt_denominated_when_conversion_available() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    tx = Transaction(
        currency_pair=cp,
        entry_price=100.0,
        exit_price=110.0,
        intended_notional_quote=2.0,
        exit_filled_notional_quote=2.0,
        exit_filled_notional_usdt=40_000.0,
    )
    portfolio = Portfolio(currency_pairs=[cp], weights=np.array([1.0]))
    pump = PumpEvent(
        currency_pair=cp,
        time=datetime(2021, 1, 2, 12, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    stats = PortfolioStats(portfolio=portfolio, txs=[tx], pump=pump)

    expected_tx_return = 110.0 / 100.0 - 1.0 - 0.002
    expected_pnl_usdt = expected_tx_return * 2.0 * (40_000.0 / 2.0)
    assert np.isclose(stats.pnl, expected_pnl_usdt)
