from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.BasePortfolio import Portfolio, PortfolioStats, Transaction
from backtest.portfolio.PriceImpact import (
    PriceImpactModel,
    aggregate_trades_to_orders,
    fit_price_impact_model_with_diagnostics,
)
from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
from backtest.utils.feature_set import FeatureSet
from backtest.utils.sample import Dataset, DatasetType
from core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
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


class DummyDatasetForSweep:

    def __init__(self, pumps: list[PumpEvent]):
        self._pumps = pumps

    def get_pumps(self) -> list[PumpEvent]:
        return self._pumps


def _build_synthetic_trades(num_orders: int = 80) -> pd.DataFrame:
    """
    Build synthetic trades that simulate meta-orders of increasing size.
    Each meta-order has 1-3 fills at the same trade_time.
    """
    rows = []
    start = datetime(2021, 1, 1, 0, 0, 0)

    notionals = np.linspace(100.0, 10_000.0, num_orders)

    for idx, notional in enumerate(notionals):
        ts = start + timedelta(seconds=idx * 10)
        base_price = 100.0
        # Simulate price impact proportional to sqrt of notional
        impact_bps = 2.0 + 0.2 * np.sqrt(notional)

        is_buy = idx % 2 == 0
        quantity = notional / base_price

        if is_buy:
            # Buy-initiated: price moves up across fills
            end_price = base_price * (1 + impact_bps / 1e4)
            # Split into 2 fills
            rows.append({
                TRADE_TIME: ts,
                PRICE: base_price,
                QUANTITY: quantity * 0.6,
                IS_BUYER_MAKER: False,  # taker buy
            })
            rows.append({
                TRADE_TIME: ts,
                PRICE: end_price,
                QUANTITY: quantity * 0.4,
                IS_BUYER_MAKER: False,  # taker buy
            })
        else:
            # Sell-initiated: price moves down across fills
            end_price = base_price * (1 - impact_bps / 1e4)
            rows.append({
                TRADE_TIME: ts,
                PRICE: base_price,
                QUANTITY: quantity * 0.5,
                IS_BUYER_MAKER: True,  # taker sell
            })
            rows.append({
                TRADE_TIME: ts,
                PRICE: end_price,
                QUANTITY: quantity * 0.5,
                IS_BUYER_MAKER: True,  # taker sell
            })

    return pd.DataFrame(rows)


def test_aggregate_trades_to_orders_classifies_buy_and_sell() -> None:
    trades = pd.DataFrame([
        {TRADE_TIME: datetime(2021, 1, 1, 0, 0, 0), PRICE: 100.0, QUANTITY: 1.0, IS_BUYER_MAKER: False},
        {TRADE_TIME: datetime(2021, 1, 1, 0, 0, 0), PRICE: 101.0, QUANTITY: 1.0, IS_BUYER_MAKER: False},
        {TRADE_TIME: datetime(2021, 1, 1, 0, 0, 1), PRICE: 100.0, QUANTITY: 1.0, IS_BUYER_MAKER: True},
        {TRADE_TIME: datetime(2021, 1, 1, 0, 0, 1), PRICE: 99.0, QUANTITY: 1.0, IS_BUYER_MAKER: True},
    ])

    orders = aggregate_trades_to_orders(trades, quote_to_usdt=2.0)

    assert orders.shape[0] == 2
    assert orders.loc[0, "side"] == 1  # buy-initiated
    assert np.isclose(orders.loc[0, "impact_bps"], 100.0)  # (101/100 - 1) * 1e4
    assert orders.loc[1, "side"] == -1  # sell-initiated
    assert np.isclose(orders.loc[1, "impact_bps"], 100.0)  # -1 * (99/100 - 1) * 1e4
    # USDT notional = quote notional * quote_to_usdt
    assert np.isclose(orders.loc[0, "notional_usdt"], orders.loc[0, "notional_quote"] * 2.0)


def test_fit_price_impact_model_produces_monotonic_impact_and_capacities() -> None:
    trades = _build_synthetic_trades()
    result = fit_price_impact_model_with_diagnostics(trades=trades, liquidity_quantile=0.8, quote_to_usdt=1.0)
    model: PriceImpactModel = result.model

    assert model.num_trades > 0
    # Impact should increase with notional (sqrt is monotonic)
    assert model.predict_impact_bps(side=1, notional_quote=5_000) >= model.predict_impact_bps(
        side=1, notional_quote=200
    )
    assert model.predict_impact_bps(side=-1, notional_quote=5_000) >= model.predict_impact_bps(
        side=-1, notional_quote=200
    )
    # Capacity should cap fill notional (capacity is in USDT, converted back to quote)
    assert model.estimate_fill_notional(side=1, intended_notional_quote=1_000_000) <= model.buy_capacity_usdt / model.quote_to_usdt + 1e-9
    assert model.estimate_fill_notional(side=-1, intended_notional_quote=1_000_000) <= model.sell_capacity_usdt / model.quote_to_usdt + 1e-9

    diagnostics = result.diagnostics.set_index("side")
    assert diagnostics.loc["buy", "num_trades"] > 0
    assert diagnostics.loc["sell", "num_trades"] > 0


def test_vwap_impact_is_less_than_terminal_impact() -> None:
    trades = _build_synthetic_trades()
    result = fit_price_impact_model_with_diagnostics(trades=trades, liquidity_quantile=0.9, quote_to_usdt=1.0)
    model = result.model

    notional = 5000.0
    for side in [1, -1]:
        terminal = model.predict_impact_bps(side=side, notional_quote=notional)
        vwap = model.predict_vwap_impact_bps(side=side, notional_quote=notional)
        assert vwap <= terminal + 1e-6, f"VWAP impact should be <= terminal for side={side}"


def test_vwap_impact_closed_form_no_dead_zone() -> None:
    """When a >= 0, VWAP is the standard integral: a + 2/3 * beta * sqrt(Q)."""
    model = PriceImpactModel(
        buy_beta0=5.0, buy_beta1=0.3,
        sell_beta0=4.0, sell_beta1=0.2,
        buy_capacity_usdt=np.inf, sell_capacity_usdt=np.inf,
        quote_to_usdt=1.0, num_trades=100,
    )
    Q = 10000.0
    expected_buy = 5.0 + (2.0 / 3.0) * 0.3 * np.sqrt(Q)
    expected_sell = 4.0 + (2.0 / 3.0) * 0.2 * np.sqrt(Q)
    assert np.isclose(model.predict_vwap_impact_bps(side=1, notional_quote=Q), expected_buy)
    assert np.isclose(model.predict_vwap_impact_bps(side=-1, notional_quote=Q), expected_sell)


def test_vwap_impact_with_dead_zone() -> None:
    """When a < 0, orders below Q* = (a/beta)^2 have zero impact."""
    model = PriceImpactModel(
        buy_beta0=-10.0, buy_beta1=1.0,
        sell_beta0=0.0, sell_beta1=0.0,
        buy_capacity_usdt=np.inf, sell_capacity_usdt=np.inf,
        quote_to_usdt=1.0, num_trades=100,
    )
    # Q* = (10/1)^2 = 100. Below threshold → zero impact.
    assert model.predict_impact_bps(side=1, notional_quote=50.0) == 0.0
    assert model.predict_vwap_impact_bps(side=1, notional_quote=50.0) == 0.0

    # Above threshold → positive impact
    assert model.predict_impact_bps(side=1, notional_quote=400.0) > 0.0
    assert model.predict_vwap_impact_bps(side=1, notional_quote=400.0) > 0.0

    # VWAP < terminal (averaging over the dead zone region)
    Q = 400.0
    terminal = model.predict_impact_bps(side=1, notional_quote=Q)
    vwap = model.predict_vwap_impact_bps(side=1, notional_quote=Q)
    assert vwap < terminal


def test_usdt_normalised_regression_is_consistent() -> None:
    """Same real-value order (in USDT) should give same impact regardless of quote_to_usdt."""
    trades = _build_synthetic_trades()
    result_1x = fit_price_impact_model_with_diagnostics(trades=trades, quote_to_usdt=1.0)
    result_2x = fit_price_impact_model_with_diagnostics(trades=trades, quote_to_usdt=2.0)

    # Query same USDT value: 3000 quote @ 1x = 1500 quote @ 2x
    impact_1x = result_1x.model.predict_impact_bps(side=1, notional_quote=3000.0)
    impact_2x = result_2x.model.predict_impact_bps(side=1, notional_quote=1500.0)
    assert abs(impact_1x - impact_2x) < 5.0, f"Same USDT value should give similar impact: {impact_1x} vs {impact_2x}"


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
    rate = 20_000.0
    manager = TOPKPortfolio(
        model=DummyModel(),
        portfolio_size=1,
        use_price_impact=True,
        order_notional_quote=100.0,
        indicative_price_provider=DummyQuoteToUSDTRateProvider(rate=rate),
    )
    # Constant 10bps impact model with capacity of 50 quote (= 50 * rate USDT)
    impact_model = PriceImpactModel(
        buy_beta0=10.0,
        buy_beta1=0.0,
        sell_beta0=10.0,
        sell_beta1=0.0,
        buy_capacity_usdt=50.0 * rate,
        sell_capacity_usdt=50.0 * rate,
        quote_to_usdt=rate,
        num_trades=100,
    )
    monkeypatch.setattr(TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model)

    ts_price = pd.Series(
        index=[pump.time - timedelta(minutes=20), pump.time + timedelta(minutes=1)],
        data=[100.0, 110.0],
    )
    tx = manager.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)

    assert tx.entry_price is not None and tx.exit_price is not None
    assert np.isclose(tx.entry_price, 100.1, atol=0.05)
    assert np.isclose(tx.exit_price, 109.89, atol=0.05)
    assert np.isclose(tx.entry_filled_notional_quote, 50.0)
    assert np.isclose(tx.exit_filled_notional_quote, 50.0)
    assert np.isclose(tx.fill_ratio, 0.5)
    assert isinstance(DummyModel().rank(dataset=dataset), pd.Series)


def test_topk_portfolio_defaults_to_14_day_price_impact_lookback() -> None:
    manager = TOPKPortfolio(model=DummyModel(), portfolio_size=1)
    assert manager.impact_lookback_days == 14


def test_topk_transaction_converts_usdt_to_quote_before_price_impact(
    monkeypatch,
) -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump = PumpEvent(
        currency_pair=cp,
        time=datetime(2021, 1, 2, 12, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    rate = 20_000.0
    manager = TOPKPortfolio(
        model=DummyModel(),
        portfolio_size=1,
        use_price_impact=True,
        order_notional_quote=0.0,
        order_notional_usdt=100.0,
        indicative_price_provider=DummyQuoteToUSDTRateProvider(rate=rate),
    )
    # Zero-impact model with infinite capacity
    impact_model = PriceImpactModel(
        buy_beta0=0.0,
        buy_beta1=0.0,
        sell_beta0=0.0,
        sell_beta1=0.0,
        buy_capacity_usdt=np.inf,
        sell_capacity_usdt=np.inf,
        quote_to_usdt=rate,
        num_trades=10,
    )
    monkeypatch.setattr(TOPKPortfolio, "_get_impact_model", lambda self, pump, cp: impact_model)

    ts_price = pd.Series(
        index=[pump.time - timedelta(minutes=20), pump.time + timedelta(minutes=1)],
        data=[100.0, 110.0],
    )
    tx = manager.regular_transaction(ts_price=ts_price, pump=pump, cp=cp)

    expected_notional_quote = 100.0 / rate
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

    expected_tx_return = 110.0 / 100.0 - 1.0 - 0.0025
    expected_pnl_usdt = expected_tx_return * 2.0 * (40_000.0 / 2.0)
    assert np.isclose(stats.pnl, expected_pnl_usdt)


def test_evaluate_pnl_for_quantities_returns_execution_diagnostics(monkeypatch) -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump = PumpEvent(
        currency_pair=cp,
        time=datetime(2021, 1, 2, 12, 0, 0),
        exchange=Exchange.BINANCE_SPOT,
    )
    dataset = DummyDatasetForSweep([pump])
    manager = TOPKPortfolio(
        model=DummyModel(),
        portfolio_size=1,
        indicative_price_provider=DummyQuoteToUSDTRateProvider(rate=1.0),
    )
    portfolio = Portfolio(currency_pairs=[cp], weights=np.array([1.0]))

    def fake_evaluate_for_pump(self, dataset, pump):
        intended_usdt = float(self.config.order_notional_usdt)
        fill_ratio = min(1.0, 500.0 / intended_usdt)
        executed_usdt = intended_usdt * fill_ratio
        tx = Transaction(
            currency_pair=cp,
            entry_price=100.0,
            exit_price=110.0,
            intended_notional_quote=intended_usdt,
            entry_filled_notional_quote=executed_usdt,
            exit_filled_notional_quote=executed_usdt,
            entry_filled_notional_usdt=executed_usdt,
            exit_filled_notional_usdt=executed_usdt,
            entry_impact_bps=2.0 * intended_usdt / 100.0,
            exit_impact_bps=3.0 * intended_usdt / 100.0,
            entry_impact_num_bars=1440,
            exit_impact_num_bars=60,
            fill_ratio=fill_ratio,
        )
        return PortfolioStats(portfolio=portfolio, txs=[tx], pump=pump)

    monkeypatch.setattr(TOPKPortfolio, "evaluate_for_pump", fake_evaluate_for_pump)

    result = manager.evaluate_pnl_for_quantities(dataset=dataset, quantities_usdt=[100.0, 1000.0])

    assert list(result["quantity_usdt"]) == [100.0, 1000.0]
    assert np.isclose(result.loc[0, "mean_fill_ratio"], 1.0)
    assert np.isclose(result.loc[1, "mean_fill_ratio"], 0.5)
    assert np.isclose(result.loc[0, "mean_executed_notional_usdt"], 100.0)
    assert np.isclose(result.loc[1, "mean_executed_notional_usdt"], 500.0)
    assert result.loc[0, "mean_roe"] > result.loc[1, "mean_roe"]
