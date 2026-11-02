from datetime import datetime, timedelta
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.notebook_helpers import analysis


def _assets():
    records = []
    # Coincident events must aggregate before daily volatility and path intervals.
    for i, time in enumerate(["2021-01-01 10:00", "2021-01-01 12:00", "2021-01-04 10:00"]):
        for j, value in enumerate([0.0, -0.05, 0.1, 0.5]):
            records.append({"event_index": i, "pump_hash": str(i), "pump_time": pd.Timestamp(time),
                            "asset_index": j, "currency_pair": f"A{j}-BTC", "is_pumped": j == 3,
                            "executed": j != 0, "return": value * (i + 1)})
    return pd.DataFrame(records)


def test_uniform_distinct_reproducible_selections_and_cash():
    assets = _assets()
    result = analysis.resample_random_portfolios(assets, [1, 2, 4], n_resamples=4000, seed=17)
    again = analysis.resample_random_portfolios(assets.sample(frac=1, random_state=2), [1, 2, 4], n_resamples=4000, seed=17)
    np.testing.assert_array_equal(result["indices"], again["indices"])
    order = result["indices"]
    np.testing.assert_array_equal(np.sort(order, axis=-1), np.broadcast_to(np.arange(4), order.shape))
    for event in range(3):
        counts = np.bincount(order[:, event, 0], minlength=4)
        np.testing.assert_allclose(counts / 4000, np.full(4, .25), atol=.025)
        # Every member (including cash) has weight 1/4; never renormalize to filled legs.
        expected = assets.query("event_index == @event")["return"].mean()
        np.testing.assert_allclose(result["event_returns"][:, event, 2], expected)
    chosen = order[:, 0, :2]
    expected = assets.query("event_index == 0")["return"].to_numpy()[chosen].mean(axis=1)
    np.testing.assert_allclose(result["event_returns"][:, 0, 1], expected)
    with pytest.raises(ValueError, match="eligible universe"):
        analysis.resample_random_portfolios(assets, [5])


def test_metrics_use_each_draw_and_full_daily_calendar(tmp_path):
    from backtest.utils.evaluation import compute_portfolio_statistics

    result = analysis.resample_random_portfolios(_assets(), [2], n_resamples=150, seed=3)
    draws = result["event_returns"][:, :, 0]
    times = result["mean_returns"].index
    for draw in (0, 49, 149):
        expected = compute_portfolio_statistics(pd.DataFrame({"return": draws[draw]}, index=times)).iloc[0]
        actual = result["metrics"].set_index("replicate").loc[draw]
        np.testing.assert_allclose(actual[expected.index], expected)
    sharpes = result["metrics"]["Sharpe ratio"]
    assert result["summary"].loc[(2, "Sharpe ratio"), "mean"] == pytest.approx(sharpes.mean())
    mean_curve_sharpe = compute_portfolio_statistics(result["mean_returns"]).loc[2, "Sharpe ratio"]
    assert not np.isclose(sharpes.mean(), mean_curve_sharpe)
    daily = pd.DataFrame(draws.T, index=times).groupby(times.normalize()).sum().reindex(pd.date_range("2021-01-01", "2021-01-04"), fill_value=0)
    expected_lower = np.quantile(daily.cumsum().to_numpy(), .025, axis=1)
    np.testing.assert_allclose(result["paths"].lower, expected_lower)
    model = compute_portfolio_statistics(result["mean_returns"])
    model.insert(1, "cumulative return", result["mean_returns"].sum())
    comparison = analysis.compare_random_portfolio_metrics(model, result)
    assert comparison.loc[(2, "cumulative return"), "model minus random mean"] == pytest.approx(0, abs=1e-12)
    analysis.export_random_portfolios(result, tmp_path)
    saved = np.load(tmp_path / "resampled_assets.npz")
    np.testing.assert_array_equal(saved["asset_indices"], result["indices"])
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "asset_returns.parquet"), result["asset_returns"])


def test_narrow_execution_reads_match_original_including_late_exits():
    from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
    from backtest.portfolio.BasePortfolio import Portfolio
    from core.currency_pair import CurrencyPair
    from core.exchange import Exchange
    from core.pump_event import PumpEvent

    target = CurrencyPair.from_string("AAA-BTC")
    pump = PumpEvent(currency_pair=target, time=datetime(2021, 1, 2, 12), exchange=Exchange.BINANCE_SPOT)
    decision = pump.time - timedelta(minutes=15)
    pairs = [CurrencyPair.from_string(s) for s in ["AAA-BTC", "BBB-BTC", "CCC-BTC", "DDD-BTC"]]
    price_data = {
        "AAA-BTC": pd.Series([100., 999., 150.], index=[decision, pump.time, pump.time + timedelta(minutes=3)]),
        "BBB-BTC": pd.Series([100., 110.], index=[decision, pump.time]),
        "CCC-BTC": pd.Series([100.], index=[pump.time]),  # No valid entry: keep allocation as cash.
        "DDD-BTC": pd.Series([100.], index=[decision]),  # No exit before original cutoff.
    }
    rates = SimpleNamespace(get_quote_to_usdt_indicative_price=lambda quote_asset, ts: 1.0)
    manager = TOPKPortfolio(model=None, indicative_price_provider=rates)
    bounds_used = []

    def load(bounds, currency_pair):
        bounds_used.append(bounds)
        prices = price_data[currency_pair.name]
        return prices[(prices.index >= bounds.start_inclusive) & (prices.index < bounds.end_exclusive)]

    manager.load_price_ts = load
    rows = pd.DataFrame({"pump_hash": [pump.as_pump_hash()] * 4, "currency_pair": [cp.name for cp in pairs]})
    dataset = SimpleNamespace(all_data=lambda: rows, get_pumps=lambda: [pump])
    outcomes = analysis.random_portfolio_asset_returns(dataset, manager)
    assert outcomes.executed.tolist() == [True, True, False, False]
    np.testing.assert_allclose(outcomes["return"], [.4975, .0975, 0, 0])
    assert any(b.end_exclusive == pump.time + timedelta(days=1) for b in bounds_used)
    transactions = manager._create_cross_section_transactions(pump, Portfolio(pairs, np.full(4, .25)))
    original = [0 if tx.is_empty() else manager._pnl_calculator.calculate_transaction_pnl(tx) for tx in transactions]
    np.testing.assert_allclose(outcomes["return"], original)


def test_plot_random_line_and_band_share_dates(tmp_path):
    times = pd.date_range("2021-01-01", periods=4)
    returns = pd.DataFrame({"Model": [.1, 0, -.02, .05], "Random": [.02, 0, -.01, .01]}, index=times)
    band = pd.DataFrame({"lower": [-.1] * 4, "upper": [.1] * 4}, index=times)
    fig = analysis.plot_equity_curves(returns, random_col="Random", random_band=band, save_path=tmp_path / "pnl.png")
    try:
        line = next(line for line in fig.axes[0].lines if line.get_label() == "Random")
        assert line.get_linestyle() == "--"
        np.testing.assert_allclose(line.get_ydata(), returns.Random.cumsum())
        assert len(fig.axes[0].collections) == 1
        assert (tmp_path / "pnl.png").stat().st_size > 0
        fig.canvas.draw()
        legend_box = fig.axes[0].get_legend().get_window_extent()
        assert legend_box.y1 < fig.axes[0].get_window_extent().y0
        assert legend_box.y0 >= 0
    finally:
        plt.close(fig)


def test_impact_asset_returns_use_full_position_notional_and_normalize_cash_flows():
    from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
    from core.currency_pair import CurrencyPair
    from core.exchange import Exchange
    from core.pump_event import PumpEvent

    pump = PumpEvent(currency_pair=CurrencyPair.from_string("AAA-BTC"),
                     time=datetime(2021, 1, 2, 12), exchange=Exchange.BINANCE_SPOT)
    decision = pump.time - timedelta(minutes=15)
    calls = []

    class Impact:
        num_samples = 100

        def estimate_vwap_price(self, base_price, side, notional_quote):
            calls.append((side, notional_quote))
            impact_bps = 100 * np.sqrt(notional_quote / 1000)
            return base_price * (1 + side * impact_bps / 10000), impact_bps

    rates = SimpleNamespace(get_quote_to_usdt_indicative_price=lambda quote_asset, ts: 1.0)
    manager = TOPKPortfolio(model=None, use_price_impact=True, order_notional_usdt=1000,
                            indicative_price_provider=rates)
    manager._get_impact_model = lambda **kwargs: Impact()
    manager._manipulated_impact_provider = SimpleNamespace(get_impact_model=lambda **kwargs: Impact())
    prices = pd.Series([100., 110., 110.], index=[decision, pump.time, pump.time + timedelta(minutes=1)])
    manager.load_price_ts = lambda bounds, currency_pair: (
        prices.iloc[1:] if currency_pair.name == "CASH-BTC" else prices
    )
    rows = pd.DataFrame({"pump_hash": [pump.as_pump_hash()] * 3,
                         "currency_pair": ["AAA-BTC", "BBB-BTC", "CASH-BTC"]})
    dataset = SimpleNamespace(all_data=lambda: rows, get_pumps=lambda: [pump])
    outcomes = analysis.random_portfolio_asset_returns(dataset, manager)
    exit_notional = (1000 / 101) * 110
    expected = exit_notional * (1 - 0.01 * np.sqrt(exit_notional / 1000)) / 1000 - 1 - .0025
    np.testing.assert_allclose(outcomes["return"], [expected, expected, 0])
    np.testing.assert_allclose(outcomes.pnl_usdt, outcomes["return"] * 1000)
    assert calls == [(1, 1000), (-1, exit_notional), (1, 1000), (-1, exit_notional)]
    assert outcomes.entry_impact_bps.iloc[0] == 100
    assert outcomes.exit_impact_bps.iloc[0] > 100
    assert not outcomes.executed.iloc[2]
