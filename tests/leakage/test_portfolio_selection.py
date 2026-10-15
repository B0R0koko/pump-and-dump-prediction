"""Three-asset examples: future outcomes must not affect deterministic top-K selection."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.portfolio.base import Transaction
from pumps_and_dumps.backtest.portfolio.pnl import USDTPnLCalculator
from pumps_and_dumps.backtest.portfolio.selector import TopKPortfolioSelector


def outcomes(returns: list[float], labels: list[bool]) -> pd.DataFrame:
    """Create three candidates with fixed pre-pump scores and supplied future outcomes."""
    return pd.DataFrame(
        {
            "pump_hash": "pump",
            "currency_pair": ["A-BTC", "B-BTC", "C-BTC"],
            "score": [0.6, 0.3, 0.1],
            "net_return": returns,
            "is_pumped": labels,
            "executed": True,
        }
    )


def evaluate(frame: pd.DataFrame):
    """Use the real top-K selector and PnL aggregation on hand-checkable outcomes.

    Only the score column is a predictor. Prices are chosen so each filled
    transaction's net return after the real 25-basis-point fee equals the
    supplied value. Missing orders become the real empty transaction type.
    """
    dataset = Dataset(frame, FeatureSet(numeric_features=["score"], target="is_pumped"), DatasetType.TEST)
    model = SimpleNamespace(rank=lambda dataset: dataset.get_data()["score"].to_numpy())
    portfolio = TopKPortfolioSelector(portfolio_size=2).select_portfolio(model, dataset)
    selected = [pair.name for pair in portfolio.currency_pairs]
    indexed = frame.set_index("currency_pair")
    transactions = []
    for pair in portfolio.currency_pairs:
        row = indexed.loc[pair.name]
        transactions.append(
            Transaction(currency_pair=pair, entry_price=1.0, exit_price=1.0 + row.net_return + 0.0025)
            if row.executed
            else Transaction.empty(pair)
        )
    pnl = USDTPnLCalculator().calculate_portfolio_pnl(portfolio, transactions)
    hit = bool(indexed.loc[selected, "is_pumped"].any())
    return portfolio, pnl, hit


def test_changing_true_label_does_not_change_selected_assets_or_equal_weights() -> None:
    """Move the true label across A/B/C while keeping scores and returns fixed.

    Scores 0.6, 0.3, 0.1 must select A and B with weights one half each,
    regardless of which asset is manipulated. Their returns 0.4 and -0.2
    give exactly 0.1. Unselected C returns 9, making accidental inclusion
    obvious. The true-label hit changes only when the label moves to C;
    composition, capital weights and profit must remain unchanged.
    """
    frame = outcomes([0.4, -0.2, 9.0], [False, False, True])
    for labels, expected_hit in [
        ([True, False, False], True),
        ([False, True, False], True),
        ([False, False, True], False),
    ]:
        portfolio, pnl, hit = evaluate(frame.assign(is_pumped=labels))
        assert [pair.name for pair in portfolio.currency_pairs] == ["A-BTC", "B-BTC"]
        np.testing.assert_array_equal(portfolio.weights, [0.5, 0.5])
        assert pnl == pytest.approx(0.1)
        assert hit is expected_hit


def test_future_returns_change_profit_without_selecting_the_future_winner() -> None:
    """Change outcomes while requiring the same portfolio from pre-pump scores.

    Replace returns [0.4, -0.2, 9] with [-0.8, 0.4, 100]. Top-K must still
    choose A/B and allocate one half to each, despite C becoming the best
    asset in hindsight. Expected profit changes from 0.1 to -0.2 and both
    portfolios miss the true label C. This catches sorting or sizing from
    future returns while verifying that realized outcomes still drive PnL.
    """
    frame = outcomes([0.4, -0.2, 9.0], [False, False, True])
    before, before_pnl, before_hit = evaluate(frame)
    after, after_pnl, after_hit = evaluate(frame.assign(net_return=[-0.8, 0.4, 100.0]))
    assert before.currency_pairs == after.currency_pairs
    assert [pair.name for pair in after.currency_pairs] == ["A-BTC", "B-BTC"]
    np.testing.assert_array_equal(before.weights, after.weights)
    assert before_pnl == pytest.approx(0.1)
    assert after_pnl == pytest.approx(-0.2)
    assert before_hit is after_hit is False


def test_unfilled_asset_keeps_its_cash_allocation_without_replacement() -> None:
    """Keep an unfilled order's original half of capital in cash.

    Top-K chooses A/B. A has zero return and B earns 0.6, so total return is
    0.3. Marking A unfilled must retain its half of capital as cash and leave
    both selection and profit unchanged. C must remain absent even though
    its return is 9 and it is the manipulated asset. Redistributing A's
    capital to B would incorrectly earn 0.6; replacing A with C would also
    fail the explicit composition and false-hit checks.
    """
    frame = outcomes([0.0, 0.6, 9.0], [False, False, True])
    filled, filled_pnl, filled_hit = evaluate(frame)
    unfilled, unfilled_pnl, unfilled_hit = evaluate(frame.assign(executed=[False, True, True]))
    assert filled.currency_pairs == unfilled.currency_pairs
    assert [pair.name for pair in unfilled.currency_pairs] == ["A-BTC", "B-BTC"]
    np.testing.assert_array_equal(unfilled.weights, [0.5, 0.5])
    assert filled_pnl == unfilled_pnl == pytest.approx(0.3)
    assert filled_hit is unfilled_hit is False
