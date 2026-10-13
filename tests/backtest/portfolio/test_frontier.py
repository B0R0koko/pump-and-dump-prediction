import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pumps_and_dumps.backtest.portfolio.frontier import compute_portfolio_frontiers
from pumps_and_dumps.backtest.portfolio.simulation import simulate_portfolio_paths


def _outcomes(returns: list[list[float]], times: list[str] | None = None) -> pd.DataFrame:
    times = times or [f"2020-01-{i + 1:02}" for i in range(len(returns))]
    return pd.DataFrame(
        [
            {"pump_hash": f"event{i}", "currency_pair": f"ASSET{j}BTC", "pump_time": times[i], "net_return": value}
            for i, values in enumerate(returns)
            for j, value in enumerate(values)
        ]
    )


def _scores(outcomes: pd.DataFrame, values: list[float] | np.ndarray) -> pd.DataFrame:
    return outcomes[["pump_hash", "currency_pair"]].assign(score=values)


def test_nested_sizes_reproduce_existing_simulator_and_full_universe() -> None:
    outcomes = _outcomes([[0.8, -0.1, 0.0, 0.2, -0.3, 0.1], [-0.4, 0.0, 0.1, 0.3, -0.2, 0.1]])
    scored = {
        "logit": _scores(outcomes, [2, 1, 0, -1, -2, -3] * 2),
        "sparse": _scores(outcomes, [1, 0.2, 0, 0, 0, 0] * 2),
        "zero": _scores(outcomes, [0] * 12),
    }
    kinds = {"logit": "logit", "sparse": "probability", "zero": "probability"}
    result = compute_portfolio_frontiers(scored, outcomes, kinds, portfolio_sizes=[1, 3, 5, 6], n_simulations=41)
    for size in [1, 3, 5, 6]:
        previous = simulate_portfolio_paths(scored, outcomes, kinds, portfolio_size=size, n_simulations=41)
        current = result.sampled_paths.loc[result.sampled_paths["portfolio_size"].eq(size)].drop(
            columns="portfolio_size"
        )
        assert_frame_equal(current.reset_index(drop=True), previous.paths, check_exact=False, atol=1e-14, rtol=1e-14)
    full = result.sampled_paths.loc[result.sampled_paths["portfolio_size"].eq(6)]
    expected_return = outcomes.groupby("pump_hash")["net_return"].mean().sum()
    np.testing.assert_allclose(full["final_return"], expected_return)
    assert "uniform" not in set(result.deterministic["strategy"])
    assert set(result.sampled["strategy"]) == {*scored, "uniform"}


def test_deterministic_rank_ties_cash_and_simultaneous_settlement() -> None:
    outcomes = _outcomes(
        [[-0.2, 5], [0.2, -4], [0, 3], [-0.3, 2]],
        ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"],
    )
    outcomes["executed"] = True
    outcomes.loc[outcomes["net_return"].eq(0), "executed"] = False
    scored = {"model": _scores(outcomes, [1] * 8)}
    result = compute_portfolio_frontiers(scored, outcomes, {"model": "logit"}, portfolio_sizes=[1, 2], n_simulations=13)
    top1 = result.deterministic.loc[result.deterministic["portfolio_size"].eq(1)].iloc[0]
    assert top1["final_return"] == pytest.approx(-0.3)
    assert top1["max_drawdown"] == pytest.approx(0.3)
    shuffled = compute_portfolio_frontiers(
        {"model": scored["model"].sample(frac=1, random_state=2)},
        outcomes.sample(frac=1, random_state=3).assign(is_pumped=True),
        {"model": "logit"},
        portfolio_sizes=[2, 1],
        n_simulations=13,
    )
    assert_frame_equal(result.deterministic, shuffled.deterministic)
    assert_frame_equal(result.sampled_paths, shuffled.sampled_paths)


def test_sampled_points_are_medians_of_complete_paths_not_median_event_returns() -> None:
    outcomes = _outcomes([[0, 0, 0.8], [0, 0, 0.8], [-0.4, -0.4, 0]])
    scores = {"model": _scores(outcomes, [0] * 9)}
    result = compute_portfolio_frontiers(scores, outcomes, {"model": "logit"}, portfolio_sizes=[1], n_simulations=1001)
    old = simulate_portfolio_paths(scores, outcomes, {"model": "logit"}, portfolio_size=1, n_simulations=1001)
    model_paths = old.paths.loc[old.paths["strategy"].eq("model")]
    point = result.sampled.loc[result.sampled["strategy"].eq("model")].iloc[0]
    assert point["final_return"] == pytest.approx(model_paths["final_return"].median())
    assert point["max_drawdown"] == pytest.approx(model_paths["max_drawdown"].median())
    event_medians = (
        old.event_returns.loc[old.event_returns["strategy"].eq("model")]
        .groupby("pump_hash")["portfolio_return"]
        .median()
    )
    assert point["final_return"] != pytest.approx(event_medians.sum())


@pytest.mark.parametrize("sizes", [[], [0], [True], [1.5], [1, 1], [1, 3]])
def test_invalid_portfolio_sizes_raise(sizes: list) -> None:
    outcomes = _outcomes([[0.1, 0.2], [0.3, 0.4, 0.5]])
    with pytest.raises(ValueError):
        compute_portfolio_frontiers(
            {"model": _scores(outcomes, [0] * 5)}, outcomes, {"model": "logit"}, portfolio_sizes=sizes
        )
