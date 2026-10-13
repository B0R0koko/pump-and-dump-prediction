import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pumps_and_dumps.backtest.portfolio.allocation import (
    allocation_log_weights,
    portfolio_prefix_returns,
    selected_allocation_weights,
)
from pumps_and_dumps.backtest.portfolio.frontier import compute_portfolio_frontiers
from pumps_and_dumps.backtest.portfolio import simulation


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pump_hash": ["first"] * 3 + ["second"] * 3,
            "currency_pair": ["ABTC", "BBTC", "CBTC"] * 2,
            "pump_time": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "net_return": [0.0, 0.4, -0.8, -0.2, 0.6, 0.1],
            "executed": [False, True, True, True, True, True],
            "is_pumped": [True, False, False, False, True, False],
        }
    )


def test_selected_and_prefix_weights_remain_stable_for_extreme_scores_and_zeros() -> None:
    # The excluded outlier must not flatten the selected scores to zero weights.
    logs = np.array([1e300, -1001.0, -1000.0])
    selections = np.array([[1, 2, 0]])
    expected = np.array([1, np.e]) / (1 + np.e)
    np.testing.assert_allclose(selected_allocation_weights(logs, selections[:, :2], "probability"), [expected])
    returns = np.array([0.7, -0.2, 0.4])
    prefixes = portfolio_prefix_returns(returns, selections, np.array([1, 2, 3]), logs, "probability")
    np.testing.assert_allclose(prefixes, [[-0.2, expected @ returns[[1, 2]], 0.7]])

    tiny_logs = allocation_log_weights(np.array([1.0, 1e-320, 2e-320, 0.0]), "probability")
    np.testing.assert_allclose(
        selected_allocation_weights(tiny_logs, np.array([[1, 2, 3]]), "probability"), [[1 / 3, 2 / 3, 0]]
    )
    np.testing.assert_allclose(
        selected_allocation_weights(np.full(3, -np.inf), np.array([[0, 1, 2]]), "probability"), [[1 / 3] * 3]
    )
    np.testing.assert_allclose(
        portfolio_prefix_returns(returns, np.array([0, 1, 2]), np.array([1, 2, 3]), np.full(3, -np.inf), "probability"),
        np.cumsum(returns) / [1, 2, 3],
    )


def test_probability_allocation_preserves_unfilled_capital_as_cash() -> None:
    outcomes = _outcomes().iloc[:3]
    scored = {"model": outcomes[["pump_hash", "currency_pair"]].assign(score=[0.75, 0.25, 0.0])}
    result = simulation.simulate_portfolio_paths(
        scored, outcomes, {"model": "probability"}, portfolio_size=3, n_simulations=9, allocation="probability"
    )
    np.testing.assert_allclose(result.paths.loc[result.paths["strategy"].eq("model"), "final_return"], 0.1)
    np.testing.assert_allclose(result.paths.loc[result.paths["strategy"].eq("uniform"), "final_return"], -0.4 / 3)
    frontier = compute_portfolio_frontiers(
        scored, outcomes, {"model": "probability"}, portfolio_sizes=[1, 2, 3], n_simulations=9, allocation="probability"
    )
    np.testing.assert_allclose(frontier.deterministic["final_return"], [0.0, 0.1, 0.1])


def test_allocation_modes_use_identical_selected_assets_and_uniform_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = _outcomes()
    scored = {"model": outcomes[["pump_hash", "currency_pair"]].assign(score=[0.6, 0.3, 0.1] * 2)}
    draws = []
    sample = simulation._sample_indices

    def record_draws(*args):
        selections = sample(*args)
        draws.append(selections.copy())
        return selections

    monkeypatch.setattr(simulation, "_sample_indices", record_draws)
    equal = simulation.simulate_portfolio_paths(
        scored, outcomes, {"model": "probability"}, portfolio_size=2, n_simulations=31
    )
    count = len(draws)
    weighted = simulation.simulate_portfolio_paths(
        scored, outcomes, {"model": "probability"}, portfolio_size=2, n_simulations=31, allocation="probability"
    )
    for original, current in zip(draws[:count], draws[count:], strict=True):
        np.testing.assert_array_equal(original, current)
    assert_frame_equal(
        equal.paths.loc[equal.paths["strategy"].eq("uniform")],
        weighted.paths.loc[weighted.paths["strategy"].eq("uniform")],
    )
    assert not equal.paths["final_return"].equals(weighted.paths["final_return"])


def test_probability_frontiers_match_simulation_and_k1_matches_equal_allocation() -> None:
    outcomes = _outcomes()
    keys = outcomes[["pump_hash", "currency_pair"]]
    scored = {
        "classifier": keys.assign(score=[0.7, 0.2, 0.1] * 2),
        "ranker": keys.assign(score=[1, -2, 3] * 2),
        "sparse": keys.assign(score=[1, 0, 0] * 2),
        "zero": keys.assign(score=0.0),
    }
    kinds = {name: "logit" if name == "ranker" else "probability" for name in scored}
    frontier = compute_portfolio_frontiers(
        scored, outcomes, kinds, portfolio_sizes=[1, 2, 3], n_simulations=37, allocation="probability"
    )
    for size in [1, 2, 3]:
        result = simulation.simulate_portfolio_paths(
            scored, outcomes, kinds, portfolio_size=size, n_simulations=37, allocation="probability"
        )
        current = frontier.sampled_paths.loc[frontier.sampled_paths["portfolio_size"].eq(size)].drop(
            columns="portfolio_size"
        )
        assert_frame_equal(current.reset_index(drop=True), result.paths, check_exact=False, atol=1e-14, rtol=1e-14)
        if size == 1:
            equal = simulation.simulate_portfolio_paths(scored, outcomes, kinds, portfolio_size=size, n_simulations=37)
            assert_frame_equal(equal.paths, result.paths)


@pytest.mark.parametrize("evaluate", [simulation.simulate_portfolio_paths, compute_portfolio_frontiers])
def test_invalid_allocation_fails_explicitly(evaluate) -> None:
    with pytest.raises(ValueError, match="allocation"):
        evaluate({}, _outcomes(), {}, allocation="unknown")
