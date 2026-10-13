import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pumps_and_dumps.backtest.portfolio.simulation import simulate_portfolio_paths


def _outcomes(returns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pump_hash": "event",
            "currency_pair": [f"ASSET{i}BTC" for i in range(len(returns))],
            "pump_time": "2020-01-01",
            "net_return": returns,
        }
    )


def _scores(outcomes: pd.DataFrame, values: list[float] | np.ndarray) -> pd.DataFrame:
    return outcomes[["pump_hash", "currency_pair"]].assign(score=values)


def test_weighted_sampling_frequency_and_equivalent_logits() -> None:
    outcomes = _outcomes([1.0, 0.0, 0.0]).assign(is_pumped=[True, False, False])
    probabilities = [0.8, 0.15, 0.05]
    result = simulate_portfolio_paths(
        {"probabilities": _scores(outcomes, probabilities), "logits": _scores(outcomes, np.log(probabilities))},
        outcomes,
        {"probabilities": "probability", "logits": "logit"},
        portfolio_size=1,
        n_simulations=6000,
    )
    events = result.event_returns.groupby("strategy")
    assert events["portfolio_return"].mean()["probabilities"] == pytest.approx(0.8, abs=0.02)
    assert events["portfolio_return"].mean()["uniform"] == pytest.approx(1 / 3, abs=0.02)
    np.testing.assert_array_equal(
        events.get_group("probabilities")["portfolio_return"], events.get_group("logits")["portfolio_return"]
    )
    np.testing.assert_array_equal(
        result.event_returns["portfolio_return"], result.event_returns["pumped_asset_in_portfolio"]
    )


def test_sparse_probabilities_fill_zero_weights_uniformly_without_replacement() -> None:
    outcomes = _outcomes([1.0, 0.0, -1.0])
    result = simulate_portfolio_paths(
        {"sparse": _scores(outcomes, [1, 0, 0]), "zero": _scores(outcomes, [0, 0, 0])},
        outcomes,
        {"sparse": "probability", "zero": "probability"},
        portfolio_size=2,
        n_simulations=2000,
    )
    events = result.event_returns.groupby("strategy")
    sparse = events.get_group("sparse")
    assert set(sparse["portfolio_return"]) == {0.0, 0.5}
    assert sparse["model_top1_in_portfolio"].all()
    assert sparse["portfolio_return"].mean() == pytest.approx(0.25, abs=0.02)
    np.testing.assert_array_equal(
        events.get_group("zero")["portfolio_return"], events.get_group("uniform")["portfolio_return"]
    )
    full = simulate_portfolio_paths(
        {"sparse": _scores(outcomes, [1, 0, 0])},
        outcomes,
        {"sparse": "probability"},
        portfolio_size=3,
        n_simulations=5,
    )
    assert full.paths["final_return"].eq(0).all()


@pytest.mark.parametrize(
    ("times", "returns", "expected_return", "expected_drawdown"),
    [
        (["2020-01-01", "2020-01-02", "2020-01-03"], [-0.2, 0.1, -0.3], -0.4, 0.4),
        (["2020-01-01", "2020-01-01"], [-0.2, 0.2], 0.0, 0.0),
    ],
)
def test_drawdown_includes_initial_capital_and_settles_simultaneous_events_together(
    times: list[str], returns: list[float], expected_return: float, expected_drawdown: float
) -> None:
    outcomes = pd.DataFrame(
        {
            "pump_hash": [f"event{i}" for i in range(len(times))],
            "currency_pair": "ABTC",
            "pump_time": times,
            "net_return": returns,
        }
    )
    result = simulate_portfolio_paths(
        {"model": _scores(outcomes, np.ones(len(outcomes)))},
        outcomes,
        {"model": "probability"},
        portfolio_size=1,
        n_simulations=3,
    )
    np.testing.assert_allclose(result.paths["final_return"], expected_return)
    np.testing.assert_allclose(result.paths["max_drawdown"], expected_drawdown)


def test_model_order_row_order_and_unexecuted_assets_do_not_change_draws() -> None:
    first = _outcomes([0.0, 0.5, -0.1]).assign(executed=[False, True, True])
    second = first.assign(pump_hash="second", pump_time="2020-01-02")
    outcomes = pd.concat([first, second], ignore_index=True)
    scored = {"a": _scores(outcomes, [1, 0, 0, 1, 0, 0]), "b": _scores(outcomes, [0, 1, 2, 2, 1, 0])}
    kinds = {"a": "probability", "b": "logit"}
    original = simulate_portfolio_paths(scored, outcomes, kinds, portfolio_size=1, n_simulations=30)
    reordered = simulate_portfolio_paths(
        {name: scored[name].sample(frac=1, random_state=2) for name in ["b", "a"]},
        outcomes.sample(frac=1, random_state=3).assign(is_pumped=True),
        kinds,
        portfolio_size=1,
        n_simulations=30,
    )
    assert_frame_equal(original.paths, reordered.paths)
    assert original.paths.loc[original.paths["strategy"].eq("a"), "final_return"].eq(0).all()
    assert reordered.event_returns["pumped_asset_in_portfolio"].all()


@pytest.mark.parametrize("problem", ["missing_key", "duplicate", "nan", "negative_probability", "too_small"])
def test_invalid_inputs_fail_explicitly(problem: str) -> None:
    outcomes = _outcomes([0.1, 0.2])
    scores = _scores(outcomes, [0.8, 0.2])
    size = 1
    if problem == "missing_key":
        scores = scores.iloc[:1]
    elif problem == "duplicate":
        scores = pd.concat([scores, scores.iloc[:1]])
    elif problem == "nan":
        scores.loc[0, "score"] = np.nan
    elif problem == "negative_probability":
        scores.loc[0, "score"] = -0.1
    elif problem == "too_small":
        size = 3
    with pytest.raises(ValueError):
        simulate_portfolio_paths({"model": scores}, outcomes, {"model": "probability"}, portfolio_size=size)
