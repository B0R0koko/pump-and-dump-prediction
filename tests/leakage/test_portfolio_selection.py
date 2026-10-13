"""Three-asset examples: outcomes may affect profit, never portfolio selection."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pumps_and_dumps.backtest.portfolio import simulation


@pytest.fixture(autouse=True)
def fixed_sampling_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix Gumbel noise so the real weighted sampler selects A and B from A/B/C."""

    class FixedNoise:
        def gumbel(self, *, size: tuple[int, int]) -> np.ndarray:
            assert size == (1, 3)
            return np.array([[0.2, 0.1, 0.0]])

    # Fix only randomness. The real sampler adds log weights and selects top K.
    # With weights A=.6, B=.3, C=.1, this draw selects A and B, never C.
    monkeypatch.setattr(simulation.np.random, "default_rng", lambda _: FixedNoise())


def outcomes(returns: list[float], labels: list[bool]) -> pd.DataFrame:
    """Create one pump's three candidate outcomes with explicitly supplied future data."""
    return pd.DataFrame(
        {
            "pump_hash": "pump",
            "currency_pair": ["ABTC", "BBTC", "CBTC"],
            "pump_time": "2020-01-01",
            "net_return": returns,
            "is_pumped": labels,
            "executed": True,
        }
    )


def simulate(frame: pd.DataFrame, allocation: str) -> simulation.SimulationResults:
    """Run one real two-asset portfolio draw with fixed scores 0.6, 0.3, and 0.1."""
    scores = frame[["pump_hash", "currency_pair"]].assign(score=[0.6, 0.3, 0.1])
    return simulation.simulate_portfolio_paths(
        {"model": scores},
        frame,
        {"model": "probability"},
        portfolio_size=2,
        n_simulations=1,
        allocation=allocation,
    )


def model_event(result: simulation.SimulationResults) -> pd.Series:
    """Extract the tested model's single event rather than a baseline strategy."""
    return result.event_returns.loc[result.event_returns["strategy"].eq("model")].iloc[0]


@pytest.mark.parametrize(("allocation", "expected"), [("equal", 0.1), ("probability", 0.2)])
def test_changing_true_label_changes_hit_flag_but_not_selected_portfolio(allocation: str, expected: float) -> None:
    """Use true labels for hit reporting without using them for asset selection.

    Fixed scores and Gumbel noise select assets A and B; their returns are
    0.4 and -0.2, while unselected C returns 9. Move the only true label
    among A, B, and C and expect hit flags True, True, and False respectively.
    Portfolio paths must remain identical for every label assignment.
    Parameterized allocation checks give (0.4 - 0.2) / 2 = 0.1 for equal
    weights and (2 * 0.4 - 0.2) / 3 = 0.2 for probability weights.
    This catches selecting the known true label or allowing labels to change
    position sizing, while confirming hit flags use the actual chosen assets.
    """
    frame = outcomes([0.4, -0.2, 9.0], [False, False, True])
    original = simulate(frame, allocation)

    # Probe each asset with the true label: A and B are selected; C is not.
    # The extreme return of C also makes accidental inclusion easy to detect.
    for labels, expected_hit in [
        ([True, False, False], True),
        ([False, True, False], True),
        ([False, False, True], False),
    ]:
        changed = simulate(frame.assign(is_pumped=labels), allocation)
        assert_frame_equal(original.paths, changed.paths)
        row = model_event(changed)
        assert bool(row["pumped_asset_in_portfolio"]) == expected_hit
        # Equal: (.4 - .2)/2 = .1. Weighted: (2*.4 - .2)/3 = .2.
        assert row["portfolio_return"] == pytest.approx(expected)


@pytest.mark.parametrize(("allocation", "before", "after"), [("equal", 0.1, -0.2), ("probability", 0.2, -0.4)])
def test_future_returns_change_profit_without_selecting_the_future_winner(
    allocation: str, before: float, after: float
) -> None:
    """Change realized returns without selecting the retrospectively best asset.

    Fixed scores and sampling noise select A and B even though C is the true
    label and initially returns 9. Replace returns [0.4, -0.2, 9] with
    [-0.8, 0.4, 100]; C must remain absent despite its exceptional payoff.
    Equal allocation changes profit from (0.4 - 0.2) / 2 = 0.1 to
    (-0.8 + 0.4) / 2 = -0.2. Probability allocation uses weights 2/3 and
    1/3, changing profit from 0.2 to -0.4. In both cases A remains present
    and the hit flag stays False, proving the expected A/B composition.
    This catches future-return sorting or sizing leaking into portfolio
    construction while checking that realized returns still drive profit.
    """
    frame = outcomes([0.4, -0.2, 9.0], [False, False, True])
    original = model_event(simulate(frame, allocation))
    changed = model_event(simulate(frame.assign(net_return=[-0.8, 0.4, 100.0]), allocation))

    # A+B remain selected even when C becomes spectacularly profitable.
    # Equal: (-.8 + .4)/2 = -.2. Weighted: (-2*.8 + .4)/3 = -.4.
    assert original["portfolio_return"] == pytest.approx(before)
    assert changed["portfolio_return"] == pytest.approx(after)
    assert bool(changed["model_top1_in_portfolio"])
    assert not bool(changed["pumped_asset_in_portfolio"])


@pytest.mark.parametrize(("allocation", "expected"), [("equal", 0.3), ("probability", 0.2)])
def test_unfilled_asset_keeps_its_cash_allocation_without_replacement(allocation: str, expected: float) -> None:
    """Keep a failed order's allocation as cash without a hindsight replacement.

    The real sampler chooses A and B, whose supplied returns are 0 and 0.6;
    unselected true-label asset C returns 9. Compare all orders filled with
    only A marked unfilled, keeping its realized return at zero in both runs.
    Expect identical paths: equal allocation earns 0/2 + 0.6/2 = 0.3, and
    probability allocation earns 0 * 2/3 + 0.6/3 = 0.2. A remains selected
    and C remains absent, so the true-label hit flag must stay False.
    This catches resampling after observing an execution failure and
    reallocating A's cash into B, which would incorrectly earn 0.6.
    """
    frame = outcomes([0.0, 0.6, 9.0], [False, False, True])
    filled = simulate(frame, allocation)
    unfilled = simulate(frame.assign(executed=[False, True, True]), allocation)

    # Missing A's fill leaves its capital in cash; C cannot replace it.
    # Equal: 0/2 + .6/2 = .3. Weighted: 0*2/3 + .6/3 = .2.
    # Renormalizing into B would incorrectly return .6.
    assert_frame_equal(filled.paths, unfilled.paths)
    row = model_event(unfilled)
    assert row["portfolio_return"] == pytest.approx(expected)
    assert bool(row["model_top1_in_portfolio"])
    assert not bool(row["pumped_asset_in_portfolio"])
