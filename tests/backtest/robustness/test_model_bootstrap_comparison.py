from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.robustness.significance import (
    _holm_adjust_p_values,
    compare_models_bootstrap_topk_percent_auc,
)
from pumps_and_dumps.backtest.datasets.columns import (
    COL_IS_PUMPED,
    COL_PROBAS_PRED,
    COL_PUMP_HASH,
)

BINS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _scored_with_ranks(ranks: list[int], size: int = 4) -> pd.DataFrame:
    """Keep the same assets/labels in every model and vary their score order."""
    rows = []
    for event, rank in enumerate(ranks):
        scores = np.arange(size, 0, -1, dtype=float)
        scores[0], scores[rank] = scores[rank], scores[0]
        for asset, score in enumerate(scores):
            rows.append(
                {
                    COL_PUMP_HASH: f"event-{event:03d}",
                    COL_IS_PUMPED: asset == 0,
                    COL_PROBAS_PRED: score,
                }
            )
    return pd.DataFrame(rows)


def _compare(
    models: dict[str, pd.DataFrame], **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = {"bins": BINS, "n_bootstrap": 999, "random_state": 7}
    parameters.update(kwargs)
    return compare_models_bootstrap_topk_percent_auc(models, **parameters)


def test_comparison_includes_all_models_and_all_unordered_pairs() -> None:
    models = {
        "weak": _scored_with_ranks([3] * 40),
        "strong": _scored_with_ranks([0] * 40),
        "mixed": _scored_with_ranks([0, 3] * 20),
        "near_mixed": _scored_with_ranks([0, 2] + [0, 3] * 19),
    }

    summary, pairwise = _compare(models)

    assert set(summary.index) == set(models)
    assert summary.index.name == "model"
    assert {
        "metric",
        "point_estimate",
        "mean_bootstrap",
        "std_bootstrap",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
        "alpha",
        "n_cross_sections",
    }.issubset(summary.columns)
    assert {
        "model_a",
        "model_b",
        "metric",
        "observed_diff",
        "mean_diff_bootstrap",
        "std_diff_bootstrap",
        "ci_lower",
        "ci_upper",
        "p_value",
        "p_value_holm",
        "alternative",
        "n_bootstrap",
        "alpha",
        "n_cross_sections",
        "reject_h0",
        "better_model",
    }.issubset(pairwise.columns)
    expected_pairs = {frozenset(pair) for pair in combinations(models, 2)}
    assert len(pairwise) == len(expected_pairs)
    assert {
        frozenset((row.model_a, row.model_b)) for row in pairwise.itertuples()
    } == expected_pairs
    assert (summary["n_cross_sections"] == 40).all()
    assert (pairwise["n_cross_sections"] == 40).all()
    assert (summary["n_bootstrap"] == 999).all()
    assert (pairwise["n_bootstrap"] == 999).all()
    assert (pairwise["alternative"] == "two-sided").all()
    assert summary.loc["strong", "point_estimate"] == pytest.approx(0.875)
    assert summary.loc["weak", "point_estimate"] == pytest.approx(0.125)
    assert summary.loc["mixed", "point_estimate"] == pytest.approx(0.5)
    assert (pairwise["p_value"] > 0).all()
    assert (pairwise["p_value_holm"] >= pairwise["p_value"]).all()
    assert (pairwise["p_value_holm"] <= 1).all()

    contrasts = pairwise.set_index(["model_a", "model_b"])
    strong_vs_weak = contrasts.loc[("weak", "strong")]
    assert strong_vs_weak["observed_diff"] == pytest.approx(-0.75)
    assert strong_vs_weak["ci_upper"] < 0
    assert strong_vs_weak["p_value_holm"] == pytest.approx(6 / 1000)
    assert strong_vs_weak["reject_h0"]
    assert strong_vs_weak["better_model"] == "strong"

    close_pair = contrasts.loc[("mixed", "near_mixed")]
    assert not close_pair["reject_h0"]
    assert close_pair["p_value_holm"] > 0.05
    assert pd.isna(close_pair["better_model"])


def test_identical_predictions_have_exactly_zero_difference_and_p_value_one() -> None:
    scored = _scored_with_ranks([0, 1, 2, 3] * 8)
    summary, pairwise = _compare({"original": scored, "identical": scored.copy()})

    assert summary.loc["original", "std_bootstrap"] > 0
    pd.testing.assert_series_equal(
        summary.loc["original"], summary.loc["identical"], check_names=False
    )
    result = pairwise.iloc[0]
    for field in (
        "observed_diff",
        "mean_diff_bootstrap",
        "std_diff_bootstrap",
        "ci_lower",
        "ci_upper",
    ):
        assert result[field] == 0
    assert result["p_value"] == 1
    assert result["p_value_holm"] == 1
    assert not result["reject_h0"]
    assert pd.isna(result["better_model"])


def test_common_event_draws_preserve_constant_paired_advantage() -> None:
    summary, pairwise = _compare(
        {
            "better": _scored_with_ranks([0, 1] * 20),
            "worse": _scored_with_ranks([1, 2] * 20),
        }
    )

    assert (summary["std_bootstrap"] > 0).all()
    result = pairwise.iloc[0]
    assert result["observed_diff"] == pytest.approx(0.25)
    assert result["mean_diff_bootstrap"] == pytest.approx(0.25)
    assert result["std_diff_bootstrap"] == pytest.approx(0)
    assert result["ci_lower"] == pytest.approx(0.25)
    assert result["ci_upper"] == pytest.approx(0.25)
    assert result["better_model"] == "better"


def test_two_sided_p_value_uses_the_centered_paired_null_distribution() -> None:
    n_bootstrap = 4095
    _, pairwise = _compare(
        {
            "a": _scored_with_ranks([0, 0, 0, 1], size=2),
            "b": _scored_with_ranks([1, 1, 1, 0], size=2),
        },
        bins=[0.0, 0.5, 1.0],
        n_bootstrap=n_bootstrap,
    )

    # Event effects are +0.5 three times and -0.5 once. For a resample with
    # k positive events, the effect is (k - 2) / 4 and its centered null
    # effect is (k - 3) / 4. Both null tails count towards the p-value.
    draws = np.random.default_rng(7).integers(0, 4, size=(n_bootstrap, 4))
    k_positive = (draws < 3).sum(axis=1)
    expected_p = (1 + np.count_nonzero(np.abs(k_positive - 3) >= 1)) / (n_bootstrap + 1)
    result = pairwise.iloc[0]
    assert result["observed_diff"] == pytest.approx(0.25)
    assert result["p_value"] == pytest.approx(expected_p)
    assert result["p_value_holm"] == result["p_value"]


def test_model_order_and_event_row_order_do_not_break_pairing() -> None:
    models = {
        "first": _scored_with_ranks([0, 1, 3, 2] * 10),
        "second": _scored_with_ranks([1, 0, 2, 3] * 10),
        "third": _scored_with_ranks([0, 0, 3, 3] * 10),
    }
    expected_summary, expected_pairwise = _compare(models)
    reordered = {
        name: frame.sample(frac=1, random_state=i)
        for i, (name, frame) in enumerate(models.items())
    }
    summary, pairwise = _compare(reordered)
    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(pairwise, expected_pairwise)

    reversed_summary, reversed_pairwise = _compare(
        dict(reversed(list(reordered.items())))
    )
    pd.testing.assert_frame_equal(
        reversed_summary.sort_index(), expected_summary.sort_index()
    )
    original_by_pair = expected_pairwise.set_index(["model_a", "model_b"])
    for result in reversed_pairwise.itertuples():
        original = original_by_pair.loc[(result.model_b, result.model_a)]
        assert result.observed_diff == pytest.approx(-original["observed_diff"])
        assert result.ci_lower == pytest.approx(-original["ci_upper"])
        assert result.ci_upper == pytest.approx(-original["ci_lower"])
        assert result.p_value == original["p_value"]
        assert result.p_value_holm == original["p_value_holm"]


def test_seed_reproduces_results_and_changes_monte_carlo_estimates() -> None:
    models = {
        "first": _scored_with_ranks([0, 1, 2, 3] * 10),
        "second": _scored_with_ranks([1, 0, 3, 2] * 10),
    }
    original = _compare(models, random_state=19)
    repeated = _compare(models, random_state=19)
    changed = _compare(models, random_state=20)
    for actual, expected in zip(repeated, original):
        pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_array_equal(
        changed[0]["point_estimate"], original[0]["point_estimate"]
    )
    assert not changed[0]["mean_bootstrap"].equals(original[0]["mean_bootstrap"])


def test_holm_adjustment_handles_unsorted_values_ties_and_clipping() -> None:
    np.testing.assert_allclose(
        _holm_adjust_p_values(np.array([0.04, 0.01, 0.03])), [0.06, 0.03, 0.06]
    )
    np.testing.assert_allclose(
        _holm_adjust_p_values(np.array([0.5, 0.5, 0.5])), [1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(_holm_adjust_p_values(np.array([0.02])), [0.02])


def test_decisions_use_holm_adjusted_p_values_across_the_full_family() -> None:
    models = {str(rank): _scored_with_ranks([rank] * 10) for rank in range(4)}
    _, pairwise = _compare(models, n_bootstrap=24)

    # At this deliberately small resample count all six raw p-values are
    # 1 / 25 = .04, but Holm adjusts each to .24. Pointwise CIs alone and
    # the uncorrected p-values would incorrectly declare six winners.
    np.testing.assert_allclose(pairwise["p_value"], 0.04)
    np.testing.assert_allclose(pairwise["p_value_holm"], 0.24)
    assert not pairwise["reject_h0"].any()
    assert pairwise["better_model"].isna().all()


def test_events_have_equal_weight_despite_different_asset_counts() -> None:
    small_event = _scored_with_ranks([0], size=2)
    large_event = _scored_with_ranks([3], size=4)
    large_event[COL_PUMP_HASH] = "event-001"
    scored = pd.concat([small_event, large_event], ignore_index=True)

    summary, _ = _compare({"a": scored, "b": scored.copy()}, bins=[0.0, 0.5, 1.0])

    # The event AUCs are .75 and .25, each with weight one half.
    np.testing.assert_allclose(summary["point_estimate"], 0.5)


@pytest.mark.parametrize(
    "parameters, expected_auc", [({}, 0.9875), ({"step": 0.07}, 0.825)]
)
def test_default_bins_normalize_only_the_requested_top_twenty_percent(
    parameters: dict, expected_auc: float
) -> None:
    summary, _ = compare_models_bootstrap_topk_percent_auc(
        {"first": _scored_with_ranks([0, 0]), "last": _scored_with_ranks([3, 3])},
        n_bootstrap=99,
        **parameters,
    )
    assert summary.loc["first", "point_estimate"] == pytest.approx(expected_auc)
    assert summary.loc["last", "point_estimate"] == 0


@pytest.mark.parametrize("n_models", [0, 1])
def test_comparison_requires_at_least_two_models(n_models: int) -> None:
    with pytest.raises(ValueError):
        _compare({str(i): _scored_with_ranks([0, 1]) for i in range(n_models)})


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_bootstrap": 0},
        {"n_bootstrap": 1},
        {"n_bootstrap": 2.5},
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": np.nan},
        {"bins": []},
        {"bins": [0]},
        {"bins": [0, 0.5, 0.5]},
        {"bins": [0, 1, 0.5]},
        {"bins": [-0.1, 0.5]},
        {"bins": [0, 1.1]},
        {"bins": [0, np.nan]},
        {"bins": [0, np.inf]},
        {"bins": None, "max_k_percent": 0},
        {"bins": None, "step": 0},
    ],
)
def test_invalid_inference_parameters_are_rejected(parameters: dict) -> None:
    scored = _scored_with_ranks([0, 1, 2, 3])
    with pytest.raises((ValueError, TypeError)):
        _compare({"a": scored, "b": scored.copy()}, **parameters)


def test_mismatched_event_sets_are_rejected_instead_of_taking_intersection() -> None:
    original = _scored_with_ranks([0, 1, 2])
    missing_event = original[original[COL_PUMP_HASH] != "event-002"]
    with pytest.raises(ValueError, match="[Cc]ross.sections|[Ee]vent"):
        _compare({"complete": original, "incomplete": missing_event})


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "single_event",
        "missing_column",
        "missing_event_id",
        "nan_score",
        "infinite_score",
        "invalid_label",
        "missing_positive",
        "different_event_size",
        "different_positive_count",
    ],
)
def test_invalid_scored_datasets_are_rejected(kind: str) -> None:
    valid = _scored_with_ranks([0, 1, 2])
    invalid = valid.copy()
    if kind == "empty":
        invalid = invalid.iloc[:0]
    elif kind == "single_event":
        valid = valid.iloc[:4]
        invalid = invalid.iloc[:4]
    elif kind == "missing_column":
        invalid = invalid.drop(columns=COL_PROBAS_PRED)
    elif kind == "missing_event_id":
        invalid.loc[0, COL_PUMP_HASH] = None
    elif kind == "nan_score":
        invalid.loc[0, COL_PROBAS_PRED] = np.nan
    elif kind == "infinite_score":
        invalid.loc[0, COL_PROBAS_PRED] = np.inf
    elif kind == "invalid_label":
        invalid[COL_IS_PUMPED] = invalid[COL_IS_PUMPED].astype(int)
        invalid.loc[0, COL_IS_PUMPED] = 2
    elif kind == "missing_positive":
        invalid.loc[invalid[COL_PUMP_HASH] == "event-001", COL_IS_PUMPED] = False
    elif kind == "different_event_size":
        invalid = invalid.drop(index=1)
    elif kind == "different_positive_count":
        invalid.loc[1, COL_IS_PUMPED] = True
    with pytest.raises((ValueError, KeyError)):
        _compare({"valid": valid, "invalid": invalid})
