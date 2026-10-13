from itertools import permutations
from math import comb

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PROBAS_PRED,
    COL_PUMP_HASH,
)
from pumps_and_dumps.backtest.robustness.analytical_selection import analytical_portfolio_metrics


def _event(scores: list[float], event: str = "event", pumped: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_PUMP_HASH: event,
            COL_CURRENCY_PAIR: [f"ASSET{i}-BTC" for i in range(len(scores))],
            COL_IS_PUMPED: np.arange(len(scores)) == pumped,
            COL_PROBAS_PRED: scores,
        }
    )


def _sequential_hit_probability(weights: np.ndarray, positives: set[int], size: int) -> float:
    """Enumerate the weighted draw law, removing each selected asset."""
    size = min(size, len(weights))
    hit_probability = 0.0
    for sequence in permutations(range(len(weights)), size):
        if positives.isdisjoint(sequence):
            continue
        remaining = set(range(len(weights)))
        probability = 1.0
        for asset in sequence:
            total = sum(weights[i] for i in remaining)
            probability *= weights[asset] / total if total else 1 / len(remaining)
            remaining.remove(asset)
        hit_probability += probability
    return hit_probability


def test_score_concentration_changes_exact_auc_even_when_rankings_match() -> None:
    concentrated = _event([0.97, 0.015, 0.01, 0.005])
    diffuse = _event([0.28, 0.26, 0.24, 0.22])
    results = analytical_portfolio_metrics(
        {"concentrated": concentrated, "diffuse": diffuse},
        {"concentrated": "probability", "diffuse": "probability"},
        topk_bins=(1,),
    )

    np.testing.assert_array_equal(
        np.argsort(concentrated[COL_PROBAS_PRED]),
        np.argsort(diffuse[COL_PROBAS_PRED]),
    )
    assert results.topk_matrices["concentrated"][0, 0] == pytest.approx(0.97)
    assert results.topk_matrices["diffuse"][0, 0] == pytest.approx(0.28)
    # Four assets need a single draw throughout 0 < percentage <= 20%.
    # Trapezoidal integration starts at zero and reaches p at 0.5%.
    assert results.event_aucs.loc["event", "concentrated"] == pytest.approx(0.97 * 0.9875)
    assert results.event_aucs.loc["event", "diffuse"] == pytest.approx(0.28 * 0.9875)


def test_uniform_draws_without_replacement_clamp_to_available_assets() -> None:
    results = analytical_portfolio_metrics(
        {"uniform": _event([1.0] * 5)},
        {"uniform": "probability"},
        topk_bins=(0, 1, 3, 5, 8),
        topk_percent_bins=(0.0, 0.2, 0.6, 1.0),
    )

    expected = [0.0, 0.2, 0.6, 1.0, 1.0]
    np.testing.assert_allclose(results.topk_matrices["uniform"][0], expected, rtol=1e-14)
    np.testing.assert_allclose(results.topk_percent_matrices["uniform"][0], expected[:4], rtol=1e-14)


def test_multiple_uniform_true_labels_follow_hypergeometric_probability() -> None:
    frame = _event([1.0] * 8)
    frame[COL_IS_PUMPED] = [True, False, True, False, True, False, False, False]
    result = analytical_portfolio_metrics({"model": frame}, {"model": "probability"}, topk_bins=tuple(range(10)))
    expected = [1 - comb(5, size) / comb(8, size) if size <= 5 else 1.0 for size in range(10)]
    np.testing.assert_allclose(result.topk_matrices["model"][0], expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("kind", ["probability", "logit"])
@pytest.mark.parametrize(
    ("probabilities", "positives"),
    [
        ([0.2, 0.3, 0.5], {0, 2}),
        ([0.04, 0.09, 0.16, 0.3, 0.41], {0, 3}),
        ([0.003, 0.006, 0.09, 0.2, 0.701], {0}),
        ([0.005, 0.7, 0.001, 0.09, 0.204], {0, 2}),
    ],
)
def test_positive_labels_match_enumeration_of_weighted_sequences_without_replacement(
    kind: str, probabilities: list[float], positives: set[int]
) -> None:
    probabilities = np.asarray(probabilities)
    scores = probabilities if kind == "probability" else np.log(probabilities)
    frame = _event(scores.tolist())
    frame[COL_IS_PUMPED] = [i in positives for i in range(len(frame))]
    bins = tuple(range(len(frame) + 2))
    results = analytical_portfolio_metrics({"model": frame}, {"model": kind}, topk_bins=bins)
    expected = [_sequential_hit_probability(probabilities, positives, size) for size in bins]
    np.testing.assert_allclose(results.topk_matrices["model"][0], expected, rtol=1e-6, atol=1e-8)


def test_same_true_label_mass_depends_on_negative_weight_concentration() -> None:
    frames = {"concentrated": _event([0.2, 0.79, 0.01]), "diffuse": _event([0.2, 0.4, 0.4])}
    result = analytical_portfolio_metrics(frames, dict.fromkeys(frames, "probability"), topk_bins=(1, 2))
    concentrated, diffuse = (result.topk_matrices[name][0] for name in frames)
    assert concentrated[0] == pytest.approx(diffuse[0], rel=1e-14)
    assert concentrated[1] > diffuse[1]
    for name, frame in frames.items():
        expected = _sequential_hit_probability(frame[COL_PROBAS_PRED].to_numpy(), {0}, 2)
        assert result.topk_matrices[name][0, 1] == pytest.approx(expected, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("positive_weights", [[0.001], [0.2], [0.13, 0.08]])
def test_equal_negative_rates_match_closed_form_for_larger_universe(positive_weights: list[float]) -> None:
    negative_count, negative_weight = 80, 0.03
    frame = _event(positive_weights + [negative_weight] * negative_count)
    frame[COL_IS_PUMPED] = np.arange(len(frame)) < len(positive_weights)
    bins = (0, 1, 2, 10, 30, 79, 80, 81, 100)
    result = analytical_portfolio_metrics({"model": frame}, {"model": "probability"}, topk_bins=bins)
    positive_weight = sum(positive_weights)
    expected = [
        (
            1
            - np.prod(
                [
                    (negative_count - draw)
                    * negative_weight
                    / (positive_weight + (negative_count - draw) * negative_weight)
                    for draw in range(size)
                ]
            )
            if size <= negative_count
            else 1.0
        )
        for size in bins
    ]
    np.testing.assert_allclose(result.topk_matrices["model"][0], expected, rtol=1e-6, atol=1e-8)


def test_percentage_sizes_round_up_and_auc_preserves_grid() -> None:
    results = analytical_portfolio_metrics(
        {"uniform": _event([1.0] * 10)},
        {"uniform": "probability"},
        topk_percent_bins=(0.0, 0.001, 0.1, 0.101, 0.2, 1.0),
    )
    np.testing.assert_allclose(
        results.topk_percent_matrices["uniform"][0],
        [0.0, 0.1, 0.1, 0.2, 0.2, 1.0],
        rtol=1e-14,
    )
    # 0..0.005 rises to .1, .005..0.1 is .1, .1..0.105 rises to .2,
    # and .105..0.2 stays .2.
    expected_auc = (0.005 * 0.1 / 2 + 0.095 * 0.1 + 0.005 * (0.1 + 0.2) / 2 + 0.095 * 0.2) / 0.2
    assert results.event_aucs.loc["event", "uniform"] == pytest.approx(expected_auc)


def test_equivalent_scores_are_independent_of_model_event_and_asset_order() -> None:
    probabilities = pd.concat(
        [_event([0.3, 0.1, 0.6], "z", pumped=2), _event([0.8, 0.15, 0.05], "a")],
        ignore_index=True,
    )
    logits = probabilities.copy()
    logits[COL_PROBAS_PRED] = np.log(logits[COL_PROBAS_PRED]) + 17.0
    frames = {"probability": probabilities, "logit": logits, "duplicate": probabilities.copy()}
    kinds = {"probability": "probability", "logit": "logit", "duplicate": "probability"}
    original = analytical_portfolio_metrics(frames, kinds)
    shuffled = analytical_portfolio_metrics(
        {name: frame.sample(frac=1, random_state=13) for name, frame in reversed(frames.items())},
        kinds,
    )

    assert original.event_aucs.index.tolist() == ["a", "z"]
    pd.testing.assert_frame_equal(shuffled.event_aucs[original.event_aucs.columns], original.event_aucs)
    for name in frames:
        np.testing.assert_allclose(original.event_aucs[name], original.event_aucs["probability"], rtol=1e-14)
        for field in ("topk_matrices", "topk_percent_matrices"):
            matrices = getattr(original, field)
            np.testing.assert_allclose(matrices[name], matrices["probability"], rtol=1e-14)
            np.testing.assert_array_equal(getattr(shuffled, field)[name], matrices[name])


def test_zero_weights_use_uniform_draws_after_positive_weight_support_is_exhausted() -> None:
    frames = {
        "sparse": _event([1.0, 0.0, 0.0, 0.0], pumped=1),
        "zeros": _event([0.0] * 4, pumped=1),
        "uniform": _event([1.0] * 4, pumped=1),
        "certain": _event([0.0, 1.0, 0.0, 0.0], pumped=1),
    }
    results = analytical_portfolio_metrics(
        frames,
        dict.fromkeys(frames, "probability"),
        topk_bins=(0, 1, 2, 4, 100),
    )

    np.testing.assert_allclose(results.topk_matrices["sparse"][0], [0, 0, 1 / 3, 1, 1], rtol=1e-14)
    np.testing.assert_array_equal(results.topk_matrices["certain"][0], [0, 1, 1, 1, 1])
    np.testing.assert_allclose(results.topk_matrices["zeros"][0], [0, 0.25, 0.5, 1, 1], rtol=1e-14)
    np.testing.assert_array_equal(results.topk_matrices["zeros"], results.topk_matrices["uniform"])


def test_multiple_zero_weight_labels_match_sequential_uniform_fallback() -> None:
    weights = np.array([0.6, 0.4, 0.0, 0.0, 0.0])
    frame = _event(weights.tolist())
    frame[COL_IS_PUMPED] = [False, False, True, False, True]
    bins = tuple(range(7))
    result = analytical_portfolio_metrics({"model": frame}, {"model": "probability"}, topk_bins=bins)
    expected = [_sequential_hit_probability(weights, {2, 4}, size) for size in bins]
    np.testing.assert_allclose(result.topk_matrices["model"][0], expected, rtol=1e-14, atol=1e-14)


def test_classifier_scores_normalize_per_event_including_subnormal_scores() -> None:
    frame = pd.concat(
        [_event([0.2, 0.1], "a"), _event([0.6, 0.3], "b"), _event([2e-320, 1e-320], "c")],
        ignore_index=True,
    )
    result = analytical_portfolio_metrics({"model": frame}, {"model": "probability"}, topk_bins=(1, 2))
    np.testing.assert_allclose(result.topk_matrices["model"], [[2 / 3, 1]] * 3, rtol=1e-14)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([np.finfo(float).max, -np.finfo(float).max], [0, 1, 1]),
        ([-np.finfo(float).max, np.finfo(float).max], [0, 0, 1]),
        ([np.finfo(float).max, np.finfo(float).max], [0, 0.5, 1]),
        ([-np.finfo(float).max, -np.finfo(float).max], [0, 0.5, 1]),
    ],
)
def test_extreme_finite_logits_remain_stable(scores: list[float], expected: list[float]) -> None:
    with np.errstate(all="raise"):
        results = analytical_portfolio_metrics({"model": _event(scores)}, {"model": "logit"}, topk_bins=(0, 1, 2))
    np.testing.assert_allclose(results.topk_matrices["model"][0], expected)


def test_tiny_hit_probability_and_huge_k_preserve_precision() -> None:
    results = analytical_portfolio_metrics(
        {"tiny": _event([1e-20, 1.0])},
        {"tiny": "probability"},
        topk_bins=(0, 1, 2, 10, 10**20, 10**100),
    )
    np.testing.assert_allclose(
        results.topk_matrices["tiny"][0],
        [0, 1e-20, 1, 1, 1, 1],
        rtol=1e-14,
        atol=0,
    )


def test_extreme_logit_gaps_preserve_order_after_dominant_asset_is_removed() -> None:
    with np.errstate(all="raise"):
        result = analytical_portfolio_metrics(
            {"model": _event([-1000.0, 0.0, -1001.0])},
            {"model": "logit"},
            topk_bins=(0, 1, 2, 3),
        )
    np.testing.assert_allclose(result.topk_matrices["model"][0], [0, 0, 1 / (1 + np.exp(-1)), 1], rtol=1e-6, atol=1e-8)


def test_multiple_positive_logits_preserve_mass_across_extreme_gaps() -> None:
    frame = _event([np.finfo(float).max, -np.finfo(float).max, 0.0])
    frame[COL_IS_PUMPED] = [True, True, False]
    with np.errstate(all="raise"):
        result = analytical_portfolio_metrics({"model": frame}, {"model": "logit"}, topk_bins=(0, 1, 2, 3))
    np.testing.assert_array_equal(result.topk_matrices["model"][0], [0, 1, 1, 1])


def test_analytical_calculation_does_not_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("Analytical inclusion probabilities must not sample portfolios")

    monkeypatch.setattr(np.random, "default_rng", fail_if_sampled)
    monkeypatch.setattr(np.random, "choice", fail_if_sampled)
    result = analytical_portfolio_metrics(
        {"model": _event([0.2, 0.3, 0.5])}, {"model": "probability"}, topk_bins=(1, 2, 3)
    )
    expected = [_sequential_hit_probability(np.array([0.2, 0.3, 0.5]), {0}, size) for size in (1, 2, 3)]
    np.testing.assert_allclose(result.topk_matrices["model"][0], expected, rtol=1e-6, atol=1e-8)


def test_single_asset_certain_hit_handles_zero_draws() -> None:
    result = analytical_portfolio_metrics(
        {"model": _event([0.0])},
        {"model": "probability"},
        topk_bins=(0, 1, 2),
        topk_percent_bins=(0, 0.01, 1),
    )
    np.testing.assert_array_equal(result.topk_matrices["model"][0], [0, 1, 1])
    np.testing.assert_array_equal(result.topk_percent_matrices["model"][0], [0, 1, 1])
    assert result.event_aucs.loc["event", "model"] == pytest.approx(0.9875)


def test_paired_comparison_rejects_different_assets_events_or_target_labels() -> None:
    reference = _event([0.6, 0.3, 0.1])
    different_assets = reference.copy()
    different_assets.loc[1, COL_CURRENCY_PAIR] = "OTHER-BTC"
    different_target = _event([0.6, 0.3, 0.1], pumped=1)
    different_event = _event([0.6, 0.3, 0.1], event="another")

    for mismatched in (different_assets, different_target, different_event):
        with pytest.raises(ValueError, match="same event/asset universe and labels"):
            analytical_portfolio_metrics(
                {"reference": reference, "mismatched": mismatched},
                {"reference": "probability", "mismatched": "probability"},
            )


@pytest.mark.parametrize("bins", [[], [[1]], [-1], [0.5], [np.nan], [np.inf], ["invalid"]])
def test_invalid_count_bins_are_rejected(bins: list) -> None:
    with pytest.raises(ValueError):
        analytical_portfolio_metrics({"model": _event([1.0])}, {"model": "probability"}, topk_bins=bins)


@pytest.mark.parametrize("bins", [[], [[0.1]], [-0.1], [1.1], [np.nan], [np.inf], ["invalid"]])
def test_invalid_percentage_bins_are_rejected(bins: list) -> None:
    with pytest.raises(ValueError):
        analytical_portfolio_metrics({"model": _event([1.0])}, {"model": "probability"}, topk_percent_bins=bins)


@pytest.mark.parametrize(
    "parameters",
    [{"topk_bins": (1, 1, 2)}, {"topk_percent_bins": (0.01, 0.01, 0.2)}],
)
def test_duplicate_bins_are_rejected_before_creating_ambiguous_curve_rows(parameters: dict) -> None:
    with pytest.raises(ValueError, match="unique values"):
        analytical_portfolio_metrics({"model": _event([0.6, 0.4])}, {"model": "probability"}, **parameters)


@pytest.mark.parametrize("scores", [[np.inf, 0], [np.nan, 0], [-0.1, 0.5], [1.1, 0.5], [1j, 0], ["invalid", 0]])
def test_invalid_classifier_scores_are_rejected(scores: list) -> None:
    with pytest.raises(ValueError):
        analytical_portfolio_metrics({"model": _event(scores)}, {"model": "probability"})


@pytest.mark.parametrize("kind", ["probability", "logit"])
def test_missing_positive_labels_are_rejected(kind: str) -> None:
    frame = _event([0.5, 0.5], pumped=-1)
    with pytest.raises(ValueError, match="at least one positive"):
        analytical_portfolio_metrics({"model": frame}, {"model": kind})


def test_duplicate_keys_and_nonbinary_labels_are_rejected() -> None:
    frame = _event([0.6, 0.4])
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate event/asset"):
        analytical_portfolio_metrics({"model": duplicated}, {"model": "probability"})
    frame[COL_IS_PUMPED] = [2, 0]
    with pytest.raises(ValueError, match="binary labels"):
        analytical_portfolio_metrics({"model": frame}, {"model": "probability"})


def test_models_require_matching_supported_score_kinds() -> None:
    with pytest.raises(ValueError, match="At least one model"):
        analytical_portfolio_metrics({}, {})
    with pytest.raises(ValueError, match="specify exactly"):
        analytical_portfolio_metrics({"model": _event([1.0])}, {"other": "logit"})
    with pytest.raises(ValueError, match="Unsupported score kind"):
        analytical_portfolio_metrics({"model": _event([1.0])}, {"model": "score"})
