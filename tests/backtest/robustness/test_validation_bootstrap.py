from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PROBAS_PRED,
    COL_PUMP_HASH,
)
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.robustness.validation_bootstrap import (
    comparisons_for_model,
    evaluate_bootstrap,
    evaluate_validation_bootstrap,
)


@dataclass
class RequestedSplitSample:
    dataset: Dataset
    requests: list[DatasetType] = field(default_factory=list)

    def get_dataset(self, ds_type: DatasetType) -> Dataset:
        self.requests.append(ds_type)
        assert ds_type is self.dataset.ds_type
        return self.dataset


@dataclass
class CountingModel:
    ranking_score_kind = "logit"
    calls: list[Dataset] = field(default_factory=list)

    def rank(self, dataset: Dataset) -> np.ndarray:
        self.calls.append(dataset)
        return dataset.all_data()["score"].to_numpy()


def _experiment(
    name: str,
    ranks: tuple[int, ...],
    dataset_type: DatasetType = DatasetType.VALIDATION,
) -> Experiment:
    # Event order deliberately differs from the sorted order used for paired
    # AUC resampling. Labels and assets match across models; only scores vary.
    rows = []
    for event, rank in zip(("z", "b", "a", "x"), ranks):
        scores = np.arange(4, 0, -1, dtype=float)
        scores[0], scores[rank] = scores[rank], scores[0]
        rows.extend(
            {
                COL_PUMP_HASH: event,
                COL_CURRENCY_PAIR: f"ASSET{asset}-BTC",
                COL_IS_PUMPED: asset == 0,
                "score": score,
            }
            for asset, score in enumerate(scores)
        )
    dataset = Dataset(pd.DataFrame(rows), FeatureSet.empty(), dataset_type)
    return Experiment(name, CountingModel(), RequestedSplitSample(dataset))


@pytest.mark.parametrize(
    ("evaluate", "dataset_type", "split_kwargs"),
    [
        (evaluate_validation_bootstrap, DatasetType.VALIDATION, {}),
        (evaluate_bootstrap, DatasetType.VALIDATION, {}),
        (evaluate_bootstrap, DatasetType.TEST, {"dataset_type": DatasetType.TEST}),
    ],
)
def test_evaluation_scores_requested_split_once_and_returns_all_models_pairs_and_thresholds(
    evaluate, dataset_type: DatasetType, split_kwargs: dict
) -> None:
    experiments = [
        _experiment("weak", (3, 3, 3, 3), dataset_type),
        _experiment("strong", (0, 0, 0, 0), dataset_type),
        _experiment("mixed", (0, 3, 3, 0), dataset_type),
    ]
    results = evaluate(experiments, n_bootstrap=199, random_state=7, **split_kwargs)

    assert list(results.auc_summary.index) == ["strong", "mixed", "weak"]
    assert len(results.pairwise) == 3
    assert len(results.topk_ci) == 3 * 7
    assert len(results.topk_percent_ci) == 3 * 5
    assert results.topk_ci.index.names == ["model", "k"]
    assert results.topk_percent_ci.index.names == ["model", "k_percent"]
    assert len(results.event_probabilities) == 3 * 4 * 7
    assert not results.event_probabilities.duplicated(["model", COL_PUMP_HASH, "k"]).any()
    assert results.event_probabilities["hit_probability"].between(0, 1).all()
    means = results.event_probabilities.groupby(["model", "k"])["hit_probability"].mean()
    pd.testing.assert_series_equal(means, results.topk_ci["point_estimate"].sort_index(), check_names=False)
    for table in (
        results.auc_summary,
        results.pairwise,
        results.topk_ci,
        results.topk_percent_ci,
    ):
        assert (table["dataset_type"] == dataset_type.name).all()
        assert (table["random_state"] == 7).all()
        assert (table["n_bootstrap"] == 199).all()
        assert "n_portfolios" not in table
        assert "portfolio_random_state" not in table
        assert table["metric"].str.startswith("analytical_").all()
        assert table["inclusion_method"].eq("analytical").all()
        assert table["draw_replacement"].eq(False).all()
        assert table.attrs["ci_method"] == "pointwise percentile"
    for experiment in experiments:
        assert experiment.sample.requests == [dataset_type]
        assert experiment.model.calls == [experiment.sample.dataset]
        assert COL_PROBAS_PRED not in experiment.sample.dataset.all_data()
        scored = results.scored_datasets[experiment.experiment_name]
        assert scored[COL_PUMP_HASH].unique().tolist() == ["a", "b", "x", "z"]
        assert COL_CURRENCY_PAIR in scored


def test_training_split_is_rejected_before_scoring() -> None:
    experiments = [
        _experiment("weak", (3, 3, 3, 3)),
        _experiment("strong", (0, 0, 0, 0)),
    ]
    with pytest.raises(ValueError, match="VALIDATION or TEST"):
        evaluate_bootstrap(experiments, dataset_type=DatasetType.TRAIN)
    assert all(not experiment.model.calls for experiment in experiments)
    assert all(not experiment.sample.requests for experiment in experiments)


def test_bootstrap_requires_at_least_two_events() -> None:
    with pytest.raises(ValueError, match="At least two events"):
        evaluate_validation_bootstrap([_experiment("first", (0,)), _experiment("second", (1,))])


def test_event_probabilities_and_model_scores_do_not_depend_on_bootstrap_seed() -> None:
    experiments = [_experiment("mixed", (0, 3, 1, 0)), _experiment("weak", (3, 3, 3, 3))]
    first = evaluate_validation_bootstrap(experiments, n_bootstrap=31, random_state=1)
    second = evaluate_validation_bootstrap(experiments, n_bootstrap=37, random_state=99)
    pd.testing.assert_frame_equal(first.event_aucs, second.event_aucs)
    pd.testing.assert_frame_equal(first.event_probabilities, second.event_probabilities)
    pd.testing.assert_series_equal(first.auc_summary["point_estimate"], second.auc_summary["point_estimate"])
    for row in first.event_probabilities.itertuples():
        scored = first.scored_datasets[row.model]
        event = scored.loc[scored[COL_PUMP_HASH] == row.pump_hash]
        logits = event[COL_PROBAS_PRED].to_numpy()
        weights = np.exp(logits - logits.max())
        labels = event[COL_IS_PUMPED].to_numpy(dtype=bool)
        expected = 0.0
        for sequence in permutations(range(len(weights)), min(row.k, len(weights))):
            if not labels[list(sequence)].any():
                continue
            remaining = set(range(len(weights)))
            probability = 1.0
            for asset in sequence:
                probability *= weights[asset] / weights[list(remaining)].sum()
                remaining.remove(asset)
            expected += probability
        assert row.hit_probability == pytest.approx(expected, abs=1e-8)
        assert row.draw_replacement is False


def test_threshold_estimates_share_the_auc_event_order_and_support_custom_bins() -> None:
    experiments = [
        _experiment("mixed", (0, 3, 1, 0)),
        _experiment("weak", (3, 3, 3, 3)),
    ]
    parameters = {"n_bootstrap": 31, "random_state": 11, "alpha": 0.1}
    results = evaluate_validation_bootstrap(
        experiments,
        topk_bins=(1, 3),
        topk_percent_bins=(0.25, 0.5),
        **parameters,
    )
    assert results.event_aucs.index.tolist() == ["a", "b", "x", "z"]
    per_event = results.event_aucs.to_numpy()
    draws = np.random.default_rng(11).integers(0, 4, size=(31, 4))
    bootstrap_aucs = per_event[draws].mean(axis=1)
    for column, name in enumerate(results.event_aucs.columns):
        summary = results.auc_summary.loc[name]
        expected = bootstrap_aucs[:, column]
        assert summary["point_estimate"] == pytest.approx(per_event[:, column].mean())
        assert summary["mean_bootstrap"] == pytest.approx(expected.mean())
        assert summary["std_bootstrap"] == pytest.approx(expected.std(ddof=1))
        np.testing.assert_allclose(
            summary[["ci_lower", "ci_upper"]].to_numpy(dtype=float), np.quantile(expected, [0.05, 0.95])
        )
        assert summary["n_cross_sections"] == 4

    # Compare pairs from the same event draws, rather than bootstrapping models
    # independently; the analytical probabilities have no simulation error.
    pair = results.pairwise.iloc[0]
    difference = bootstrap_aucs[:, 0] - bootstrap_aucs[:, 1]
    assert pair["mean_diff_bootstrap"] == pytest.approx(difference.mean())
    np.testing.assert_allclose(
        pair[["ci_lower", "ci_upper"]].to_numpy(dtype=float), np.quantile(difference, [0.05, 0.95])
    )

    # With four assets, every positive AUC threshold is K=1; the initial
    # zero-point trapezoid scales the analytical Top@1 curve by .9875.
    assert results.auc_summary.loc["mixed", "mean_bootstrap"] == pytest.approx(
        results.topk_ci.loc[("mixed", 1), "mean_bootstrap"] * 0.9875
    )
    assert results.auc_summary.loc["mixed", "std_bootstrap"] == pytest.approx(
        results.topk_ci.loc[("mixed", 1), "std_bootstrap"] * 0.9875
    )
    for column in ("point_estimate", "mean_bootstrap", "std_bootstrap", "ci_lower", "ci_upper"):
        assert results.topk_ci.loc[("mixed", 1), column] == pytest.approx(
            results.topk_percent_ci.loc[("mixed", 0.25), column]
        )


@pytest.mark.parametrize(
    "names",
    [("duplicate", "duplicate"), ("", "valid"), ("  ", "valid"), (None, "valid")],
)
def test_invalid_names_are_rejected_before_scoring(names: tuple) -> None:
    experiments = [_experiment(name, (0, 0, 0, 0)) for name in names]
    with pytest.raises(ValueError, match="names"):
        evaluate_validation_bootstrap(experiments)
    assert all(not experiment.model.calls for experiment in experiments)
    assert all(not experiment.sample.requests for experiment in experiments)


@pytest.mark.parametrize("n_models", [0, 1])
def test_at_least_two_experiments_are_required(n_models: int) -> None:
    experiments = [_experiment("only", (0, 0, 0, 0)) for _ in range(n_models)]
    with pytest.raises(ValueError, match="At least two"):
        evaluate_validation_bootstrap(experiments)
    assert all(not experiment.model.calls for experiment in experiments)


def test_chosen_comparisons_reverse_means_and_intervals_without_changing_full_family_correction() -> None:
    pairwise = pd.DataFrame(
        {
            "model_a": ["a", "a", "b"],
            "model_b": ["b", "c", "c"],
            "observed_diff": [-0.2, 0.02, -0.1],
            "mean_diff_bootstrap": [-0.21, 0.03, -0.11],
            "std_diff_bootstrap": [0.04, 0.05, 0.06],
            "ci_lower": [-0.4, -0.1, -0.3],
            "ci_upper": [-0.1, 0.2, -0.05],
            "p_value": [0.003, 0.3, 0.01],
            "p_value_holm": [0.009, 0.3, 0.02],
            "reject_h0": [True, False, True],
        }
    )
    pairwise.attrs["p_value_adjustment"] = "Holm"
    original = pairwise.copy(deep=True)

    chosen = comparisons_for_model(pairwise, "b")

    assert chosen.index.tolist() == ["a", "c"]
    assert (chosen["chosen_model"] == "b").all()
    assert chosen.loc["a", "observed_diff"] == pytest.approx(0.2)
    assert chosen.loc["a", "mean_diff_bootstrap"] == pytest.approx(0.21)
    assert chosen.loc["a", "ci_lower"] == pytest.approx(0.1)
    assert chosen.loc["a", "ci_upper"] == pytest.approx(0.4)
    assert chosen.loc["a", "std_diff_bootstrap"] == pytest.approx(0.04)
    assert chosen.loc["c", "observed_diff"] == pytest.approx(-0.1)
    assert chosen.loc["c", "ci_lower"] == pytest.approx(-0.3)
    assert chosen.loc["c", "ci_upper"] == pytest.approx(-0.05)
    assert chosen["p_value"].tolist() == [0.003, 0.01]
    assert chosen["p_value_holm"].tolist() == [0.009, 0.02]
    assert chosen["conclusion"].tolist() == ["Chosen model better", "Comparator better"]
    assert chosen.attrs["p_value_adjustment"] == "Holm"
    assert comparisons_for_model(pairwise, "a").loc["c", "conclusion"] == "Inconclusive"
    pd.testing.assert_frame_equal(pairwise, original)
    with pytest.raises(ValueError, match="Unknown model"):
        comparisons_for_model(pairwise, "missing")


def test_chosen_comparisons_keep_holm_decisions_when_raw_p_values_are_significant() -> None:
    experiments = [
        _experiment("weak", (3, 3, 3, 3)),
        _experiment("strong", (0, 0, 0, 0)),
        _experiment("mixed", (0, 0, 0, 3)),
    ]
    results = evaluate_validation_bootstrap(experiments, n_bootstrap=24)
    chosen = comparisons_for_model(results.pairwise, "strong")

    assert chosen.loc["weak", "p_value"] == pytest.approx(0.04)
    assert chosen.loc["weak", "p_value_holm"] == pytest.approx(0.12)
    assert chosen.loc["weak", "conclusion"] == "Inconclusive"
