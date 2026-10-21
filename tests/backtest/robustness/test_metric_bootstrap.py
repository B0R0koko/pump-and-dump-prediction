"""Small, inspectable oracles for paired whole-pump test metric resampling."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_IS_PUMPED, COL_PUMP_HASH
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType, Sample
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.robustness.metric_bootstrap import METRIC_NAMES, evaluate_test_metric_bootstrap


@dataclass
class TestOnlySample:
    __test__ = False
    dataset: Dataset
    requests: list = field(default_factory=list)

    def get_dataset(self, dataset_type):
        self.requests.append(dataset_type)
        assert dataset_type is DatasetType.TEST
        return self.dataset


@dataclass
class FrozenScoreModel:
    score_column: str
    calls: list = field(default_factory=list)

    def rank(self, dataset):
        self.calls.append(dataset)
        return dataset.all_data()[self.score_column].to_numpy(copy=True)


def _experiments():
    rows = []
    # Input event order is intentionally unsorted. Events have 3, 5 and 2
    # candidates and respectively 2, 1 and 2 positives, exposing both event
    # weighting mistakes and averaging-of-ratios mistakes.
    for event, labels, good, weak in (
        ("z", [1, 1, 0], [3, 2, 1], [2, 1, 3]),
        ("a", [0, 0, 1, 0, 0], [4, 3, 5, 2, 1], [5, 4, 1, 3, 2]),
        ("m", [1, 1], [2, 1], [1, 2]),
    ):
        for index, (label, strong_score, weak_score) in enumerate(zip(labels, good, weak)):
            rows.append(
                {
                    COL_PUMP_HASH: event,
                    COL_CURRENCY_PAIR: f"ASSET{index}-BTC",
                    COL_IS_PUMPED: bool(label),
                    "good": strong_score,
                    "weak": weak_score,
                }
            )
    return [
        Experiment(
            name,
            FrozenScoreModel(score),
            TestOnlySample(Dataset(pd.DataFrame(rows), FeatureSet.empty(), DatasetType.TEST)),
        )
        for name, score in (("good", "good"), ("weak", "weak"), ("good_copy", "good"))
    ]


def _naive_metrics(experiment, pump_draws, top_k):
    """Rebuild sampled candidate rows and compute standard sklearn metrics."""
    truth, predicted, hit_curves = [], [], []
    bins = np.linspace(0, 0.2, 41)
    source = experiment.sample.dataset.all_data()
    for event in pump_draws:
        frame = source[source[COL_PUMP_HASH] == event]
        scores = frame[experiment.model.score_column].to_numpy(dtype=float)
        order = np.argsort(-scores, kind="stable")
        labels = frame[COL_IS_PUMPED].to_numpy(dtype=bool)
        decisions = np.zeros(len(frame), dtype=bool)
        decisions[order[:top_k]] = True
        truth.extend(labels)
        predicted.extend(decisions)
        hit_curves.append([bool(labels[order[: int(np.ceil(len(frame) * percent))]].any()) for percent in bins])
    return {
        "topk_percent_auc": np.trapezoid(np.mean(hit_curves, axis=0), x=bins) / 0.2,
        "f1": f1_score(truth, predicted, zero_division=0),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
    }


@pytest.mark.parametrize("top_k", [1, 2, 20])
def test_each_replicate_matches_rebuilt_cross_sections_and_sklearn(top_k):
    """Check every metric against full, explicitly duplicated candidate rows.

    Three pumps have unequal candidate counts and different positive counts.
    For each saved draw we append each selected pump's complete cross-section,
    including repeats, then use sklearn for binary F1/precision/recall and a
    separately rebuilt hit curve for AUC. K=20 also checks clipping to smaller
    cross-sections. This would fail if pumps were deduplicated, candidate rows
    were resampled independently, or per-pump classification ratios averaged.
    """
    experiments = _experiments()
    result = evaluate_test_metric_bootstrap(experiments, n_bootstrap=41, random_state=9, classification_top_k=top_k)
    expected_draws = np.array(["a", "m", "z"])[np.random.default_rng(9).integers(0, 3, size=(41, 3))]
    np.testing.assert_array_equal(result.sampled_events.to_numpy(), expected_draws)
    assert any(len(set(draw)) < 3 for draw in expected_draws)
    distributions = result.bootstrap_metrics.set_index(["replicate", "model"])
    summary = result.metric_summary.set_index(["model", "metric"])
    for experiment in experiments:
        name = experiment.experiment_name
        for replicate, draws in enumerate(expected_draws):
            expected = _naive_metrics(experiment, draws, top_k)
            for metric in METRIC_NAMES:
                assert distributions.loc[(replicate, name), metric] == pytest.approx(expected[metric])
        expected_points = _naive_metrics(experiment, ["a", "m", "z"], top_k)
        for metric in METRIC_NAMES:
            actual = distributions.xs(name, level="model")[metric]
            row = summary.loc[(name, metric)]
            assert row["point_estimate"] == pytest.approx(expected_points[metric])
            assert row["mean_bootstrap"] == pytest.approx(actual.mean())
            assert row["std_bootstrap"] == pytest.approx(actual.std(ddof=1))
            np.testing.assert_allclose(
                row[["ci_lower", "ci_upper"]].to_numpy(dtype=float), actual.quantile([0.025, 0.975])
            )


def test_classification_pools_confusion_counts_instead_of_averaging_pump_ratios():
    """Use hand-calculated TP/FP/FN to make the aggregation rule explicit.

    At K=2 the stronger model has event counts (1,1,0), (2,0,0), (2,0,0),
    so pooled precision is 5/6, recall is 1, and F1 is 10/11. Its average
    per-event F1 is 8/9 and must not be reported. The weaker model's totals
    (3,3,2) give precision 1/2, recall 3/5, and F1 6/11. These values also
    exercise multiple true labels within one pump, beyond the usual dataset.
    """
    result = evaluate_test_metric_bootstrap(_experiments(), n_bootstrap=7, classification_top_k=2)
    points = result.metric_summary.pivot(index="model", columns="metric", values="point_estimate")
    assert points.loc["good", "precision"] == pytest.approx(5 / 6)
    assert points.loc["good", "recall"] == 1
    assert points.loc["good", "f1"] == pytest.approx(10 / 11)
    assert points.loc["good", "f1"] != pytest.approx(8 / 9)
    assert points.loc["weak", "precision"] == pytest.approx(1 / 2)
    assert points.loc["weak", "recall"] == pytest.approx(3 / 5)
    assert points.loc["weak", "f1"] == pytest.approx(6 / 11)
    counts = result.event_metrics.set_index([COL_PUMP_HASH, "model"])
    np.testing.assert_array_equal(counts.loc[("a", "good"), ["tp", "fp", "fn"]], [1, 1, 0])
    np.testing.assert_array_equal(counts.loc[("z", "weak"), ["tp", "fp", "fn"]], [1, 1, 1])


def test_paired_contrasts_use_same_draws_and_one_holm_family_across_metrics():
    """Reconstruct confidence intervals and multiplicity correction independently.

    Three models yield three unordered pairs for each of four metrics. For
    all 12 contrasts we subtract distributions with the same replicate ID,
    center by the original test-set difference, and count null tails with
    the add-one correction. Holm is then computed by an independent sorted
    step-down loop across all 12 p-values, not independently by metric or
    after filtering to a favorite model. Duplicate models must remain ties.
    """
    result = evaluate_test_metric_bootstrap(_experiments(), n_bootstrap=199, alpha=0.1, random_state=2)
    points = result.metric_summary.set_index(["model", "metric"])["point_estimate"]
    p_values = []
    for row in result.pairwise.itertuples(index=False):
        matrix = result.bootstrap_metrics.pivot(index="replicate", columns="model", values=row.metric)
        differences = matrix[row.model_a] - matrix[row.model_b]
        observed = points.loc[(row.model_a, row.metric)] - points.loc[(row.model_b, row.metric)]
        p_value = (1 + (abs(differences - observed) >= abs(observed) - 32 * np.finfo(float).eps).sum()) / 200
        p_values.append(p_value)
        assert row.observed_diff == pytest.approx(observed)
        assert row.mean_diff_bootstrap == pytest.approx(differences.mean())
        assert row.std_diff_bootstrap == pytest.approx(differences.std(ddof=1))
        assert row.p_value == pytest.approx(p_value)
        np.testing.assert_allclose([row.ci_lower, row.ci_upper], differences.quantile([0.05, 0.95]))
    expected_adjusted = np.empty(12)
    previous = 0
    for rank, index in enumerate(np.argsort(p_values)):
        previous = max(previous, (12 - rank) * p_values[index])
        expected_adjusted[index] = min(previous, 1)
    np.testing.assert_allclose(result.pairwise["p_value_holm"], expected_adjusted)
    assert result.pairwise["reject_h0"].tolist() == (expected_adjusted < 0.1).tolist()
    assert result.pairwise.attrs["n_hypotheses"] == 12
    assert result.pairwise.attrs["p_value_adjustment"] == "Holm"
    ties = result.pairwise.query("model_a == 'good' and model_b == 'good_copy'")
    assert (ties[["observed_diff", "ci_lower", "ci_upper"]] == 0).all().all()
    assert (ties["p_value"] == 1).all()
    assert ties["better_model"].isna().all()


def test_models_only_score_test_once_and_inputs_remain_unchanged():
    """Ensure resampling never requests training data or reruns fitted models.

    Each dummy sample raises immediately for a non-TEST request and each
    rank method records its calls. Hundreds of replicates must still cause
    exactly one test request and one inference call per model. Deep copies
    of the original datasets ensure adding scores and bootstrap bookkeeping
    cannot mutate the data later used elsewhere in the notebook.
    """
    experiments = _experiments()
    originals = [experiment.sample.dataset.all_data().copy(deep=True) for experiment in experiments]
    result = evaluate_test_metric_bootstrap(experiments, n_bootstrap=201)
    for experiment, original in zip(experiments, originals):
        assert experiment.sample.requests == [DatasetType.TEST]
        assert experiment.model.calls == [experiment.sample.dataset]
        pd.testing.assert_frame_equal(experiment.sample.dataset.all_data(), original)
    assert (result.metric_summary["dataset_type"] == "TEST").all()
    assert (result.metric_summary["classification_top_k"] == 1).all()
    assert result.sampled_events.index.name == "replicate"
    assert result.sampled_events.columns.tolist() == ["draw_0", "draw_1", "draw_2"]


def test_seed_reproducibility_and_score_gaps_do_not_change_deterministic_rank_metrics():
    """Separate pump sampling randomness from deterministic model predictions.

    Repeating the same seed must reproduce every draw and metric exactly.
    Multiplying scores by 100 preserves ranks, so it cannot change these
    results. Changing the seed changes draws but leaves original test-set
    point estimates unchanged. This protects against accidentally bringing
    back probability-weighted asset sampling or interpreting logits as odds.
    """
    experiments = _experiments()
    first = evaluate_test_metric_bootstrap(experiments, n_bootstrap=31, random_state=5)
    for experiment in experiments:
        experiment.sample.dataset.all_data()[experiment.model.score_column] *= 100
    repeated = evaluate_test_metric_bootstrap(experiments, n_bootstrap=31, random_state=5)
    changed = evaluate_test_metric_bootstrap(experiments, n_bootstrap=31, random_state=17)
    pd.testing.assert_frame_equal(first.sampled_events, repeated.sampled_events)
    pd.testing.assert_frame_equal(first.bootstrap_metrics, repeated.bootstrap_metrics)
    pd.testing.assert_frame_equal(first.metric_summary, repeated.metric_summary)
    assert not first.sampled_events.equals(changed.sampled_events)
    pd.testing.assert_series_equal(first.metric_summary["point_estimate"], changed.metric_summary["point_estimate"])


def test_stable_score_ties_preserve_candidate_order_and_original_event_ids():
    """Make tie handling and saved pump identities directly inspectable.

    Every candidate receives score one, so the first input row is the top
    prediction. Event 'a' starts with a negative label and must have zero
    true positives; sorting candidates by label would falsely improve it.
    Numeric pump IDs are preserved in saved draws rather than silently
    converting them to strings for grouping or serialization.
    """
    experiments = _experiments()
    for experiment in experiments:
        frame = experiment.sample.dataset.all_data()
        frame[experiment.model.score_column] = 1.0
        frame[COL_PUMP_HASH] = frame[COL_PUMP_HASH].map({"a": 10, "m": 20, "z": 30})
    result = evaluate_test_metric_bootstrap(experiments, n_bootstrap=19)
    first = result.event_metrics.query("model == 'good'").set_index(COL_PUMP_HASH)
    assert first.loc[10, "tp"] == 0
    assert first.loc[20, "tp"] == 1
    assert first.loc[30, "tp"] == 1
    assert set(result.sampled_events.to_numpy().ravel()) == {10, 20, 30}


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("n_bootstrap", 1),
        ("n_bootstrap", True),
        ("n_bootstrap", 2.5),
        ("classification_top_k", 0),
        ("classification_top_k", False),
        ("classification_top_k", 1.5),
        ("alpha", 0),
        ("alpha", 1),
    ],
)
def test_invalid_resampling_or_classification_parameters_fail_before_inference(parameter, value):
    """Reject invalid bootstrap and decision-rule settings before model access.

    Sample standard deviations need at least two replicates, top-k requires
    a positive integer, and confidence levels must lie strictly between zero
    and one. Booleans are rejected even though Python treats them as integers.
    No invalid request may consume a test prediction or read a dataset.
    """
    experiments = _experiments()
    with pytest.raises(ValueError):
        evaluate_test_metric_bootstrap(experiments, **{parameter: value})
    assert all(not experiment.model.calls and not experiment.sample.requests for experiment in experiments)


@pytest.mark.parametrize("change", ["asset", "label", "missing_positive", "score", "duplicate"])
def test_invalid_or_misaligned_cross_sections_are_rejected(change):
    """Prevent apparently paired comparisons from using different observations.

    Change one model's candidate identity, ground truth, score validity or
    duplicate an asset row and require failure. A separate case removes all
    true labels in one pump for every model: ranking hit AUC requires at least
    one positive per event and must reject such a pump, never silently drop it
    while classification metrics retain it under a different denominator.
    """
    experiments = _experiments()
    frame = experiments[1].sample.dataset.all_data()
    if change == "asset":
        frame.loc[0, COL_CURRENCY_PAIR] = "OTHER-BTC"
    elif change == "label":
        frame.loc[0, COL_IS_PUMPED] = False
    elif change == "missing_positive":
        for experiment in experiments:
            rows = experiment.sample.dataset.all_data()
            rows.loc[rows[COL_PUMP_HASH] == "a", COL_IS_PUMPED] = False
    elif change == "score":
        frame["weak"] = frame["weak"].astype(float)
        frame.loc[0, "weak"] = np.nan
    else:
        frame.loc[1, COL_CURRENCY_PAIR] = frame.loc[0, COL_CURRENCY_PAIR]
    with pytest.raises(ValueError):
        evaluate_test_metric_bootstrap(experiments, n_bootstrap=3)


def test_missing_test_partition_never_falls_back_to_comparison_or_training():
    """A sample without a held-out TEST partition must stop evaluation.

    Both experiments expose only TRAIN and VALIDATION datasets. The public
    test-only API must raise for TEST instead of using whichever partition
    happens to be available, and it must not invoke model inference first.
    """
    experiments = _experiments()
    for experiment in experiments:
        experiment.sample = Sample(
            {DatasetType.TRAIN: experiment.sample.dataset, DatasetType.VALIDATION: experiment.sample.dataset}
        )
    with pytest.raises(KeyError, match="TEST"):
        evaluate_test_metric_bootstrap(experiments, n_bootstrap=3)
    assert all(not experiment.model.calls for experiment in experiments)
