"""Paired bootstrap distributions of ranking and classification metrics on test pumps."""

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import COL_IS_PUMPED, COL_PROBAS_PRED, COL_PUMP_HASH
from pumps_and_dumps.backtest.datasets.sample import DatasetType
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.robustness.significance import (
    _bootstrap_indices,
    _comparison_bins,
    _comparison_event_aucs,
    _holm_adjust_p_values,
    _validate_alpha,
)
from pumps_and_dumps.backtest.robustness.validation_bootstrap import _score_experiments

METRIC_NAMES = ("topk_percent_auc", "f1", "precision", "recall")


@dataclass(frozen=True)
class TestMetricBootstrapResults:
    """Auditable draws, per-pump contributions and paired test metric distributions."""

    scored_datasets: dict[str, pd.DataFrame]
    bootstrap_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    pairwise: pd.DataFrame
    event_metrics: pd.DataFrame
    sampled_events: pd.DataFrame


def evaluate_test_metric_bootstrap(
    experiments: Sequence[Experiment],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
    classification_top_k: int = 1,
) -> TestMetricBootstrapResults:
    """Resample whole TEST pumps and evaluate every fitted model on the same draws.

    Each replicate draws N pumps with replacement from the N test pumps. A
    repeated pump contributes its entire candidate cross-section again. This
    preserves the dependence between candidates within a pump and pairs all
    model comparisons. Fitted models score the original test panel once:
    reusing these fixed predictions is equivalent to running inference again
    on the same resampled cross-sections. Training, tuning and comparison
    partitions are never requested, and the bootstrap does not refit models.

    Top@k%AUC is the normalized area under the event-averaged top-k-percent hit
    curve on [0, 0.20], in steps of 0.005, using the established ceil(N_assets*k)
    convention. F1, precision and recall mark the highest-scored
    ``classification_top_k`` candidates in each pump positive (default one),
    with stable input-order ties and K clipped to each pump's candidate count.
    They are computed from TP/FP/FN pooled over all drawn candidate rows,
    including repeated pumps; they are not means of per-pump ratios.
    Undefined ratios are zero. The classification rule must be chosen before
    inspecting test results; it is not tuned inside this function.

    Percentile confidence intervals are pointwise. Pairwise tests use the
    centered bootstrap distribution of each metric difference, with an add-one
    Monte Carlo correction. These are approximate conditional bootstrap tests,
    not a measure of training uncertainty. Holm correction covers one family
    containing every model pair across all four metrics. Original pump IDs
    and duplicate draw slots are retained in ``sampled_events`` for inspection.
    """
    _validate_alpha(alpha)
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if (
        isinstance(classification_top_k, (bool, np.bool_))
        or not isinstance(classification_top_k, (int, np.integer))
        or classification_top_k < 1
    ):
        raise ValueError("classification_top_k must be a positive integer")
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required for bootstrap model comparison")

    scored_datasets = _score_experiments(experiments, DatasetType.TEST)
    names = list(scored_datasets)
    event_ids = sorted(next(iter(scored_datasets.values()))[COL_PUMP_HASH].unique(), key=str)
    bins = _comparison_bins(None, max_k_percent=0.2, step=0.005)
    # This also validates finite scores, binary labels, matching event counts,
    # at least two pumps and at least one true label in every cross-section.
    event_aucs = _comparison_event_aucs(scored_datasets, bins)
    counts = _event_confusion_counts(scored_datasets, event_ids, classification_top_k)
    n_events, n_models = event_aucs.shape
    sampled_indices = _bootstrap_indices(n_events, n_bootstrap, random_state)
    points = _metric_values(event_aucs.mean(axis=0), counts.sum(axis=0))
    bootstrap_values = np.empty((n_bootstrap, n_models, len(METRIC_NAMES)))
    # Bound temporary arrays, including the TP/FP/FN dimension, while keeping
    # a single reproducible draw matrix shared by every model and metric.
    chunk_size = max(1, min(n_bootstrap, 1_000_000 // (n_events * n_models * 3)))
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        draws = sampled_indices[start:stop]
        bootstrap_values[start:stop] = _metric_values(event_aucs[draws].mean(axis=1), counts[draws].sum(axis=1))

    bootstrap_metrics = pd.DataFrame(bootstrap_values.reshape(-1, len(METRIC_NAMES)), columns=METRIC_NAMES)
    bootstrap_metrics.insert(0, "model", np.tile(names, n_bootstrap))
    bootstrap_metrics.insert(0, "replicate", np.repeat(np.arange(n_bootstrap), n_models))
    event_metrics = pd.DataFrame(
        {
            COL_PUMP_HASH: np.repeat(np.asarray(event_ids, dtype=object), n_models),
            "model": np.tile(names, n_events),
            "topk_percent_auc": event_aucs.ravel(),
            "tp": counts[:, :, 0].ravel(),
            "fp": counts[:, :, 1].ravel(),
            "fn": counts[:, :, 2].ravel(),
        }
    )
    sampled_events = pd.DataFrame(
        np.asarray(event_ids, dtype=object)[sampled_indices],
        index=pd.RangeIndex(n_bootstrap, name="replicate"),
        columns=[f"draw_{draw}" for draw in range(n_events)],
    )
    metric_summary = _metric_summary(names, points, bootstrap_values, alpha)
    pairwise = _paired_metric_comparisons(names, points, bootstrap_values, alpha)
    parameters = {
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
        "n_cross_sections": n_events,
        "dataset_type": DatasetType.TEST.name,
        "random_state": random_state,
        "classification_top_k": int(classification_top_k),
    }
    for table in (metric_summary, pairwise):
        for column, value in parameters.items():
            table[column] = value
    for table in (bootstrap_metrics, metric_summary, pairwise, event_metrics, sampled_events):
        table.attrs.update(parameters)
        table.attrs.update(auc_max_k_percent=0.2, auc_step=0.005, ci_method="pointwise percentile")
    return TestMetricBootstrapResults(
        scored_datasets=scored_datasets,
        bootstrap_metrics=bootstrap_metrics,
        metric_summary=metric_summary,
        pairwise=pairwise,
        event_metrics=event_metrics,
        sampled_events=sampled_events,
    )


def _event_confusion_counts(scored_datasets, event_ids, top_k: int) -> np.ndarray:
    """Keep one TP/FP/FN vector for each event/model, in the AUC event order."""
    event_positions = {str(event): position for position, event in enumerate(event_ids)}
    counts = np.empty((len(event_ids), len(scored_datasets), 3), dtype=np.int64)
    for model_index, frame in enumerate(scored_datasets.values()):
        # The common AUC validator accepts numeric strings; normalize before
        # sorting so classification and AUC apply precisely the same order.
        frame = frame.assign(**{COL_PROBAS_PRED: frame[COL_PROBAS_PRED].to_numpy(dtype=float)})
        for event, cross_section in frame.groupby(COL_PUMP_HASH, sort=False, observed=True):
            ranked_labels = cross_section.sort_values(COL_PROBAS_PRED, ascending=False, kind="stable")[
                COL_IS_PUMPED
            ].to_numpy(dtype=bool)
            selected = min(top_k, len(ranked_labels))
            true_positives = int(ranked_labels[:selected].sum())
            counts[event_positions[str(event)], model_index] = (
                true_positives,
                selected - true_positives,
                int(ranked_labels.sum()) - true_positives,
            )
    return counts


def _metric_values(auc: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Derive micro binary classification metrics from pooled confusion counts."""
    tp, fp, fn = np.moveaxis(counts, -1, 0)

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)

    return np.stack((auc, ratio(2 * tp, 2 * tp + fp + fn), ratio(tp, tp + fp), ratio(tp, tp + fn)), axis=-1)


def _metric_summary(names, points, samples, alpha: float) -> pd.DataFrame:
    rows = []
    for metric_index, metric in enumerate(METRIC_NAMES):
        for model_index, model in enumerate(names):
            distribution = samples[:, model_index, metric_index]
            lower, upper = np.quantile(distribution, [alpha / 2, 1 - alpha / 2])
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "point_estimate": float(points[model_index, metric_index]),
                    "mean_bootstrap": float(distribution.mean()),
                    "std_bootstrap": float(distribution.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def _paired_metric_comparisons(names, points, samples, alpha: float) -> pd.DataFrame:
    """Test paired differences and correct the complete model-pair/metric family."""
    rows = []
    tolerance = 32 * np.finfo(float).eps
    for metric_index, metric in enumerate(METRIC_NAMES):
        for a, b in combinations(range(len(names)), 2):
            observed = float(points[a, metric_index] - points[b, metric_index])
            differences = samples[:, a, metric_index] - samples[:, b, metric_index]
            centered = differences - observed
            p_value = (1 + np.count_nonzero(np.abs(centered) >= abs(observed) - tolerance)) / (len(samples) + 1)
            lower, upper = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
            rows.append(
                {
                    "metric": metric,
                    "model_a": names[a],
                    "model_b": names[b],
                    "observed_diff": observed,
                    "mean_diff_bootstrap": float(differences.mean()),
                    "std_diff_bootstrap": float(differences.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "p_value": float(p_value),
                    "alternative": "two-sided",
                }
            )
    pairwise = pd.DataFrame(rows)
    pairwise["p_value_holm"] = _holm_adjust_p_values(pairwise["p_value"].to_numpy())
    pairwise["reject_h0"] = pairwise["p_value_holm"] < alpha
    pairwise["better_model"] = [
        (row.model_a if row.observed_diff > 0 else row.model_b) if row.reject_h0 else None
        for row in pairwise.itertuples(index=False)
    ]
    pairwise.attrs.update(
        p_value_adjustment="Holm",
        multiplicity_family="all model pairs × all four metrics",
        n_hypotheses=len(pairwise),
        test_method="two-sided centered paired bootstrap with add-one Monte Carlo correction",
    )
    return pairwise
