"""Paired event-bootstrap evaluation of deterministic top-K model rankings."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.robustness.significance import (
    _auc_summary,
    _comparison_bins,
    _comparison_event_aucs,
    _cross_section_indicator_vectors_topk,
    _cross_section_indicator_vectors_topk_percent,
    _curve_ci_from_matrix,
    _matrix_from_vectors,
    _paired_auc_comparisons,
    _resample_event_means,
    _validate_alpha,
    score_dataset,
)
from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_IS_PUMPED, COL_PUMP_HASH
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.datasets.sample import DatasetType


@dataclass(frozen=True)
class ValidationBootstrapResults:
    scored_datasets: dict[str, pd.DataFrame]
    auc_summary: pd.DataFrame
    pairwise: pd.DataFrame
    topk_ci: pd.DataFrame
    topk_percent_ci: pd.DataFrame
    event_aucs: pd.DataFrame


def evaluate_validation_bootstrap(
    experiments: Sequence[Experiment],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
    topk_bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30),
    topk_percent_bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
) -> ValidationBootstrapResults:
    """Evaluate validation with the original validation-only API."""
    return evaluate_bootstrap(
        experiments,
        dataset_type=DatasetType.VALIDATION,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
        topk_bins=topk_bins,
        topk_percent_bins=topk_percent_bins,
    )


def evaluate_bootstrap(
    experiments: Sequence[Experiment],
    dataset_type: DatasetType = DatasetType.VALIDATION,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
    topk_bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30),
    topk_percent_bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
) -> ValidationBootstrapResults:
    """Compare deterministic top-K accuracy using a paired event bootstrap.

    Scores determine an ordered asset list for each event. Top@K is one when
    that list's first K assets contain a true label and zero otherwise; score
    magnitudes never determine draw probabilities or portfolio allocations.
    Top@k%AUC integrates these indicators over the first 20% of candidates.

    Models are scored once on the requested evaluation partition. Whole pump
    events are resampled with the same draws for all models and thresholds.
    Pointwise percentile intervals condition on the fitted models, while Holm
    correction covers the full family of pairwise model comparisons.
    """
    if dataset_type not in (DatasetType.VALIDATION, DatasetType.TEST):
        raise ValueError("Bootstrap evaluation requires VALIDATION or TEST")
    _validate_alpha(alpha)
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required for bootstrap model comparison")

    scored_datasets = _score_experiments(experiments, dataset_type)
    event_order = sorted(map(str, next(iter(scored_datasets.values()))[COL_PUMP_HASH].unique()))
    if len(event_order) < 2:
        raise ValueError("At least two events are required for bootstrap model comparison")
    names = list(scored_datasets)
    auc_bins = _comparison_bins(None, max_k_percent=0.2, step=0.005)
    event_aucs = _comparison_event_aucs(scored_datasets, auc_bins)
    bootstrap_aucs = _resample_event_means(event_aucs, n_bootstrap, random_state)
    auc_summary = _auc_summary(names, event_aucs, bootstrap_aucs, alpha).sort_values(
        "point_estimate", ascending=False, kind="stable"
    )
    pairwise = _paired_auc_comparisons(names, event_aucs, bootstrap_aucs, alpha)
    parameters = {"n_bootstrap": n_bootstrap, "alpha": alpha, "random_state": random_state}
    topk_ci = _ranking_curve_intervals(
        scored_datasets, topk_bins, "k", event_order, _cross_section_indicator_vectors_topk, parameters
    )
    topk_percent_ci = _ranking_curve_intervals(
        scored_datasets,
        topk_percent_bins,
        "k_percent",
        event_order,
        _cross_section_indicator_vectors_topk_percent,
        parameters,
    )
    for result, metric in (
        (auc_summary, "topk_percent_auc"),
        (pairwise, "topk_percent_auc"),
        (topk_ci, "topk"),
        (topk_percent_ci, "topk_percent"),
    ):
        result["metric"] = metric
        result["dataset_type"] = dataset_type.name
        result["random_state"] = random_state
        result.attrs["ci_method"] = "pointwise percentile"

    return ValidationBootstrapResults(
        scored_datasets=scored_datasets,
        auc_summary=auc_summary,
        pairwise=pairwise,
        topk_ci=topk_ci,
        topk_percent_ci=topk_percent_ci,
        event_aucs=pd.DataFrame(event_aucs, index=pd.Index(event_order, name=COL_PUMP_HASH), columns=names),
    )


def _score_experiments(
    experiments: Sequence[Experiment],
    dataset_type: DatasetType,
) -> dict[str, pd.DataFrame]:
    """Score matching event/asset/label panels without reading another split."""
    model_names = [experiment.get_experiment_name() for experiment in experiments]
    if not model_names:
        raise ValueError("At least one experiment is required for scoring")
    if any(not isinstance(name, str) or not name.strip() for name in model_names):
        raise ValueError("Experiment names must be nonempty strings")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Experiment names must be unique for model comparison")

    scored_datasets = {}
    reference_labels = None
    for model_name, experiment in zip(model_names, experiments):
        dataset = experiment.get_sample().get_dataset(dataset_type)
        scored = score_dataset(model=experiment.get_model(), dataset=dataset)
        scored[COL_CURRENCY_PAIR] = dataset.all_data()[COL_CURRENCY_PAIR].to_numpy(copy=True)
        keys = [COL_PUMP_HASH, COL_CURRENCY_PAIR]
        if scored[keys].isna().any().any() or scored.duplicated(keys).any():
            raise ValueError(f"{model_name} requires unique, nonmissing event and asset identifiers")
        labels = scored.set_index(keys)[COL_IS_PUMPED]
        if reference_labels is None:
            reference_labels = labels
        elif set(labels.index) != set(reference_labels.index) or not labels.reindex(reference_labels.index).equals(
            reference_labels
        ):
            raise ValueError("All models must use the same event assets and true labels")
        scored_datasets[model_name] = scored.sort_values(
            COL_PUMP_HASH, key=lambda event_ids: event_ids.map(str), kind="stable"
        )
    return scored_datasets


def _ranking_curve_intervals(scored_datasets, bins, index_name, event_order, indicator, parameters) -> pd.DataFrame:
    return pd.concat(
        {
            name: _curve_ci_from_matrix(
                matrix=_matrix_from_vectors(indicator(frame, bins), event_order), bins=bins, **parameters
            )
            for name, frame in scored_datasets.items()
        },
        names=["model", index_name],
    )


def comparisons_for_model(pairwise: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Orient existing pairwise results as chosen model minus comparator.

    P-values and decisions retain Holm correction over the original full
    family; filtering to one model never recomputes that correction. Reversing
    a contrast negates its means and swaps/negates its confidence bounds.
    The input table is not modified.
    """
    model_names = pd.unique(pd.concat([pairwise["model_a"], pairwise["model_b"]], ignore_index=True)).tolist()
    if model_name not in model_names:
        raise ValueError(f"Unknown model {model_name!r}; choose from {model_names}")

    comparisons = pairwise.loc[(pairwise["model_a"] == model_name) | (pairwise["model_b"] == model_name)].copy()
    reverse = comparisons["model_b"] == model_name
    comparisons["comparator"] = comparisons["model_b"].where(~reverse, comparisons["model_a"])
    for column in ("observed_diff", "mean_diff_bootstrap"):
        comparisons.loc[reverse, column] *= -1
    original_lower = comparisons["ci_lower"].copy()
    comparisons.loc[reverse, "ci_lower"] = -comparisons.loc[reverse, "ci_upper"]
    comparisons.loc[reverse, "ci_upper"] = -original_lower.loc[reverse]

    comparisons["conclusion"] = "Inconclusive"
    significant = comparisons["reject_h0"]
    comparisons.loc[significant & (comparisons["observed_diff"] > 0), "conclusion"] = "Chosen model better"
    comparisons.loc[significant & (comparisons["observed_diff"] < 0), "conclusion"] = "Comparator better"
    comparisons = comparisons.set_index("comparator").sort_values("observed_diff", ascending=False, kind="stable")[
        [
            "observed_diff",
            "mean_diff_bootstrap",
            "std_diff_bootstrap",
            "ci_lower",
            "ci_upper",
            "p_value",
            "p_value_holm",
            "conclusion",
        ]
    ]
    comparisons.insert(0, "chosen_model", model_name)
    return comparisons
