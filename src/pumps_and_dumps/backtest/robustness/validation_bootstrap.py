"""Paired bootstrap evaluation of fitted experiments on an evaluation split."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.robustness.significance import (
    _auc_summary,
    _curve_ci_from_matrix,
    _paired_auc_comparisons,
    _resample_event_means,
    _validate_alpha,
    score_dataset,
)
from pumps_and_dumps.backtest.robustness.analytical_selection import analytical_portfolio_metrics
from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_PUMP_HASH
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
    event_probabilities: pd.DataFrame


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
    *,
    dataset_type: DatasetType = DatasetType.VALIDATION,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
    topk_bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30),
    topk_percent_bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
) -> ValidationBootstrapResults:
    """Bootstrap inclusion probabilities for weighted draws without replacement.

    The successive weighted draw law is evaluated by deterministic numerical
    integration, then its inclusion curve is integrated over [0, 20%]. Only
    whole events are resampled, in pairs across models, for event uncertainty.
    No portfolios are simulated. Fitted models stay fixed, so intervals
    exclude training uncertainty. Holm correction covers all model pairs.
    """
    if dataset_type not in (DatasetType.VALIDATION, DatasetType.TEST):
        raise ValueError("Bootstrap evaluation requires VALIDATION or TEST")
    _validate_alpha(alpha)
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required for bootstrap model comparison")

    scored_datasets, score_kinds = _score_experiments(experiments, dataset_type)
    metrics = analytical_portfolio_metrics(
        scored_datasets,
        score_kinds,
        topk_bins=topk_bins,
        topk_percent_bins=topk_percent_bins,
    )
    event_aucs = metrics.event_aucs.to_numpy()
    if len(event_aucs) < 2:
        raise ValueError("At least two events are required for bootstrap model comparison")
    bootstrap_aucs = _resample_event_means(event_aucs, n_bootstrap, random_state)
    names = list(metrics.event_aucs.columns)
    auc_summary = _auc_summary(names, event_aucs, bootstrap_aucs, alpha).sort_values(
        "point_estimate", ascending=False, kind="stable"
    )
    pairwise = _paired_auc_comparisons(names, event_aucs, bootstrap_aucs, alpha)
    parameters = {"n_bootstrap": n_bootstrap, "alpha": alpha, "random_state": random_state}
    topk_ci = _analytical_curve_intervals(metrics.topk_matrices, topk_bins, "k", parameters)
    topk_percent_ci = _analytical_curve_intervals(
        metrics.topk_percent_matrices, topk_percent_bins, "k_percent", parameters
    )
    event_probabilities = pd.concat(
        [
            pd.DataFrame(matrix, index=metrics.event_aucs.index, columns=pd.Index(topk_bins, name="k"))
            .reset_index()
            .melt(id_vars=COL_PUMP_HASH, value_name="hit_probability")
            .assign(model=name)
            for name, matrix in metrics.topk_matrices.items()
        ],
        ignore_index=True,
    )[["model", COL_PUMP_HASH, "k", "hit_probability"]]
    event_probabilities["draw_replacement"] = False
    for result, metric in (
        (auc_summary, "analytical_topk_percent_auc"),
        (pairwise, "analytical_topk_percent_auc"),
        (topk_ci, "analytical_topk"),
        (topk_percent_ci, "analytical_topk_percent"),
    ):
        result["metric"] = metric
        result["dataset_type"] = dataset_type.name
        result["random_state"] = random_state
        result["inclusion_method"] = "analytical"
        result["draw_replacement"] = False
        result.attrs["ci_method"] = "pointwise percentile"

    return ValidationBootstrapResults(
        scored_datasets=scored_datasets,
        auc_summary=auc_summary,
        pairwise=pairwise,
        topk_ci=topk_ci,
        topk_percent_ci=topk_percent_ci,
        event_aucs=metrics.event_aucs,
        event_probabilities=event_probabilities,
    )


def _score_experiments(
    experiments: Sequence[Experiment],
    dataset_type: DatasetType,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    model_names = [experiment.get_experiment_name() for experiment in experiments]
    if not model_names:
        raise ValueError("At least one experiment is required for scoring")
    if any(not isinstance(name, str) or not name.strip() for name in model_names):
        raise ValueError("Experiment names must be nonempty strings")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Experiment names must be unique for model comparison")

    scored_datasets, score_kinds = {}, {}
    for model_name, experiment in zip(model_names, experiments):
        model = experiment.get_model()
        kind = getattr(model, "ranking_score_kind", None)
        if kind not in {"probability", "logit"}:
            raise ValueError(f"{model_name} must declare ranking_score_kind as probability or logit")
        dataset = experiment.get_sample().get_dataset(dataset_type)
        scored = score_dataset(model=model, dataset=dataset)
        scored[COL_CURRENCY_PAIR] = dataset.all_data()[COL_CURRENCY_PAIR].to_numpy(copy=True)
        scored_datasets[model_name] = scored.sort_values(
            COL_PUMP_HASH, key=lambda event_ids: event_ids.map(str), kind="stable"
        )
        score_kinds[model_name] = kind
    return scored_datasets, score_kinds


def _analytical_curve_intervals(matrices, bins, index_name, parameters) -> pd.DataFrame:
    return pd.concat(
        {name: _curve_ci_from_matrix(matrix=matrix, bins=bins, **parameters) for name, matrix in matrices.items()},
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
