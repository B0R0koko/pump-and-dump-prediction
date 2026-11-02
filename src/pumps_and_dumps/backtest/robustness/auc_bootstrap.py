"""Auditable paired test bootstrap for fixed models, including historical notebooks."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.sample import DatasetType
from pumps_and_dumps.backtest.robustness.significance import (
    _auc_summary,
    _comparison_bins,
    _comparison_event_aucs,
    _paired_auc_comparisons,
    _validate_alpha,
)
from pumps_and_dumps.backtest.robustness.validation_bootstrap import _score_experiments


@dataclass(frozen=True)
class AucBootstrapResults:
    scored_datasets: dict[str, pd.DataFrame]
    event_aucs: pd.DataFrame
    bootstrap_aucs: pd.DataFrame
    sampled_indices: np.ndarray
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    bins: np.ndarray
    random_state: int
    alpha: float


def evaluate_test_auc_bootstrap(
    experiments: Sequence,
    *,
    dataset_type: Enum = DatasetType.TEST,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> AucBootstrapResults:
    """Score TEST once per model and resample whole events with replacement.

    ``dataset_type`` can be the historical notebook's TEST enum. No other
    partition is read, and no model is fitted or selected here. Every replicate
    draws N events from the N test events, identically for all models. Repeated
    events retain their multiplicity. The mean per-event normalised Top@k%AUC
    equals integrating the resampled Top@k% curve (0--20%, step 0.5%).

    Percentile intervals are pointwise. Two-sided, null-centred bootstrap
    p-values receive Holm correction over all model pairs. These conditional
    intervals assume independent events and exclude training uncertainty.
    """
    if not isinstance(dataset_type, Enum) or dataset_type.name != "TEST":
        raise ValueError("Only the TEST partition may be bootstrapped here")
    _validate_alpha(alpha)
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required")

    scored = _score_experiments(experiments, dataset_type)
    names = list(scored)
    event_order = sorted(map(str, next(iter(scored.values()))["pump_hash"].unique()))
    bins = _comparison_bins(None, max_k_percent=0.2, step=0.005)
    event_values = _comparison_event_aucs(scored, bins)
    indices = np.random.default_rng(random_state).integers(0, len(event_order), size=(n_bootstrap, len(event_order)))
    # Integrating once per event is exact by linearity; chunk the gather array.
    draws = np.empty((n_bootstrap, len(names)))
    chunk = max(1, 1_000_000 // (len(event_order) * len(names)))
    for start in range(0, n_bootstrap, chunk):
        draws[start : start + chunk] = event_values[indices[start : start + chunk]].mean(axis=1)

    summary = _auc_summary(names, event_values, draws, alpha).sort_values(
        "point_estimate", ascending=False, kind="stable"
    )
    pairwise = _paired_auc_comparisons(names, event_values, draws, alpha)
    for table in (summary, pairwise):
        table["dataset_type"] = "TEST"
        table["random_state"] = random_state
    return AucBootstrapResults(
        scored_datasets=scored,
        event_aucs=pd.DataFrame(event_values, index=pd.Index(event_order, name="pump_hash"), columns=names),
        bootstrap_aucs=pd.DataFrame(draws, index=pd.RangeIndex(n_bootstrap, name="replicate"), columns=names),
        sampled_indices=indices,
        summary=summary,
        pairwise=pairwise,
        bins=bins,
        random_state=random_state,
        alpha=alpha,
    )
