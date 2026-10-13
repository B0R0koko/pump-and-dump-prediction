from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Sequence, Literal, Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from pumps_and_dumps.backtest.pipelines.base_model import ImplementsRank
from pumps_and_dumps.backtest.datasets.columns import (
    COL_IS_PUMPED,
    COL_PROBAS_PRED,
    COL_PUMP_HASH,
)
from pumps_and_dumps.backtest.datasets.sample import Dataset


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    point_estimate: float
    mean_bootstrap: float
    std_bootstrap: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    alpha: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapTestResult:
    metric: str
    observed_diff: float
    mean_diff_bootstrap: float
    ci_lower: float
    ci_upper: float
    p_value: float
    alternative: str
    n_bootstrap: int
    alpha: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


def score_dataset(model: ImplementsRank, dataset: Dataset) -> pd.DataFrame:
    """
    Evaluate model scores once and keep only the columns required for grouped bootstrap.
    """
    scores: np.ndarray = model.rank(dataset=dataset)
    df: pd.DataFrame = dataset.all_data()[[COL_PUMP_HASH, COL_IS_PUMPED]].copy(
        deep=True
    )
    if len(scores) != len(df):
        raise ValueError(
            f"Model returned {len(scores)} scores, but dataset contains {len(df)} rows"
        )
    df[COL_PROBAS_PRED] = scores
    return df


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _validate_n_bootstrap(n_bootstrap: int) -> None:
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")


def _cross_section_indicator_vectors_topk(
    scored_df: pd.DataFrame, bins: Sequence[int]
) -> dict[str, np.ndarray]:
    if len(bins) == 0:
        raise ValueError("bins must not be empty")
    if any(k < 1 for k in bins):
        raise ValueError(f"all bins must be >= 1, got {bins}")

    vectors: dict[str, np.ndarray] = {}
    for pump_hash, df_cs in scored_df.groupby(COL_PUMP_HASH, sort=False):
        df_cs = df_cs.sort_values(by=COL_PROBAS_PRED, ascending=False, kind="stable")
        is_pumped: np.ndarray = df_cs[COL_IS_PUMPED].to_numpy(dtype=bool)
        if is_pumped.size == 0 or not is_pumped.any():
            continue
        cum_any: np.ndarray = np.logical_or.accumulate(is_pumped)
        k_idxs = np.clip(np.array(bins, dtype=int), 1, is_pumped.size) - 1
        vectors[str(pump_hash)] = cum_any[k_idxs].astype(float)

    if len(vectors) == 0:
        raise ValueError("scored_df has no non-empty cross-sections")
    return vectors


def _cross_section_indicator_vectors_topk_percent(
    scored_df: pd.DataFrame, bins: Sequence[float]
) -> dict[str, np.ndarray]:
    if len(bins) == 0:
        raise ValueError("bins must not be empty")
    if any(pct_bin < 0 for pct_bin in bins):
        raise ValueError(f"all bins must be >= 0, got {bins}")

    vectors: dict[str, np.ndarray] = {}
    bins_array: np.ndarray = np.array(bins, dtype=float)

    for pump_hash, df_cs in scored_df.groupby(COL_PUMP_HASH, sort=False):
        df_cs = df_cs.sort_values(by=COL_PROBAS_PRED, ascending=False, kind="stable")
        is_pumped: np.ndarray = df_cs[COL_IS_PUMPED].to_numpy(dtype=bool)
        if is_pumped.size == 0 or not is_pumped.any():
            continue
        cum_any: np.ndarray = np.logical_or.accumulate(is_pumped)
        k_raw: np.ndarray = np.ceil(is_pumped.size * bins_array).astype(int)
        positive: np.ndarray = k_raw > 0
        indicator: np.ndarray = np.zeros(bins_array.size, dtype=bool)
        if positive.any():
            k_idxs: np.ndarray = np.clip(k_raw[positive] - 1, 0, is_pumped.size - 1)
            indicator[positive] = cum_any[k_idxs]
        vectors[str(pump_hash)] = indicator.astype(float)

    if len(vectors) == 0:
        raise ValueError("scored_df has no non-empty cross-sections")
    return vectors


def _matrix_from_vectors(
    vectors: dict[str, np.ndarray], order: Sequence[str]
) -> np.ndarray:
    return np.vstack([vectors[pump_hash] for pump_hash in order])


def _bootstrap_indices(
    n_cross_sections: int, n_bootstrap: int, random_state: int
) -> np.ndarray:
    rng = np.random.default_rng(seed=random_state)
    return rng.integers(0, n_cross_sections, size=(n_bootstrap, n_cross_sections))


def _curve_ci_from_matrix(
    matrix: np.ndarray,
    bins: Sequence[int | float],
    n_bootstrap: int,
    alpha: float,
    random_state: int,
) -> pd.DataFrame:
    _validate_alpha(alpha=alpha)
    _validate_n_bootstrap(n_bootstrap=n_bootstrap)

    n_cross_sections: int = matrix.shape[0]
    sampled_indices: np.ndarray = _bootstrap_indices(
        n_cross_sections=n_cross_sections,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_curves: np.ndarray = matrix[sampled_indices].mean(axis=1)
    point_curve: np.ndarray = matrix.mean(axis=0)

    return pd.DataFrame(
        {
            "point_estimate": point_curve,
            "mean_bootstrap": bootstrap_curves.mean(axis=0),
            "std_bootstrap": bootstrap_curves.std(axis=0, ddof=1),
            "ci_lower": np.quantile(bootstrap_curves, alpha / 2, axis=0),
            "ci_upper": np.quantile(bootstrap_curves, 1 - alpha / 2, axis=0),
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
        },
        index=bins,
    )


def bootstrap_topk_ci(
    scored_df: pd.DataFrame,
    bins: Sequence[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    vectors: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk(
        scored_df=scored_df, bins=bins
    )
    order: list[str] = list(vectors.keys())
    matrix: np.ndarray = _matrix_from_vectors(vectors=vectors, order=order)
    return _curve_ci_from_matrix(
        matrix=matrix,
        bins=bins,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )


def bootstrap_topk_percent_ci(
    scored_df: pd.DataFrame,
    bins: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    vectors: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(
        scored_df=scored_df, bins=bins
    )
    order: list[str] = list(vectors.keys())
    matrix: np.ndarray = _matrix_from_vectors(vectors=vectors, order=order)
    return _curve_ci_from_matrix(
        matrix=matrix,
        bins=bins,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )


def _bootstrap_auc_distribution(
    matrix: np.ndarray,
    bins: Sequence[float],
    n_bootstrap: int,
    random_state: int,
    normaliser: float = 1.0,
) -> np.ndarray:
    sampled_indices: np.ndarray = _bootstrap_indices(
        n_cross_sections=matrix.shape[0],
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_curves: np.ndarray = matrix[sampled_indices].mean(axis=1)
    bins_array: np.ndarray = np.array(bins, dtype=float)
    return np.array(
        [auc(x=bins_array, y=curve) / normaliser for curve in bootstrap_curves],
        dtype=float,
    )


def bootstrap_topk_percent_auc_ci(
    scored_df: pd.DataFrame,
    bins: Sequence[float] | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    max_k_percent: float = 0.20,
    step: float = 0.005,
) -> BootstrapCI:
    if bins is None:
        bins = np.arange(0, max_k_percent + step, step)

    _validate_alpha(alpha=alpha)
    _validate_n_bootstrap(n_bootstrap=n_bootstrap)

    vectors: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(
        scored_df=scored_df, bins=bins
    )
    order: list[str] = list(vectors.keys())
    matrix: np.ndarray = _matrix_from_vectors(vectors=vectors, order=order)
    bins_array: np.ndarray = np.array(bins, dtype=float)
    normaliser: float = float(bins_array[-1] - bins_array[0])
    if normaliser <= 0:
        raise ValueError(
            f"bins range must be positive, got {bins_array[0]}..{bins_array[-1]}"
        )

    point_curve: np.ndarray = matrix.mean(axis=0)
    point_estimate: float = float(auc(x=bins_array, y=point_curve)) / normaliser
    bootstrap_aucs: np.ndarray = _bootstrap_auc_distribution(
        matrix=matrix,
        bins=bins,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        normaliser=normaliser,
    )
    ci_lower, ci_upper = np.quantile(bootstrap_aucs, [alpha / 2, 1 - alpha / 2])

    return BootstrapCI(
        metric="topk_percent_auc",
        point_estimate=point_estimate,
        mean_bootstrap=float(np.mean(bootstrap_aucs)),
        std_bootstrap=float(np.std(bootstrap_aucs, ddof=1)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def paired_bootstrap_topk_percent_auc_test(
    scored_df_a: pd.DataFrame,
    scored_df_b: pd.DataFrame,
    bins: Sequence[float] | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    max_k_percent: float = 0.20,
    step: float = 0.005,
) -> PairedBootstrapTestResult:
    if bins is None:
        bins = np.arange(0, max_k_percent + step, step)

    _validate_alpha(alpha=alpha)
    _validate_n_bootstrap(n_bootstrap=n_bootstrap)
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative={alternative}")

    vectors_a: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(
        scored_df=scored_df_a, bins=bins
    )
    vectors_b: dict[str, np.ndarray] = _cross_section_indicator_vectors_topk_percent(
        scored_df=scored_df_b, bins=bins
    )

    pump_hashes_a = set(vectors_a.keys())
    pump_hashes_b = set(vectors_b.keys())
    if pump_hashes_a != pump_hashes_b:
        missing_a = pump_hashes_b - pump_hashes_a
        missing_b = pump_hashes_a - pump_hashes_b
        raise ValueError(
            f"Cross-sections must match for paired bootstrap. Missing in A: {len(missing_a)}; Missing in B: {len(missing_b)}"
        )

    order: list[str] = list(vectors_a.keys())
    matrix_a: np.ndarray = _matrix_from_vectors(vectors=vectors_a, order=order)
    matrix_b: np.ndarray = _matrix_from_vectors(vectors=vectors_b, order=order)
    bins_array: np.ndarray = np.array(bins, dtype=float)
    normaliser: float = float(bins_array[-1] - bins_array[0])
    if normaliser <= 0:
        raise ValueError(
            f"bins range must be positive, got {bins_array[0]}..{bins_array[-1]}"
        )

    observed_diff: float = (
        float(
            auc(x=bins_array, y=matrix_a.mean(axis=0))
            - auc(x=bins_array, y=matrix_b.mean(axis=0))
        )
        / normaliser
    )

    sampled_indices: np.ndarray = _bootstrap_indices(
        n_cross_sections=matrix_a.shape[0],
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_curves_a: np.ndarray = matrix_a[sampled_indices].mean(axis=1)
    bootstrap_curves_b: np.ndarray = matrix_b[sampled_indices].mean(axis=1)
    auc_dist_a: np.ndarray = np.array(
        [auc(x=bins_array, y=curve) / normaliser for curve in bootstrap_curves_a],
        dtype=float,
    )
    auc_dist_b: np.ndarray = np.array(
        [auc(x=bins_array, y=curve) / normaliser for curve in bootstrap_curves_b],
        dtype=float,
    )
    diffs: np.ndarray = auc_dist_a - auc_dist_b

    ci_lower, ci_upper = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # Build the bootstrap null distribution by centring the paired per-event
    # contributions at zero. Counting signs in the ordinary (uncentred)
    # confidence-interval distribution is not a valid test of H0: difference=0.
    per_cross_section_auc_a: np.ndarray = (
        np.trapezoid(matrix_a, x=bins_array, axis=1) / normaliser
    )
    per_cross_section_auc_b: np.ndarray = (
        np.trapezoid(matrix_b, x=bins_array, axis=1) / normaliser
    )
    paired_contributions: np.ndarray = per_cross_section_auc_a - per_cross_section_auc_b
    centred_contributions: np.ndarray = (
        paired_contributions - paired_contributions.mean()
    )
    null_diffs: np.ndarray = centred_contributions[sampled_indices].mean(axis=1)

    # Add-one Monte Carlo estimator: a finite resample cannot report p=0.
    if alternative == "greater":
        p_value: float = float(
            (1 + int(np.sum(null_diffs >= observed_diff))) / (1 + n_bootstrap)
        )
    elif alternative == "less":
        p_value = float(
            (1 + int(np.sum(null_diffs <= observed_diff))) / (1 + n_bootstrap)
        )
    else:
        p_value = float(
            (1 + int(np.sum(np.abs(null_diffs) >= abs(observed_diff))))
            / (1 + n_bootstrap)
        )

    return PairedBootstrapTestResult(
        metric="topk_percent_auc",
        observed_diff=observed_diff,
        mean_diff_bootstrap=float(np.mean(diffs)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=p_value,
        alternative=alternative,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def _comparison_bins(
    bins: Sequence[float] | None, max_k_percent: float, step: float
) -> np.ndarray:
    if bins is None:
        if not np.isfinite(max_k_percent) or not 0 < max_k_percent <= 1:
            raise ValueError("max_k_percent must be finite and in (0, 1]")
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")
        # Include the requested endpoint exactly, including when step does not
        # divide max_k_percent. Do not integrate beyond the requested range.
        bins_array = np.append(np.arange(0, max_k_percent, step), max_k_percent)
    else:
        bins_array = np.asarray(bins, dtype=float)
    if (
        bins_array.ndim != 1
        or bins_array.size < 2
        or not np.isfinite(bins_array).all()
        or np.any((bins_array < 0) | (bins_array > 1))
        or np.any(np.diff(bins_array) <= 0)
    ):
        raise ValueError(
            "bins must contain at least two finite, strictly increasing values in [0, 1]"
        )
    return bins_array


def _validated_comparison_frame(
    scored_df: pd.DataFrame, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject incomplete events before the legacy indicator helper can skip them."""
    if not isinstance(scored_df, pd.DataFrame):
        raise ValueError(f"Scored data for {model_name!r} must be a pandas DataFrame")
    required_columns = [COL_PUMP_HASH, COL_IS_PUMPED, COL_PROBAS_PRED]
    missing_columns = set(required_columns) - set(scored_df.columns)
    if missing_columns:
        raise ValueError(
            f"Scored data for {model_name!r} is missing columns: {sorted(missing_columns)}"
        )
    if not scored_df.columns.is_unique:
        raise ValueError(
            f"Scored data for {model_name!r} must have unique column names"
        )
    if scored_df.empty:
        raise ValueError(f"Scored data for {model_name!r} must not be empty")

    frame = scored_df[required_columns].copy()
    if frame[COL_PUMP_HASH].isna().any():
        raise ValueError(
            f"Scored data for {model_name!r} contains missing cross-section IDs"
        )
    if frame[COL_PUMP_HASH].map(str).nunique() != frame[COL_PUMP_HASH].nunique():
        raise ValueError(
            f"Cross-section IDs for {model_name!r} must remain unique when converted to strings"
        )
    if frame[COL_IS_PUMPED].isna().any() or not frame[COL_IS_PUMPED].isin([0, 1]).all():
        raise ValueError(
            f"Scored data for {model_name!r} must contain nonmissing binary labels"
        )
    try:
        if np.iscomplexobj(frame[COL_PROBAS_PRED].to_numpy()):
            raise ValueError("Complex scores are not supported")
        scores = frame[COL_PROBAS_PRED].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scored data for {model_name!r} must contain finite numeric scores"
        ) from exc
    if not np.isfinite(scores).all():
        raise ValueError(
            f"Scored data for {model_name!r} must contain finite numeric scores"
        )
    frame[COL_PROBAS_PRED] = scores
    frame[COL_IS_PUMPED] = frame[COL_IS_PUMPED].astype(bool)
    event_counts = frame.groupby(COL_PUMP_HASH, sort=False, observed=True)[
        COL_IS_PUMPED
    ].agg(["size", "sum"])
    if (event_counts["sum"] == 0).any():
        raise ValueError(
            f"Every cross-section for {model_name!r} must contain at least one positive label"
        )
    if len(event_counts) < 2:
        raise ValueError(
            "At least two cross-sections are required for bootstrap model comparison"
        )
    return frame, event_counts


def _holm_adjust_p_values(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values for one family of comparisons."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1))
    )
    return adjusted


def _comparison_event_aucs(
    scored_datasets: Mapping[str, pd.DataFrame], bins_array: np.ndarray
) -> np.ndarray:
    """Validate matching evaluation events and integrate each event once per model."""
    normaliser = float(bins_array[-1] - bins_array[0])
    reference_counts: pd.DataFrame | None = None
    event_order: list[str] = []
    event_aucs = []

    for model_name, scored_df in scored_datasets.items():
        frame, event_counts = _validated_comparison_frame(
            scored_df=scored_df, model_name=model_name
        )
        if reference_counts is None:
            reference_counts = event_counts
            event_order = sorted(str(event_id) for event_id in event_counts.index)
        _validate_matching_events(
            event_counts, reference_counts, event_order, model_name
        )
        vectors = _cross_section_indicator_vectors_topk_percent(
            scored_df=frame, bins=bins_array
        )
        matrix = _matrix_from_vectors(vectors=vectors, order=event_order)
        event_aucs.append(np.trapezoid(matrix, x=bins_array, axis=1) / normaliser)

    return np.column_stack(event_aucs)


def _validate_matching_events(
    counts: pd.DataFrame,
    reference: pd.DataFrame,
    event_order: Sequence[str],
    model_name: str,
) -> None:
    ids_match = set(counts.index) == set(reference.index)
    string_ids_match = sorted(map(str, counts.index)) == list(event_order)
    if not ids_match or not string_ids_match:
        raise ValueError(
            f"Cross-sections must match for every model; event IDs differ for {model_name!r}"
        )
    aligned_counts = counts.reindex(reference.index)
    if not np.array_equal(aligned_counts.to_numpy(), reference.to_numpy()):
        raise ValueError(
            f"Cross-section row and positive-label counts must match for {model_name!r}"
        )


def _resample_event_means(
    per_event_auc: np.ndarray, n_bootstrap: int, random_state: int
) -> np.ndarray:
    """Use one shared event draw for every fixed model in each replicate."""
    # AUC is linear in the curve, so bootstrapping these per-event scalars is
    # equivalent to rebuilding and integrating every resampled top-k curve.
    n_cross_sections, n_models = per_event_auc.shape
    bootstrap_aucs = np.empty((n_bootstrap, n_models), dtype=float)
    rng = np.random.default_rng(seed=random_state)
    # Bound the temporary gather array rather than allocating B x events x
    # models x bins. Chunking preserves the random sequence of a single draw.
    chunk_size = max(1, min(n_bootstrap, 1_000_000 // (n_cross_sections * n_models)))
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sampled_indices = rng.integers(
            0, n_cross_sections, size=(stop - start, n_cross_sections)
        )
        bootstrap_aucs[start:stop] = per_event_auc[sampled_indices].mean(axis=1)

    return bootstrap_aucs


def _auc_summary(
    model_names: Sequence[str],
    per_event_auc: np.ndarray,
    bootstrap_aucs: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    n_bootstrap = len(bootstrap_aucs)
    n_cross_sections = len(per_event_auc)
    point_estimates = per_event_auc.mean(axis=0)
    summary_rows = []
    for model_idx, model_name in enumerate(model_names):
        samples = bootstrap_aucs[:, model_idx]
        ci_lower, ci_upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
        summary_rows.append(
            {
                "model": model_name,
                **BootstrapCI(
                    metric="topk_percent_auc",
                    point_estimate=float(point_estimates[model_idx]),
                    mean_bootstrap=float(samples.mean()),
                    std_bootstrap=float(samples.std(ddof=1)),
                    ci_lower=float(ci_lower),
                    ci_upper=float(ci_upper),
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                ).to_dict(),
                "n_cross_sections": n_cross_sections,
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("model")
    summary.attrs["ci_method"] = "pointwise percentile"

    return summary


def _paired_auc_comparisons(
    model_names: Sequence[str],
    per_event_auc: np.ndarray,
    bootstrap_aucs: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    n_bootstrap = len(bootstrap_aucs)
    n_cross_sections, n_models = per_event_auc.shape
    comparison_rows = []
    # Allow roundoff at equality in the null-tail count. In particular, equal
    # models or exactly zero observed differences must never be significant.
    tolerance = 32 * np.finfo(float).eps
    for model_idx_a, model_idx_b in combinations(range(n_models), 2):
        observed_diff = float(
            (per_event_auc[:, model_idx_a] - per_event_auc[:, model_idx_b]).mean()
        )
        diffs = bootstrap_aucs[:, model_idx_a] - bootstrap_aucs[:, model_idx_b]
        null_diffs = diffs - observed_diff
        p_value = float(
            (1 + np.count_nonzero(np.abs(null_diffs) >= abs(observed_diff) - tolerance))
            / (n_bootstrap + 1)
        )
        ci_lower, ci_upper = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
        comparison_rows.append(
            {
                "model_a": model_names[model_idx_a],
                "model_b": model_names[model_idx_b],
                **PairedBootstrapTestResult(
                    metric="topk_percent_auc",
                    observed_diff=observed_diff,
                    mean_diff_bootstrap=float(diffs.mean()),
                    ci_lower=float(ci_lower),
                    ci_upper=float(ci_upper),
                    p_value=p_value,
                    alternative="two-sided",
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                ).to_dict(),
                "std_diff_bootstrap": float(diffs.std(ddof=1)),
                "n_cross_sections": n_cross_sections,
            }
        )
    pairwise = pd.DataFrame(comparison_rows)
    pairwise["p_value_holm"] = _holm_adjust_p_values(pairwise["p_value"].to_numpy())
    pairwise["reject_h0"] = pairwise["p_value_holm"] < alpha
    pairwise["better_model"] = [
        (row.model_a if row.observed_diff > 0 else row.model_b)
        if row.reject_h0
        else None
        for row in pairwise.itertuples(index=False)
    ]
    pairwise.attrs["ci_method"] = "pointwise percentile"
    pairwise.attrs["p_value_adjustment"] = "Holm"
    return pairwise


def compare_models_bootstrap_topk_percent_auc(
    scored_datasets: Mapping[str, pd.DataFrame],
    bins: Sequence[float] | None = None,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
    max_k_percent: float = 0.20,
    step: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every model using a paired bootstrap over evaluation events.

    Each mapping value contains scores for the same evaluation cross-sections,
    with matching row and positive-label counts per event. Missing scores, IDs,
    or positive-free events are rejected rather than silently excluded. The
    caller remains responsible for using the same assets and ground truth;
    these three score columns do not identify individual assets.

    Normalised top-k-percent AUC is integrated once per event. Each bootstrap
    replicate resamples whole events, equally weighted, using the same draws
    for every model. Events are sorted by ID for reproducibility; tied model
    scores retain their input order, as in the other top-k metric functions.

    Returns a model summary indexed by ``model`` and one row per unordered
    model pair. Positive ``observed_diff`` means model A is better. All
    ``ci_lower``/``ci_upper`` values are *pointwise*, unadjusted percentile
    confidence intervals, not simultaneous intervals across models or pairs.
    Two-sided p-values use a null distribution centred at zero and the add-one
    Monte Carlo correction. ``p_value_holm`` controls the family-wise error
    rate across all pairs; only its rejection sets ``better_model``.

    Inference assumes events are independent. It concerns these fixed fitted
    models on the supplied sample; it does not account for training or
    hyperparameter selection on that same sample, or distribution changes in
    later periods. Use an untouched test set for a final selected-model claim.
    """
    _validate_alpha(alpha=alpha)
    if (
        isinstance(n_bootstrap, (bool, np.bool_))
        or not isinstance(n_bootstrap, (int, np.integer))
        or n_bootstrap < 2
    ):
        raise ValueError(
            "n_bootstrap must be an integer >= 2 for bootstrap model comparison"
        )
    if not isinstance(scored_datasets, Mapping) or len(scored_datasets) < 2:
        raise ValueError(
            "At least two models are required for bootstrap model comparison"
        )
    if any(not isinstance(name, str) or not name.strip() for name in scored_datasets):
        raise ValueError("Model names must be nonempty strings")
    bins_array = _comparison_bins(bins=bins, max_k_percent=max_k_percent, step=step)
    model_names = list(scored_datasets)
    per_event_auc = _comparison_event_aucs(scored_datasets, bins_array)
    bootstrap_aucs = _resample_event_means(per_event_auc, n_bootstrap, random_state)
    summary = _auc_summary(model_names, per_event_auc, bootstrap_aucs, alpha)
    pairwise = _paired_auc_comparisons(
        model_names, per_event_auc, bootstrap_aucs, alpha
    )
    return summary, pairwise
