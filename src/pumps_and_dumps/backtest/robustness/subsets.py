import logging
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import COL_PUMP_HASH
from pumps_and_dumps.backtest.metrics import (
    calculate_topk_percent,
    calculate_topk_percent_auc,
)
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType, Sample
from pumps_and_dumps.backtest.pipelines.base_model import ImplementsRank
from pumps_and_dumps.backtest.pipelines.inspection import feature_importances


def _copy_datasets(
    datasets: Dict[DatasetType, pd.DataFrame],
) -> Dict[DatasetType, pd.DataFrame]:
    return {ds_type: df.copy(deep=True) for ds_type, df in datasets.items()}


def subset_cross_sections(
    df: pd.DataFrame,
    subset_fraction: float,
    random_state: int,
    group_col: str = COL_PUMP_HASH,
    min_cross_sections: int = 1,
) -> pd.DataFrame:
    """
    Sample a subset of complete cross-sections from a dataframe.
    """
    if not 0 < subset_fraction <= 1:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
    if min_cross_sections < 1:
        raise ValueError(f"min_cross_sections must be >= 1, got {min_cross_sections}")

    cross_sections: np.ndarray = df[group_col].drop_duplicates().to_numpy()
    n_cross_sections: int = int(cross_sections.size)
    if n_cross_sections == 0:
        raise ValueError("Dataset has no cross-sections to sample from")

    n_sample: int = max(
        min_cross_sections, int(np.ceil(n_cross_sections * subset_fraction))
    )
    n_sample = min(n_sample, n_cross_sections)

    rng = np.random.default_rng(seed=random_state)
    selected: np.ndarray = rng.choice(cross_sections, size=n_sample, replace=False)
    return df[df[group_col].isin(selected)].copy(deep=True).reset_index(drop=True)


def _extract_feature_importance(model: Any) -> pd.Series:
    return feature_importances(model)


def _create_subset_sample(
    pipeline: Any, datasets: Dict[DatasetType, pd.DataFrame]
) -> Sample:
    create_from_datasets = getattr(pipeline, "create_sample_from_datasets", None)
    if create_from_datasets is not None:
        return create_from_datasets(_copy_datasets(datasets))

    # Compatibility for external pipeline factories predating explicit sample
    # inputs. Project pipelines always use create_sample_from_datasets above.
    original_build_datasets: Callable[[], Any] = pipeline.build_datasets

    def _build_datasets_override(self: Any) -> Dict[DatasetType, pd.DataFrame]:
        return _copy_datasets(datasets)

    pipeline.build_datasets = MethodType(_build_datasets_override, pipeline)  # type: ignore[method-assign]
    try:
        return pipeline.create_sample()
    finally:
        pipeline.build_datasets = original_build_datasets


def _test_metrics(
    model: ImplementsRank, dataset: Dataset, bins: Sequence[float]
) -> dict[str, float | int]:
    topk_percent_auc = calculate_topk_percent_auc(model=model, dataset=dataset)
    topk_values = calculate_topk_percent(model=model, dataset=dataset, bins=bins)
    return {
        "test_rows": len(dataset.all_data()),
        "test_cross_sections": int(dataset.all_data()[COL_PUMP_HASH].nunique()),
        "topk_percent_auc": float(topk_percent_auc),
        **{
            f"topk_percent@{float(bin_value):g}": float(value)
            for bin_value, value in topk_values.items()
        },
    }


def _importance_rows(model: ImplementsRank, run_idx: int, seed: int) -> list[dict]:
    try:
        values = _extract_feature_importance(model)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to extract feature importances on robustness run {run_idx} (seed {seed})"
        ) from exc
    return [
        {
            "run_idx": run_idx,
            "seed": seed,
            "feature": str(name),
            "feature_importance": float(value),
        }
        for name, value in values.items()
    ]


def _save_csv(frame: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_cross_section_subset_robustness(
    pipeline_factory: Callable[[], Any],
    subset_fraction: float,
    n_runs: int,
    tuned: bool = False,
    topk_bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
    base_seed: int = 42,
    output_path: str | Path | None = None,
    collect_feature_importances: bool = False,
    feature_importance_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Train/evaluate a pipeline repeatedly where each run uses a random subset of train-cross-sections.
    Tuning, independent validation, and test splits are kept fixed.

    When ``collect_feature_importances`` is True the model's feature importances are also captured on
    every run (the same resample-train-fit loop that produces the Top@k%AUC distribution), giving a
    bootstrap distribution of feature importances. The long-format frame (one row per run x feature with
    columns ``run_idx``, ``seed``, ``feature``, ``feature_importance``) is attached to the returned frame
    under ``results.attrs["feature_importances"]`` and, if ``feature_importance_output_path`` is given,
    written to CSV. Use :func:`summarise_feature_importance_distribution` to reduce it to per-feature
    quantiles sorted by median.

    Note: ``results.attrs`` is in-memory only and is not preserved by ``results.to_csv(...)``; pass
    ``feature_importance_output_path`` to persist the feature importances, and reload them from that CSV
    rather than from a reloaded metrics frame.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    if collect_feature_importances and feature_importance_output_path is None:
        logging.warning(
            "collect_feature_importances=True but feature_importance_output_path is None; the bootstrap "
            "feature importances will only be available via results.attrs['feature_importances'] in memory "
            "and are NOT persisted when results are written to CSV."
        )

    template_pipeline = pipeline_factory()
    full_datasets: Dict[DatasetType, pd.DataFrame] = template_pipeline.build_datasets()
    train_df: pd.DataFrame = full_datasets[DatasetType.TRAIN]

    rows: List[Dict[str, float | int]] = []
    fi_rows: List[Dict[str, float | int | str]] = []
    for run_idx in range(n_runs):
        seed: int = base_seed + run_idx
        train_subset: pd.DataFrame = subset_cross_sections(
            df=train_df, subset_fraction=subset_fraction, random_state=seed
        )
        run_datasets = full_datasets | {DatasetType.TRAIN: train_subset}

        pipeline = pipeline_factory()
        sample = _create_subset_sample(pipeline, run_datasets)

        model = pipeline.train(sample=sample, tuned=tuned)
        dataset_test = sample.get_dataset(ds_type=DatasetType.TEST)

        result: Dict[str, float | int] = {
            "run_idx": run_idx,
            "seed": seed,
            "subset_fraction": subset_fraction,
            "train_rows": int(train_subset.shape[0]),
            "train_cross_sections": int(train_subset[COL_PUMP_HASH].nunique()),
            **_test_metrics(model, dataset_test, topk_bins),
        }
        rows.append(result)

        if collect_feature_importances:
            fi_rows.extend(_importance_rows(model, run_idx, seed))

    results: pd.DataFrame = pd.DataFrame(rows)

    feature_importances: pd.DataFrame = pd.DataFrame(
        fi_rows, columns=["run_idx", "seed", "feature", "feature_importance"]
    )
    results.attrs["feature_importances"] = feature_importances

    _save_csv(results, output_path)
    if collect_feature_importances:
        _save_csv(feature_importances, feature_importance_output_path)

    return results


def summarise_robustness_distribution(
    results: pd.DataFrame,
    metric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = [col for col in results.columns if col.startswith("topk_")]

    summary: pd.DataFrame = (
        results[list(metric_cols)].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    )
    return summary


def summarise_feature_importance_distribution(
    feature_importances: pd.DataFrame,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    top_n: Optional[int] = None,
    sort_quantile: float = 0.5,
) -> pd.DataFrame:
    """
    Reduce the bootstrap feature-importance frame (long format, as returned in
    ``results.attrs["feature_importances"]`` by :func:`run_cross_section_subset_robustness`) to a
    per-feature summary: ``mean``, ``std`` and one column per requested quantile
    (``p10``, ``p25``, ``p50`` ...). Rows are sorted by the median importance (``sort_quantile``)
    in descending order, so the most important features sit at the top; pass ``top_n`` to keep only
    the leading features.
    """
    if feature_importances.empty:
        raise ValueError(
            "feature_importances is empty; run with collect_feature_importances=True first"
        )
    if not 0 <= sort_quantile <= 1:
        raise ValueError(f"sort_quantile must be in [0, 1], got {sort_quantile}")

    grouped = feature_importances.groupby("feature")["feature_importance"]
    summary: pd.DataFrame = pd.DataFrame(
        {"mean": grouped.mean(), "std": grouped.std(ddof=1)}
    )
    for q in quantiles:
        summary[f"p{int(round(q * 100))}"] = grouped.quantile(q)

    sort_col: str = f"p{int(round(sort_quantile * 100))}"
    if sort_col not in summary.columns:
        summary[sort_col] = grouped.quantile(sort_quantile)
    summary = summary.sort_values(by=sort_col, ascending=False)

    if top_n is not None:
        summary = summary.head(top_n)
    return summary


def evaluate_subperiod_metrics(
    model: ImplementsRank,
    dataset_df: pd.DataFrame,
    date_col: str,
    split_date: str,
    topk_bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
    min_pumps: int = 10,
    feature_set: FeatureSet | None = None,
) -> pd.DataFrame:
    """
    Split the test set into early and late subperiods by pump event date
    and evaluate Top@k% and Top@k%AUC on each half independently.

    Subperiods with fewer than ``min_pumps`` pump events are skipped with a
    warning. Reporting metrics on tiny subperiods (e.g. n=3) produces noisy
    numbers that are unsafe to interpret as performance differences.

    Parameters
    ----------
    model : ImplementsRank
        Trained model with a .rank() method.
    dataset_df : pd.DataFrame
        The full test dataset including a date column and pump_hash.
    date_col : str
        Column containing the event timestamp used for splitting.
    split_date : str
        ISO date string to split the test set into early (< split_date) and late (>= split_date).
    topk_bins : Sequence[float]
        Percentage bins for Top@k% evaluation.
    min_pumps : int
        Minimum pump events required for a subperiod to be evaluated.

    Returns
    -------
    pd.DataFrame
        One row per subperiod with metric columns. Subperiods below
        ``min_pumps`` are omitted.
    """
    dataset_df = dataset_df.copy()
    dataset_df[date_col] = pd.to_datetime(dataset_df[date_col])
    split_ts = pd.Timestamp(split_date)

    pump_dates = dataset_df.groupby(COL_PUMP_HASH)[date_col].min()
    early_hashes = pump_dates[pump_dates < split_ts].index
    late_hashes = pump_dates[pump_dates >= split_ts].index

    feature_set = FeatureSet.auto() if feature_set is None else feature_set

    rows: List[Dict[str, float | int | str]] = []
    for label, hashes in [("early", early_hashes), ("late", late_hashes)]:
        subset = dataset_df[dataset_df[COL_PUMP_HASH].isin(hashes)].reset_index(
            drop=True
        )
        n_pumps = int(subset[COL_PUMP_HASH].nunique()) if not subset.empty else 0
        if n_pumps < min_pumps:
            logging.warning(
                "Skipping '%s' subperiod: only %d pump(s), below min_pumps=%d. "
                "Metrics on tiny subperiods are unstable.",
                label,
                n_pumps,
                min_pumps,
            )
            continue
        sub_dataset = Dataset(
            data=subset, feature_set=feature_set, ds_type=DatasetType.TEST
        )
        sub_dataset.add_pool()
        topk_pct_auc: float = calculate_topk_percent_auc(
            model=model, dataset=sub_dataset
        )
        topk_values: pd.Series = calculate_topk_percent(
            model=model, dataset=sub_dataset, bins=topk_bins
        )
        result: Dict[str, float | int | str] = {
            "subperiod": label,
            "n_pumps": n_pumps,
            "n_rows": int(subset.shape[0]),
            "topk_percent_auc": float(topk_pct_auc),
        }
        for pct_bin, val in topk_values.items():
            result[f"topk_percent@{float(pct_bin):g}"] = float(val)
        rows.append(result)

    return pd.DataFrame(rows)
