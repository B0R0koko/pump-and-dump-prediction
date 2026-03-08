from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from analysis.utils.columns import COL_PUMP_HASH
from analysis.utils.metrics import calculate_topk_percent, calculate_topk_percent_auc
from analysis.utils.sample import DatasetType


def _copy_datasets(datasets: Dict[DatasetType, pd.DataFrame]) -> Dict[DatasetType, pd.DataFrame]:
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

    n_sample: int = max(min_cross_sections, int(np.ceil(n_cross_sections * subset_fraction)))
    n_sample = min(n_sample, n_cross_sections)

    rng = np.random.default_rng(seed=random_state)
    selected: np.ndarray = rng.choice(cross_sections, size=n_sample, replace=False)
    return df[df[group_col].isin(selected)].copy(deep=True).reset_index(drop=True)


def _override_build_datasets(pipeline: Any, datasets: Dict[DatasetType, pd.DataFrame]) -> Callable[[], Any]:
    original_build_datasets: Callable[[], Any] = pipeline.build_datasets

    def _build_datasets_override(self: Any) -> Dict[DatasetType, pd.DataFrame]:
        return _copy_datasets(datasets)

    pipeline.build_datasets = MethodType(_build_datasets_override, pipeline)  # type: ignore[method-assign]
    return original_build_datasets


def run_cross_section_subset_robustness(
        pipeline_factory: Callable[[], Any],
        subset_fraction: float,
        n_runs: int,
        tuned: bool = False,
        topk_bins: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
        base_seed: int = 42,
        output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Train/evaluate a pipeline repeatedly where each run uses a random subset of train-cross-sections.
    Validation and test splits are kept fixed.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    template_pipeline = pipeline_factory()
    full_datasets: Dict[DatasetType, pd.DataFrame] = template_pipeline.build_datasets()
    train_df: pd.DataFrame = full_datasets[DatasetType.TRAIN]
    val_df: pd.DataFrame = full_datasets[DatasetType.VALIDATION]
    test_df: pd.DataFrame = full_datasets[DatasetType.TEST]

    rows: List[Dict[str, float | int]] = []
    for run_idx in range(n_runs):
        seed: int = base_seed + run_idx
        train_subset: pd.DataFrame = subset_cross_sections(
            df=train_df, subset_fraction=subset_fraction, random_state=seed
        )
        run_datasets: Dict[DatasetType, pd.DataFrame] = {
            DatasetType.TRAIN: train_subset,
            DatasetType.VALIDATION: val_df,
            DatasetType.TEST: test_df,
        }

        pipeline = pipeline_factory()
        original_build_datasets: Callable[[], Any] = _override_build_datasets(
            pipeline=pipeline, datasets=run_datasets
        )
        try:
            sample = pipeline.create_sample()
        finally:
            pipeline.build_datasets = original_build_datasets  # type: ignore[method-assign]

        model = pipeline.train(sample=sample, tuned=tuned)
        dataset_test = sample.get_dataset(ds_type=DatasetType.TEST)

        topk_percent_auc: float = calculate_topk_percent_auc(model=model, dataset=dataset_test)
        topk_values: pd.Series = calculate_topk_percent(
            model=model,
            dataset=dataset_test,
            bins=topk_bins,
        )

        result: Dict[str, float | int] = {
            "run_idx": run_idx,
            "seed": seed,
            "subset_fraction": subset_fraction,
            "train_rows": int(train_subset.shape[0]),
            "train_cross_sections": int(train_subset[COL_PUMP_HASH].nunique()),
            "test_rows": int(dataset_test.all_data().shape[0]),
            "test_cross_sections": int(dataset_test.all_data()[COL_PUMP_HASH].nunique()),
            "topk_percent_auc": float(topk_percent_auc),
        }
        for pct_bin, metric_value in topk_values.items():
            result[f"topk_percent@{float(pct_bin):g}"] = float(metric_value)
        rows.append(result)

    results: pd.DataFrame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

    return results


def summarise_robustness_distribution(
        results: pd.DataFrame,
        metric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = [col for col in results.columns if col.startswith("topk_")]

    summary: pd.DataFrame = results[list(metric_cols)].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    return summary
