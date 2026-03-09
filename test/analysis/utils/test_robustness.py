from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.pipelines.BaseModel import ImplementsRank
from analysis.robust.robustness import (
    run_cross_section_subset_robustness,
    subset_cross_sections,
    summarise_robustness_distribution,
)
from analysis.utils.columns import COL_IS_PUMPED, COL_PUMP_HASH
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import Dataset, DatasetType, Sample

FEATURE_SET = FeatureSet(
    numeric_features=["score"],
    target=COL_IS_PUMPED,
    categorical_features=None,
    eval_fields=[COL_PUMP_HASH],
)


class DummyModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> np.ndarray:
        return dataset.get_data()["score"].to_numpy()


class DummyPipeline:

    def __init__(
        self,
        datasets: Dict[DatasetType, pd.DataFrame],
        seen_train_cross_sections: List[int],
    ):
        self._datasets = datasets
        self._seen_train_cross_sections = seen_train_cross_sections
        self.feature_set = FEATURE_SET

    def build_datasets(self) -> Dict[DatasetType, pd.DataFrame]:
        return {ds_type: df.copy(deep=True) for ds_type, df in self._datasets.items()}

    def create_sample(self) -> Sample:
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        self._seen_train_cross_sections.append(
            int(datasets[DatasetType.TRAIN][COL_PUMP_HASH].nunique())
        )
        return Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)

    def train(self, sample: Sample, tuned: bool = False) -> DummyModel:  # noqa: ARG002
        return DummyModel()


def _build_cross_sections(
    prefix: str, n_cross_sections: int, rows_per_cross_section: int
) -> pd.DataFrame:
    rows = []
    for cs_idx in range(n_cross_sections):
        pump_hash = f"{prefix}-{cs_idx}"
        for row_idx in range(rows_per_cross_section):
            rows.append(
                {
                    COL_PUMP_HASH: pump_hash,
                    COL_IS_PUMPED: row_idx == 0,
                    "score": float(rows_per_cross_section - row_idx),
                }
            )

    return pd.DataFrame(rows)


def test_subset_cross_sections_returns_complete_groups() -> None:
    df: pd.DataFrame = _build_cross_sections(
        prefix="train", n_cross_sections=6, rows_per_cross_section=3
    )

    subset_a: pd.DataFrame = subset_cross_sections(
        df=df, subset_fraction=0.5, random_state=17
    )
    subset_b: pd.DataFrame = subset_cross_sections(
        df=df, subset_fraction=0.5, random_state=17
    )

    assert subset_a[COL_PUMP_HASH].nunique() == 3
    assert subset_b[COL_PUMP_HASH].nunique() == 3
    assert set(subset_a[COL_PUMP_HASH].unique()) == set(
        subset_b[COL_PUMP_HASH].unique()
    )
    assert subset_a.groupby(COL_PUMP_HASH).size().nunique() == 1
    assert subset_a.groupby(COL_PUMP_HASH).size().iloc[0] == 3


def test_run_cross_section_subset_robustness_saves_distribution(tmp_path: Path) -> None:
    datasets: Dict[DatasetType, pd.DataFrame] = {
        DatasetType.TRAIN: _build_cross_sections(
            prefix="train", n_cross_sections=6, rows_per_cross_section=3
        ),
        DatasetType.VALIDATION: _build_cross_sections(
            prefix="val", n_cross_sections=2, rows_per_cross_section=3
        ),
        DatasetType.TEST: _build_cross_sections(
            prefix="test", n_cross_sections=4, rows_per_cross_section=3
        ),
    }
    seen_train_cross_sections: List[int] = []

    def factory() -> DummyPipeline:
        return DummyPipeline(
            datasets=datasets, seen_train_cross_sections=seen_train_cross_sections
        )

    output_path: Path = tmp_path / "robustness.csv"
    results: pd.DataFrame = run_cross_section_subset_robustness(
        pipeline_factory=factory,
        subset_fraction=0.5,
        n_runs=4,
        tuned=False,
        topk_bins=[0.1, 0.5],
        base_seed=10,
        output_path=output_path,
    )

    assert results.shape[0] == 4
    assert {"run_idx", "seed", "train_cross_sections", "topk_percent_auc"}.issubset(
        results.columns
    )
    assert (results["train_cross_sections"] == 3).all()
    assert len(seen_train_cross_sections) == 4
    assert all(cs == 3 for cs in seen_train_cross_sections)
    assert (results["topk_percent_auc"] > 0.9).all()
    assert results["topk_percent_auc"].nunique() == 1
    assert np.allclose(results["topk_percent@0.1"].to_numpy(), np.ones(4))
    assert np.allclose(results["topk_percent@0.5"].to_numpy(), np.ones(4))
    assert output_path.exists()

    loaded: pd.DataFrame = pd.read_csv(output_path)
    assert loaded.shape == results.shape


def test_summarise_robustness_distribution_uses_topk_columns() -> None:
    results: pd.DataFrame = pd.DataFrame(
        {
            "run_idx": [0, 1, 2],
            "topk_percent_auc": [0.6, 0.7, 0.8],
            "topk_percent@0.1": [0.2, 0.3, 0.4],
        }
    )
    summary: pd.DataFrame = summarise_robustness_distribution(results=results)

    assert "topk_percent_auc" in summary.index
    assert "topk_percent@0.1" in summary.index
    assert "mean" in summary.columns
    assert "50%" in summary.columns
