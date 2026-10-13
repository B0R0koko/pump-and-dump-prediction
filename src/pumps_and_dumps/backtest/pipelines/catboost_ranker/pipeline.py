import logging
from functools import partial
from typing import Dict, Any

import pandas as pd
from optuna import Study, Trial
from overrides import overrides

from pumps_and_dumps.backtest.pipelines.base_pipeline import (
    BasePipeline,
    cross_section_standardisation,
    prepare_cross_section_features,
)
from pumps_and_dumps.backtest.pipelines.catboost_ranker.model import CatboostRankerModel
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.backtest.pipelines.catboost_samples import add_catboost_pools
from pumps_and_dumps.backtest.datasets.columns import COL_PUMP_ID
from pumps_and_dumps.backtest.metrics import calculate_topk_percent_auc
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample, Dataset
from pumps_and_dumps.core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "YetiRank:mode=NDCG",
    "border_count": 255,
    "verbose": False,
    "random_seed": 42,
}


def build_return_rank_labels(df: pd.DataFrame) -> pd.Series:
    """Rank returns highest-first; percentile rank ``1 / N`` is the first position."""
    return df.groupby(COL_PUMP_ID, sort=False)["target_return@5MIN"].rank(
        pct=True, ascending=False
    )


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 10, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
    }

    model: CatboostRankerModel = CatboostRankerModel(params=_BASE_PARAMS | tuned_params)
    model.train(sample=sample)

    tuning: Dataset = sample.get_dataset(ds_type=DatasetType.TUNING)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=tuning)
    return topkpauc


class CatboostRankerPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_set.target = "asset_return_rank"  # first rank has the highest return within the cross-section

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the supervised ranking label after past-only feature preprocessing."""
        df = prepare_cross_section_features(df)
        # Fillna target for ranker which is target_return@5MIN
        df["target_return@5MIN"] = df["target_return@5MIN"].fillna(0)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        # Create highest-first percentile ranks (the first position is 1 / group size).
        df_scaled[self.feature_set.target] = build_return_rank_labels(df_scaled)
        if not df_scaled[COL_PUMP_ID].is_monotonic_increasing:
            raise ValueError("GroupId must be monotonic increasing")
        return df_scaled

    def create_sample_from_datasets(
        self, datasets: Dict[DatasetType, pd.DataFrame]
    ) -> Sample:
        sample = super().create_sample_from_datasets(datasets)
        return add_catboost_pools(sample, group_column=COL_PUMP_ID)

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Running <optimize_parameters> for CatboostRankerPipeline")
        sample: Sample = self.create_sample()
        study = create_study(
            "CatboostRankerPipelineStudy", start_new=True, storage=self.study_storage
        )
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostRankerModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            model_params = self.get_model_params(
                base_params=_BASE_PARAMS, study_name="CatboostRankerPipelineStudy"
            )

        model: CatboostRankerModel = CatboostRankerModel(params=model_params)
        model.train(sample=sample)
        return model


def main():
    configure_logging()
    pipe = CatboostRankerPipeline()
    pipe.optimize_parameters(n_trials=100)


if __name__ == "__main__":
    main()
