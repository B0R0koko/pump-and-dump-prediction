import logging
from functools import partial
from typing import Dict, Any

import pandas as pd
from optuna import Trial, Study
from overrides import overrides

from pumps_and_dumps.backtest.pipelines.base_pipeline import (
    BasePipeline,
    prepare_cross_section_features,
    cross_section_standardisation,
    impute_training_features,
)
from pumps_and_dumps.backtest.pipelines.random_forest.model import RandomForestModel
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.backtest.metrics import calculate_topk_percent_auc
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample, Dataset
from pumps_and_dumps.core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "criterion": "gini",
    "n_jobs": -1,
    "verbose": False,
    "random_state": 42,
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "class_weight": {0: 1, 1: trial.suggest_float("class_weight", 10, 300)},
        "max_features": trial.suggest_float("max_features", 0.5, 1),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }

    model: RandomForestModel = RandomForestModel(_BASE_PARAMS | tuned_params)
    model.train(sample=sample)

    tuning: Dataset = sample.get_dataset(ds_type=DatasetType.TUNING)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=tuning)
    return topkpauc


class RandomForestPipeline(BasePipeline):
    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute NaNs with past-only priors, then standardize by cross-section."""
        df = prepare_cross_section_features(df)
        df = impute_training_features(df=df, feature_set=self.feature_set)
        return cross_section_standardisation(df=df)

    @overrides
    def get_model_params(
        self, base_params: Dict[str, Any], study_name: str
    ) -> Dict[str, Any]:
        model_params = super().get_model_params(base_params, study_name)
        model_params["class_weight"] = {0: 1, 1: model_params["class_weight"]}
        return model_params

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Running <optimize_parameters> for RandomForestPipeline")
        sample: Sample = self.create_sample()
        study = create_study(
            "RandomForestPipelineStudy", start_new=True, storage=self.study_storage
        )
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> RandomForestModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            # Read optimal parameters from optuna.RDBStorage
            model_params = self.get_model_params(
                base_params=_BASE_PARAMS, study_name="RandomForestPipelineStudy"
            )
        model: RandomForestModel = RandomForestModel(params=model_params)
        model.train(sample=sample)
        return model


if __name__ == "__main__":
    configure_logging()
    pipeline = RandomForestPipeline()
    pipeline.optimize_parameters(n_trials=100)
