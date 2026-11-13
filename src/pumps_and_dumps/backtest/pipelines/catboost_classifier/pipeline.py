import logging
from functools import partial
from typing import Dict, Any

import pandas as pd
from optuna import Study, Trial

from pumps_and_dumps.backtest.pipelines.base_pipeline import BasePipeline
from pumps_and_dumps.backtest.pipelines.catboost_classifier.model import (
    CatboostClassifierModel,
)
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.backtest.pipelines.catboost_samples import add_catboost_pools
from pumps_and_dumps.backtest.metrics import calculate_topk_percent_auc
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample, Dataset
from pumps_and_dumps.core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "random_seed": 42,
}


def suggest_classifier_params(trial: Trial) -> Dict[str, Any]:
    """Use the same search space for the classifier, SMOTE and early-stopping variants."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }


def evaluate_classifier_trial(trial: Trial, sample: Sample, base_params: Dict[str, Any]) -> float:
    """Fit and score a classifier trial using the calling pipeline's current defaults."""
    model = CatboostClassifierModel(params=base_params | suggest_classifier_params(trial))
    model.train(sample=sample)

    tuning: Dataset = sample.get_dataset(ds_type=DatasetType.TUNING)
    return calculate_topk_percent_auc(model=model, dataset=tuning)


def _objective(trial: Trial, sample: Sample) -> float:
    return evaluate_classifier_trial(trial, sample, _BASE_PARAMS)


class CatboostClassifierPipeline(BasePipeline):
    def create_sample_from_datasets(
        self, datasets: Dict[DatasetType, pd.DataFrame]
    ) -> Sample:
        sample = super().create_sample_from_datasets(datasets)
        return add_catboost_pools(sample)

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Running <optimize_parameters> for CatboostClassifierPipeline")
        sample: Sample = self.create_sample()
        study = create_study(
            "CatboostClassifierPipelineStudy",
            start_new=True,
            storage=self.study_storage,
        )
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostClassifierModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            model_params = self.get_model_params(
                base_params=_BASE_PARAMS, study_name="CatboostClassifierPipelineStudy"
            )

        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model


def main():
    configure_logging()
    pipe = CatboostClassifierPipeline()
    pipe.optimize_parameters(n_trials=100)


if __name__ == "__main__":
    main()
