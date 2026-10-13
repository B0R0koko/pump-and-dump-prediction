import logging
from functools import partial
from typing import Dict, Any

import pandas as pd
from optuna import Trial, Study
from overrides import overrides

from pumps_and_dumps.backtest.pipelines.base_pipeline import (
    BasePipeline,
    cross_section_standardisation,
    impute_training_features,
    prepare_cross_section_features,
)
from pumps_and_dumps.backtest.pipelines.logistic_regression.model import (
    LogisticRegressionModel,
)
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.backtest.metrics import calculate_topk_percent_auc
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample, Dataset
from pumps_and_dumps.core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "penalty": "l1",
    "max_iter": 500,
    "verbose": False,
    "solver": "liblinear",
    "random_state": 42,
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "class_weight": {0: 1, 1: trial.suggest_float("class_weight", 10, 300)},
        "C": 1 / trial.suggest_float("lambda", 10, 1000),  # C = 1/lambda
    }

    model: LogisticRegressionModel = LogisticRegressionModel(
        params=_BASE_PARAMS | tuned_params
    )
    model.train(sample=sample)

    tuning: Dataset = sample.get_dataset(ds_type=DatasetType.TUNING)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=tuning)
    return topkpauc


class LogisticRegressionPipeline(BasePipeline):
    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply only transformations available without post-pump outcomes."""
        df = prepare_cross_section_features(df)
        df = impute_training_features(df=df, feature_set=self.feature_set)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        return df_scaled

    @overrides
    def get_model_params(
        self, base_params: Dict[str, Any], study_name: str
    ) -> Dict[str, Any]:
        model_params = super().get_model_params(base_params, study_name)
        # Change some parameters to the way LogisticRegression expects them to be
        lambd: float = model_params.pop("lambda")
        class_weight: float = model_params.pop("class_weight")
        model_params["class_weight"] = {0: 1, 1: class_weight}
        model_params["C"] = 1 / lambd

        return model_params

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Running <optimize_parameters> for LogisticRegressionPipeline")
        sample: Sample = self.create_sample()
        study = create_study(
            "LogisticRegressionPipelineStudy",
            start_new=True,
            storage=self.study_storage,
        )
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> LogisticRegressionModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            # Read optimal parameters from optuna.RDBStorage
            model_params = self.get_model_params(
                base_params=_BASE_PARAMS, study_name="LogisticRegressionPipelineStudy"
            )
        model: LogisticRegressionModel = LogisticRegressionModel(params=model_params)
        model.train(sample=sample)
        return model


if __name__ == "__main__":
    configure_logging()
    pipeline = LogisticRegressionPipeline()
    pipeline.optimize_parameters(n_trials=100)
