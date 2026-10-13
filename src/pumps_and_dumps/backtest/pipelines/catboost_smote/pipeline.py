import logging
from functools import partial
from typing import Dict, Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from optuna import Trial, Study
from overrides import overrides

from pumps_and_dumps.backtest.pipelines.base_pipeline import (
    BasePipeline,
    cross_section_standardisation,
    impute_training_features,
    prepare_cross_section_features,
)
from pumps_and_dumps.backtest.pipelines.catboost_classifier.model import (
    CatboostClassifierModel,
)
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.backtest.pipelines.catboost_samples import add_catboost_pools
from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PUMPED_CURRENCY_PAIR,
)
from pumps_and_dumps.backtest.metrics import calculate_topk_percent_auc
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample, Dataset
from pumps_and_dumps.core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
    "verbose": False,
    "random_seed": 42,
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }

    model: CatboostClassifierModel = CatboostClassifierModel(
        params=_BASE_PARAMS | tuned_params
    )
    model.train(sample=sample)

    tuning: Dataset = sample.get_dataset(ds_type=DatasetType.TUNING)
    topkpauc: float = calculate_topk_percent_auc(model=model, dataset=tuning)
    return topkpauc


class CatboostClassifierSMOTEPipeline(BasePipeline):
    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply only transformations available without post-pump outcomes."""
        df = prepare_cross_section_features(df)
        df[COL_IS_PUMPED] = (
            df[COL_CURRENCY_PAIR] == df[COL_PUMPED_CURRENCY_PAIR]
        )  # attach binary target
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        df_scaled = impute_training_features(
            df=df_scaled, feature_set=self.feature_set
        )
        return df_scaled

    def apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the SMOTE algorithm to the Dataset
        """
        logging.info("Applying SMOTE")
        X, y = SMOTE(random_state=42).fit_resample(
            X=df[self.feature_set.regressors], y=df[self.feature_set.target]
        )
        df_train: pd.DataFrame = X
        df_train[self.feature_set.target] = y
        return df_train

    def create_sample_from_datasets(
        self, datasets: Dict[DatasetType, pd.DataFrame]
    ) -> Sample:
        prepared = {
            ds_type: frame.copy(deep=True) for ds_type, frame in datasets.items()
        }
        prepared[DatasetType.TRAIN] = self.apply_smote(prepared[DatasetType.TRAIN])
        sample = super().create_sample_from_datasets(prepared)
        return add_catboost_pools(sample)

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info(
            "Running <optimize_parameters> for CatboostClassifierSMOTEPipeline"
        )
        sample: Sample = self.create_sample()
        study = create_study(
            "CatboostClassifierSMOTEPipelineStudy",
            start_new=True,
            storage=self.study_storage,
        )
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostClassifierModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            model_params = self.get_model_params(
                base_params=_BASE_PARAMS,
                study_name="CatboostClassifierSMOTEPipelineStudy",
            )

        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model


def main():
    configure_logging()
    pipe = CatboostClassifierSMOTEPipeline()
    pipe.optimize_parameters(n_trials=100)


if __name__ == "__main__":
    main()
