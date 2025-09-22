import logging
from functools import partial
from typing import Dict, List, Any

import optuna
import pandas as pd
from catboost import Pool
from imblearn.over_sampling import SMOTE
from optuna import Trial, Study
from overrides import overrides

from analysis.pipelines.BasePipeline import BasePipeline, cross_section_standardisation, fillna_with_median
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.utils.columns import *
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent, calculate_topk_percent_auc
from analysis.utils.sample import DatasetType, Sample, Dataset
from core.feature_type import FeatureType
from core.utils import configure_logging
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
    "auto_class_weights": "Balanced",
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }

    model: CatboostClassifierModel = CatboostClassifierModel(params=_BASE_PARAMS | tuned_params)
    model.train(sample=sample)

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkauc


class CatboostClassifierSMOTEPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here"""
        df[COL_IS_PUMPED] = df[COL_CURRENCY_PAIR] == df[COL_PUMPED_CURRENCY_PAIR]  # attach binary target
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        df_scaled = fillna_with_median(df=df_scaled, feature_set=self.feature_set)
        return df_scaled

    def apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the SMOTE algorithm to the Dataset
        """
        logging.info("Applying SMOTE")
        X, y = SMOTE().fit_resample(X=df[self.feature_set.regressors], y=df[self.feature_set.target])
        df_train: pd.DataFrame = X
        df_train[self.feature_set.target] = y
        return df_train

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage="sqlite:///my_study.db")
        model_params: Dict[str, Any] = base_params | study.best_params
        model_params["class_weight"] = {0: 1, 1: model_params["class_weight"]}
        return model_params

    def optimize_parameters(self):
        logging.info("Running <optimize_parameters> for CatboostRankerPipeline")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)

        study: Study = optuna.create_study(direction="maximize", study_name="CatboostClassifierSMOTEPipelineStudy")
        study.optimize(partial(_objective, sample=sample), n_trials=10)

    def build_model(self) -> None:
        logging.info("Building Random Forest Model")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()

        df_train: pd.DataFrame = self.apply_smote(df=datasets.get(DatasetType.TRAIN))
        datasets[DatasetType.TRAIN] = df_train

        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        # we also need to set_pools as Catboost uses Pool under the hood
        for ds_type, dataset in sample.iter_datasets():
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features
                )
            )

        model_params: Dict[str, Any] = self.get_model_params(
            base_params=_BASE_PARAMS, study_name="CatboostClassifierSMOTEPipelineStudy"
        )
        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.TEST),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2]
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)


def main():
    configure_logging()
    pipe = CatboostClassifierSMOTEPipeline()
    pipe.build_model()


if __name__ == "__main__":
    main()
