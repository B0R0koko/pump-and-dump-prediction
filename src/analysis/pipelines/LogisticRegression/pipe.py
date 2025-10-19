import logging
from functools import partial
from typing import Dict, Any, List

import optuna
import pandas as pd
from optuna import Trial, Study
from overrides import overrides

from analysis.pipelines.BaseModel import BaseModel
from analysis.pipelines.BasePipeline import BasePipeline, cross_section_standardisation, \
    fillna_with_median_by_cross_section, remove_failed_pump_cross_sections, add_col_pump_id
from analysis.pipelines.LogisticRegression.model import LogisticRegressionModel
from analysis.pipelines.study import create_study
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent, calculate_topk_percent_auc
from analysis.utils.sample import DatasetType, Sample, Dataset
from core.feature_type import FeatureType
from core.paths import SQLITE_URL
from core.utils import configure_logging
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS

_BASE_PARAMS: Dict[str, Any] = {
    "penalty": "l1",
    "max_iter": 500,
    "verbose": False,
    "solver": "liblinear",
}


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "class_weight": {0: 1, 1: trial.suggest_float("class_weight", 10, 300)},
        "C": 1 / trial.suggest_float("lambda", 10, 1000),  # C = 1/lambda
    }

    model: LogisticRegressionModel = LogisticRegressionModel(params=_BASE_PARAMS | tuned_params)
    model.train(sample=sample)

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkauc


class LogisticRegressionPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here"""
        df = add_col_pump_id(df=df)
        df = remove_failed_pump_cross_sections(df=df)
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df = fillna_with_median_by_cross_section(df=df, feature_set=self.feature_set)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        return df_scaled

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        # Change some parameters to the way LogisticRegression expects them to be
        tuned_params: Dict[str, Any] = study.best_params
        lambd: float = tuned_params.pop("lambda")
        class_weight: float = tuned_params.pop("class_weight")
        model_params: Dict[str, Any] = base_params | tuned_params
        model_params["class_weight"] = {0: 1, 1: class_weight}
        model_params["C"] = 1 / lambd

        return model_params

    def create_sample(self) -> Sample:
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        return sample

    def optimize_parameters(self) -> Study:
        logging.info("Running <optimize_parameters> for LogisticRegressionPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="LogisticRegressionPipelineStudy")
        study.optimize(partial(_objective, sample=sample), n_trials=20)
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

    def build_model(self, tuned: bool = True) -> BaseModel:
        logging.info("Running <build_model> for LogisticRegressionPipeline")
        sample: Sample = self.create_sample()
        model: LogisticRegressionModel = self.train(sample=sample, tuned=tuned)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2]
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)
        return model


if __name__ == "__main__":
    configure_logging()
    pipeline = LogisticRegressionPipeline()
    pipeline.optimize_parameters()
