import logging
from functools import partial
from typing import Dict, Any

import optuna
import pandas as pd
from optuna import Trial, Study
from overrides import overrides

from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.RandomForest.model import RandomForestModel
from analysis.pipelines.study import create_study
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent, calculate_topk_percent_auc
from analysis.utils.sample import DatasetType, Sample, Dataset
from core.paths import SQLITE_URL
from core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "criterion": "gini",
    "n_jobs": -1,
    "verbose": False
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

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkauc


class RandomForestPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    def create_sample(self) -> Sample:
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        return sample

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        model_params: Dict[str, Any] = base_params | study.best_params
        model_params["class_weight"] = {0: 1, 1: model_params["class_weight"]}
        return model_params

    def optimize_parameters(self):
        logging.info("Running <optimize_parameters> for RandomForestPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="RandomForestPipelineStudy")
        study.optimize(partial(_objective, sample=sample), n_trials=10)

    def train(self, sample: Sample, tuned: bool = True) -> RandomForestModel:
        model_params: Dict[str, Any] = _BASE_PARAMS
        if tuned:
            # Read optimal parameters from optuna.RDBStorage
            model_params = self.get_model_params(base_params=_BASE_PARAMS, study_name="RandomForestPipelineStudy")
        model: RandomForestModel = RandomForestModel(params=model_params)
        model.train(sample=sample)
        return model

    def build_model(self, tuned: bool = True) -> None:
        logging.info("Running <build_model> for RandomForestPipeline")
        sample: Sample = self.create_sample()
        model: RandomForestModel = self.train(sample=sample, tuned=tuned)
        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2]
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)


if __name__ == "__main__":
    configure_logging()
    pipeline = RandomForestPipeline()
    pipeline.build_model()
