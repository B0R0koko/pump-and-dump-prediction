import logging
from functools import partial
from typing import Dict, Any

import optuna
import pandas as pd
from catboost import Pool
from optuna import Study, Trial
from overrides import overrides

from analysis.pipelines.BaseModel import BaseModel
from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.pipelines.study import create_study
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent_auc
from analysis.utils.sample import DatasetType, Sample, Dataset
from core.paths import SQLITE_URL
from core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
    "auto_class_weights": "Balanced",
    "verbose": False
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


class CatboostClassifierPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        model_params: Dict[str, Any] = base_params | study.best_params
        model_params["class_weight"] = {0: 1, 1: model_params["class_weight"]}
        return model_params

    def _create_sample(self) -> Sample:
        # we also need to set_pools as Catboost uses Pool under the hood
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        for ds_type, dataset in sample.iter_datasets():
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features
                )
            )

        return sample

    def optimize_parameters(self):
        logging.info("Running <optimize_parameters> for CatboostRankerPipeline")
        sample: Sample = self._create_sample()
        study: Study = create_study(study_name="CatboostClassifierPipelineStudy")
        study.optimize(partial(_objective, sample=sample), n_trials=10)

    def build_model(self) -> BaseModel:
        logging.info("Building Random Forest Model")
        sample: Sample = self._create_sample()
        model_params: Dict[str, Any] = self.get_model_params(
            base_params=_BASE_PARAMS, study_name="CatboostRankerPipelineStudy"
        )
        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model


def main():
    configure_logging()
    pipe = CatboostClassifierPipeline()
    pipe.optimize_parameters()


if __name__ == "__main__":
    main()
