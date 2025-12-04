import logging
from functools import partial
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from catboost import Pool
from optuna import Study, Trial
from overrides import overrides
from sklearn.metrics import auc

from analysis.pipelines.BaseModel import BaseModel
from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.pipelines.study import create_study
from analysis.utils.columns import COL_PUMP_HASH, COL_PROBAS_PRED, COL_IS_PUMPED
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent_auc, calculate_topk_percent
from analysis.utils.sample import DatasetType, Sample, Dataset
from core.paths import SQLITE_URL
from core.utils import configure_logging

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "Logloss",
    "sampling_frequency": "PerTree",
    "num_boost_round": 1000,
    "auto_class_weights": "Balanced",
    "verbose": 10
}


def _compute_topk_percent_auc(probas_pred: np.ndarray, df_val: pd.DataFrame) -> float:
    df_val[COL_PROBAS_PRED] = probas_pred

    bins: np.ndarray = np.arange(0, 1, 0.01)
    n_bins = bins.size

    counts = np.zeros(n_bins, dtype=np.int64)

    num_pumped: int = int(df_val[COL_IS_PUMPED].sum())
    if num_pumped == 0:
        return 0.0

    for _, df_cross_section in df_val.groupby(COL_PUMP_HASH, sort=False):
        df_cross_section = df_cross_section.sort_values(by=COL_PROBAS_PRED, ascending=False)

        is_pumped = df_cross_section[COL_IS_PUMPED].to_numpy(dtype=bool)
        n_rows = is_pumped.size
        if n_rows == 0:
            continue

        cum_any = np.logical_or.accumulate(is_pumped)
        ks = np.maximum(1, np.ceil(n_rows * bins).astype(int)) - 1

        contains = cum_any[ks]
        counts += contains.astype(int)

    return auc(x=bins, y=counts / num_pumped)


class TOPKPAUCMetric:
    """
    Custom CatBoost evaluation metric that computes the Top-K Percent AUC score.

    This metric measures how well the model ranks "pumped" samples (positive class)
    within each pump cross-section. For each pump (identified by `COL_PUMP_HASH`),
    the method:
      - Sorts predictions in descending order by predicted probability (`COL_PROBAS_PRED`).
      - Iterates over percentile bins (0%, 1%, 2%, ..., 99%) to determine how often
        a "pumped" sample appears within the top-K fraction of predictions.
      - Computes the normalized AUC (area under the curve) of that cumulative detection
        curve to produce a single scalar value.

    The metric is designed such that **higher is better** (is_max_optimal → True).

    Notes:
    ------
    - Compatible with CatBoost's Python API for custom metrics.
    - CatBoost expects the `evaluate` method to return a tuple: (metric_value, weight_sum).
    - Logs the metric value every 10 evaluation iterations for monitoring.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset (used when evaluate() is called on training predictions).
    df_val : pd.DataFrame
        Validation dataset (used when evaluate() is called on validation predictions).
    """

    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        self.df_train: pd.DataFrame = df_train
        self.df_val: pd.DataFrame = df_val

    def is_max_optimal(self):
        return True  # greater is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        probas_pred: np.ndarray = approxes[0]
        metric: float = _compute_topk_percent_auc(
            probas_pred=probas_pred,
            df_val=self.df_val if len(probas_pred) == len(self.df_val) else self.df_train,
        )
        return metric, 1

    def get_final_error(self, error, weight):
        return error


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }
    df_train: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.TRAIN).all_data()
    df_val: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.VALIDATION).all_data()
    # Add custom evaluation metric that maximizes TOPKAUC
    base_params: Dict[str, Any] = _BASE_PARAMS | {"eval_metric": TOPKPAUCMetric(df_train=df_train, df_val=df_val)}

    model: CatboostClassifierModel = CatboostClassifierModel(params=base_params | tuned_params)
    model.train(sample=sample)

    val: Dataset = sample.get_dataset(ds_type=DatasetType.VALIDATION)
    topkauc: float = calculate_topk_percent_auc(model=model, dataset=val)
    return topkauc


class CatboostClassifierTOPKAUCPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        model_params: Dict[str, Any] = base_params | study.best_params
        return model_params

    def create_sample(self) -> Sample:
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

    def optimize_parameters(self) -> Study:
        logging.info("Running <optimize_parameters> for CatboostClassifierTOPKAUCPipeline")
        sample: Sample = self.create_sample()
        study: Study = create_study(study_name="CatboostClassifierTOPKAUCPipeline", start_new=False)
        study.optimize(partial(_objective, sample=sample), n_trials=100)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostClassifierModel:
        df_train: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.TRAIN).all_data()
        df_val: pd.DataFrame = sample.get_dataset(ds_type=DatasetType.VALIDATION).all_data()
        # Add custom evaluation metric that maximizes TOPKAUC
        model_params: Dict[str, Any] = _BASE_PARAMS | {"eval_metric": TOPKPAUCMetric(df_train=df_train, df_val=df_val)}
        if tuned:
            model_params = self.get_model_params(
                base_params=model_params, study_name="CatboostClassifierTOPKAUCPipeline"
            )

        model: CatboostClassifierModel = CatboostClassifierModel(params=model_params)
        model.train(sample=sample)
        return model

    def build_model(self, tuned: bool = True) -> BaseModel:
        logging.info("Running <build_model> for CatboostClassifierTOPKAUCPipeline")
        sample: Sample = self.create_sample()
        model: CatboostClassifierModel = self.train(sample=sample, tuned=tuned)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2]
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)

        return model


def main():
    configure_logging()
    pipe = CatboostClassifierTOPKAUCPipeline()
    pipe.optimize_parameters()


if __name__ == "__main__":
    main()
