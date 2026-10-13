"""Train a separate pump classifier with a within-event softmax objective."""

import argparse
import logging
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool
from optuna import Study, Trial

from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_IS_PUMPED
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample
from pumps_and_dumps.backtest.pipelines.base_pipeline import BasePipeline
from pumps_and_dumps.backtest.pipelines.catboost_query_softmax.model import (
    CatboostQuerySoftmaxModel,
    query_softmax_loss,
    validate_event_targets,
)
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.core.utils import configure_logging

STUDY_NAME = "CatboostQuerySoftmaxPipelineStudy"
_BASE_PARAMS: dict[str, Any] = {
    "objective": "QuerySoftMax:beta=1",
    "eval_metric": "QuerySoftMax:beta=1",
    "iterations": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "border_count": 255,
    "verbose": False,
    "random_seed": 42,
}


def _group_event_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    group_column = validate_event_targets(data)
    if COL_CURRENCY_PAIR not in data or data[COL_CURRENCY_PAIR].isna().any():
        raise ValueError("QuerySoftMax requires a currency_pair for every asset")
    if data.duplicated([group_column, COL_CURRENCY_PAIR]).any():
        raise ValueError("QuerySoftMax requires each asset to occur once per event")
    # CatBoost needs contiguous groups. Reorder the entire frame so predictions,
    # labels, and evaluation metadata retain exactly the same row alignment.
    group_codes, _ = pd.factorize(data[group_column], sort=False)
    positions = np.argsort(group_codes, kind="stable")
    return data.iloc[positions].copy().reset_index(drop=True), group_column


def _objective(trial: Trial, sample: Sample) -> float:
    tuned_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
    }
    model = CatboostQuerySoftmaxModel(params=_BASE_PARAMS | tuned_params)
    model.train(sample)
    return query_softmax_loss(model, sample.get_dataset(DatasetType.TUNING))


class CatboostQuerySoftmaxPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_set = replace(self.feature_set, target=COL_IS_PUMPED)

    def create_sample_from_datasets(self, datasets: dict[DatasetType, pd.DataFrame]) -> Sample:
        grouped = {ds_type: _group_event_rows(data) for ds_type, data in datasets.items()}
        sample = super().create_sample_from_datasets({ds_type: data for ds_type, (data, _) in grouped.items()})
        for ds_type, dataset in sample.iter_datasets():
            group_column = grouped[ds_type][1]
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features,
                    group_id=dataset.all_data()[group_column],
                )
            )
        return sample

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Optimizing event log loss for %s", type(self).__name__)
        sample = self.create_sample()
        study = create_study(STUDY_NAME, start_new=True, storage=self.study_storage, direction="minimize")
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study

    def train(self, sample: Sample, tuned: bool = True) -> CatboostQuerySoftmaxModel:
        params = dict(_BASE_PARAMS)
        if tuned:
            try:
                params = self.get_model_params(base_params=params, study_name=STUDY_NAME)
            except KeyError as exc:
                raise RuntimeError(
                    f"No {STUDY_NAME} study exists. Run optimize_parameters() first " "or train with tuned=False."
                ) from exc
        model = CatboostQuerySoftmaxModel(params=params)
        model.train(sample)
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    configure_logging()
    CatboostQuerySoftmaxPipeline().optimize_parameters(n_trials=args.trials)


if __name__ == "__main__":
    main()
