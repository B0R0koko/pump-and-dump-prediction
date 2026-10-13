"""Train and tune the event-level Top-5 margin loss."""

import argparse
import logging
from dataclasses import replace
from functools import partial

import pandas as pd
from optuna import Study, Trial

from pumps_and_dumps.backtest.datasets.columns import COL_IS_PUMPED
from pumps_and_dumps.backtest.datasets.sample import DatasetType, Sample
from pumps_and_dumps.backtest.pipelines.base_pipeline import BasePipeline
from pumps_and_dumps.backtest.pipelines.catboost_top5.loss import EventGroups
from pumps_and_dumps.backtest.pipelines.catboost_top5.model import DEFAULT_PARAMS, CatboostTop5Model
from pumps_and_dumps.backtest.pipelines.study import create_study
from pumps_and_dumps.core.utils import configure_logging

STUDY_NAME = "CatboostTop5PipelineStudy"
_BASE_PARAMS = dict(DEFAULT_PARAMS)


def _objective(trial: Trial, sample: Sample) -> float:
    params = _BASE_PARAMS | {
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }
    model = CatboostTop5Model(params).train(sample)
    trial.set_user_attr("best_iteration", model.best_iteration_)
    return model.best_score_


class CatboostTop5Pipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_set = replace(self.feature_set, target=COL_IS_PUMPED)

    def create_sample_from_datasets(self, datasets: dict[DatasetType, pd.DataFrame]) -> Sample:
        for data in datasets.values():
            EventGroups.from_frame(data)
        # The grouped objective uses positional indices, so interleaved events
        # need no sorting and predictions retain the supplied metadata order.
        return super().create_sample_from_datasets(datasets)

    def train(self, sample: Sample, tuned: bool = True) -> CatboostTop5Model:
        params = dict(_BASE_PARAMS)
        if tuned:
            try:
                params = self.get_model_params(params, STUDY_NAME)
            except KeyError as exc:
                raise RuntimeError(
                    f"No {STUDY_NAME} study exists. Run optimize_parameters() first or train with tuned=False."
                ) from exc
        return CatboostTop5Model(params).train(sample)

    def optimize_parameters(self, n_trials: int = 100) -> Study:
        logging.info("Optimizing mean event Top-5 margin loss")
        sample = self.create_sample()
        study = create_study(STUDY_NAME, storage=self.study_storage, direction="minimize")
        study.set_user_attr("objective", "equal_event_top5_hinge_v1")
        study.set_user_attr("top_k", 5)
        study.set_user_attr("base_parameters", _BASE_PARAMS)
        study.set_user_attr("regressors", self.feature_set.regressors)
        study.set_user_attr("optimization_dataset", "TUNING")
        study.set_user_attr("early_stopping_dataset", "TUNING")
        study.optimize(partial(_objective, sample=sample), n_trials=n_trials)
        return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    configure_logging()
    CatboostTop5Pipeline().optimize_parameters(n_trials=args.trials)


if __name__ == "__main__":
    main()
