from functools import partial
from typing import Dict, Any, Optional

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from optuna import Trial, Study
from sklearn.ensemble import RandomForestClassifier

from analysis.models.BaseModel import BaseModelTrait, ImplementsRank
from analysis.utils.dataset import Sample, DatasetType, Dataset
from analysis.utils.metrics import calculate_topk_percent_auc


def _objective(trial: Trial, sample: Sample) -> float:
    """Define objective function for searching optimal hyperparameters for RandomForestClassifier."""
    params: Dict[str, Any] = {
        "objective": "Logloss",
        "num_boost_round": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "auto_class_weights": "Balanced",
    }
    val: Dataset = sample.get_dataset(DatasetType.VALIDATION)

    model: CatboostClassifierModel = CatboostClassifierModel(params=params)
    model.train(sample=sample)

    return calculate_topk_percent_auc(model=model, dataset=val)


class CatboostClassifierModel(BaseModelTrait, ImplementsRank):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = params
        self._model: Optional[RandomForestClassifier] = None
        self._study: Optional[Study] = None

    def model(self) -> RandomForestClassifier:
        assert self._model is not None, "Model must be fitted first"
        return self._model

    def train(self, sample: Sample) -> "CatboostClassifierModel":
        self._model = CatBoostClassifier(**self.params, verbose=False)
        ptrain: Pool = sample.get_pool(DatasetType.TRAIN)
        pval: Pool = sample.get_pool(DatasetType.VALIDATION)
        self._model.fit(X=ptrain, eval_set=pval, early_stopping_rounds=50, use_best_model=True)
        return self

    def predict(self, dataset: Dataset, *args, **kwargs) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(X=dataset.as_pool())

    def predict_proba(self, dataset: Dataset, *args, **kwargs) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.as_pool())

    @staticmethod
    def optimize_hyperparameters(sample: Sample, n_trials: int = 5) -> Study:
        # run optuna study to maximize top_k over hyperparams
        study = optuna.create_study(direction="maximize")
        study.optimize(
            partial(_objective, sample=sample),
            n_trials=n_trials
        )
        return study

    def rank(self, dataset: Dataset) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return pd.Series(self._model.predict_proba(X=dataset.as_pool())[:, 1])
