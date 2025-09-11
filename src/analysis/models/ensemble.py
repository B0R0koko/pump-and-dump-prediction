from functools import partial
from typing import Dict, Any, Optional

import numpy as np
import optuna
import pandas as pd
from optuna import Trial, Study
from sklearn.ensemble import RandomForestClassifier

from analysis.models.BaseModel import BaseModelTrait, ImplementsRank
from analysis.utils.dataset import Sample, DatasetType, Dataset
from analysis.utils.metrics import calculate_topk_percent_auc


def _objective(trial: Trial, sample: Sample) -> float:
    """Define objective function for searching optimal hyperparameters for RandomForestClassifier."""
    params: Dict[str, Any] = {
        "criterion": "gini",
        "class_weight": {0: 1, 1: trial.suggest_float("class_weight", 10, 300)},
        "max_features": trial.suggest_float("max_features", 0.5, 1),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1),
        "n_jobs": -1,  # use all 24 cpu cores
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
    }
    val: Dataset = sample.get_dataset(DatasetType.VALIDATION)

    model = RandomForestModel(params=params)
    model.train(sample=sample)

    return calculate_topk_percent_auc(model=model, dataset=val)


class RandomForestModel(BaseModelTrait, ImplementsRank):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = params
        self._model: Optional[RandomForestClassifier] = None
        self._study: Optional[Study] = None

    def model(self) -> RandomForestClassifier:
        assert self._model is not None, "Model must be fitted first"
        return self._model

    def train(self, sample: Sample) -> "RandomForestModel":
        self._model = RandomForestClassifier(**self.params)
        self._model.fit(
            X=sample.get_data(ds_type=DatasetType.TRAIN),
            y=sample.get_label(ds_type=DatasetType.TRAIN),
        )
        return self

    def predict(self, dataset: Dataset, *args, **kwargs) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(X=dataset.get_data())

    def predict_proba(self, dataset: Dataset, *args, **kwargs) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.get_data())

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
        return pd.Series(self._model.predict_proba(X=dataset.get_data())[:, 1])
