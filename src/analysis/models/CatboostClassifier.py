from functools import partial
from typing import Dict, Any, Optional

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from optuna import Trial, Study
from sklearn.ensemble import RandomForestClassifier

from analysis.models.BaseModel import BaseModelTrait, ImplementsRank
from analysis.utils.metrics import calculate_topk_percent_auc
from analysis.utils.sample import Sample, DatasetType, Dataset


def _objective(trial: Trial, params: Dict[str, Any], sample: Sample) -> float:
    """Define objective function for searching optimal hyperparameters for RandomForestClassifier."""
    optimized_params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }
    val: Dataset = sample.get_dataset(DatasetType.VALIDATION)

    model: CatboostClassifierModel = CatboostClassifierModel(params=params | optimized_params)
    model.train(sample=sample)

    return calculate_topk_percent_auc(model=model, dataset=val)


class CatboostClassifierModel(BaseModelTrait, ImplementsRank):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = params
        self._model: Optional[RandomForestClassifier] = None
        self._study: Optional[Study] = None

    def get_params(self) -> Dict[str, Any]:
        """
        If the model has been optimized using optimize_hyperparameters, then get_params will return
        the optimized hyperparameters.
        """
        model_params: Dict[str, Any] = self.params | self._study.best_params if self._study else self.params
        return model_params

    def model(self) -> RandomForestClassifier:
        assert self._model is not None, "Model must be fitted first"
        return self._model

    def train(self, sample: Sample) -> "CatboostClassifierModel":
        self._model = CatBoostClassifier(**self.get_params(), verbose=False)
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

    def optimize_hyperparameters(self, sample: Sample, n_trials: int = 5) -> None:
        # run optuna study to maximize top_k over hyperparams
        self._study = optuna.create_study(direction="maximize")
        self._study.optimize(partial(_objective, params=self.params, sample=sample), n_trials=n_trials)

    def rank(self, dataset: Dataset) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.as_pool())[:, 1]
