from functools import partial
from typing import Dict, Any, Optional

import numpy as np
import optuna
import pandas as pd
from optuna import Trial, Study
from sklearn.ensemble import RandomForestClassifier

from analysis.models.BaseModel import BaseModelTrait
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent_auc


def _objective(
        trial: Trial, df_train: pd.DataFrame, df_val: pd.DataFrame, feature_set: FeatureSet
) -> float:
    """Define objective function for searching optimal hyperparameters for RandomForestClassifier."""
    params: Dict[str, Any] = {
        "criterion": "gini",
        "class_weight": {0: 1, 1: trial.suggest_float("class_weight", 10, 300)},
        "max_features": trial.suggest_float("max_features", 0.5, 1),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1),
        "n_jobs": -1,  # use all 24 cpu cores
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
    }
    model = RandomForestModel(params=params)
    # Check that feature sets are the same
    feature_set.check_against(df=df_train)
    feature_set.check_against(df=df_val)

    model.train(X=df_train[feature_set.regressors], y=df_train[feature_set.target])
    probas_pred: np.ndarray = model.predict_proba(df_val[feature_set.regressors])[:, 1]

    return calculate_topk_percent_auc(df=df_val, probas_pred=probas_pred)


class RandomForestModel(BaseModelTrait):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = params
        self._model: Optional[RandomForestClassifier] = None
        self._study: Optional[Study] = None

    def model(self) -> RandomForestClassifier:
        assert self._model is not None, "Model must be fitted first"
        return self._model

    def train(self, X: pd.DataFrame, y: pd.Series):
        self._model = RandomForestClassifier(**self.params).fit(X=X, y=y)

    def predict(self, X: pd.DataFrame, *args, **kwargs) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X)

    @staticmethod
    def optimize_hyperparameters(
            df_train: pd.DataFrame, df_val: pd.DataFrame, feature_set: FeatureSet, n_trials: int = 5
    ) -> Study:
        # run optuna study to maximize top_k over hyperparams
        study = optuna.create_study(direction="maximize")
        study.optimize(
            partial(_objective, df_train=df_train, df_val=df_val, feature_set=feature_set),
            n_trials=n_trials
        )
        return study
