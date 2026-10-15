"""Fit CatBoost trees to the full-event Top-5 hinge loss gradient."""

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostRegressor, sum_models

from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType, Sample
from pumps_and_dumps.backtest.pipelines.base_model import BaseModel
from pumps_and_dumps.backtest.pipelines.catboost_top5.loss import Top5Loss

DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.3,
    "max_depth": 4,
    "border_count": 64,
    "l2_leaf_reg": 3.0,
    "colsample_bylevel": 1.0,
    "random_seed": 42,
    "margin": 1.0,
    "early_stopping_rounds": 40,
    "thread_count": -1,
}


class CatboostTop5Model(BaseModel):
    """Functional gradient boosting for an objective over complete pump events.

    CatBoost's Python objective callback receives row subsets without event
    boundaries. Instead, compute the grouped gradient here and fit one tree to
    it with squared error. The resulting ensemble optimizes the Top-5 loss;
    RMSE is only the projection of each gradient onto a regression tree.

    Scores are uncalibrated margins used only to order candidate assets.
    """


    def __init__(self, params: dict[str, Any]):
        unknown = params.keys() - DEFAULT_PARAMS.keys()
        if unknown:
            raise ValueError(f"Unknown Top5 model parameters: {sorted(unknown)}")
        self.params = DEFAULT_PARAMS | params
        self._validate_boosting_params()
        self._model: CatBoost | None = None
        self._is_fitted = False
        self.feature_names_: list[str] = []
        self.training_history_: list[dict[str, float | int]] = []
        # Zero represents the initial zero-score model, before any tree.
        self.best_iteration_ = 0
        self.best_score_ = float("inf")

    def _validate_boosting_params(self) -> None:
        for name in ("iterations", "early_stopping_rounds"):
            value = self.params[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        learning_rate = self.params["learning_rate"]
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

    def train(self, sample: Sample) -> "CatboostTop5Model":
        self._is_fitted = False
        self._model = None
        training = sample.get_dataset(DatasetType.TRAIN)
        tuning = sample.get_dataset(DatasetType.TUNING)
        train_loss = self._loss(training)
        tuning_loss = self._loss(tuning)
        train_data = training.get_data()
        tuning_data = tuning.get_data()
        self.feature_names_ = list(train_data.columns)
        if list(tuning_data.columns) != self.feature_names_:
            raise ValueError("Training and tuning feature columns must match")
        train_scores = np.zeros(len(train_data))
        tuning_scores = np.zeros(len(tuning_data))
        self._start_history(train_loss.value(train_scores), tuning_loss.value(tuning_scores))
        trees = []

        for iteration in range(1, self.params["iterations"] + 1):
            # Remove the constant 1 / event_count factor so learning_rate has
            # the same meaning when the number of training events changes.
            residual = train_loss.negative_gradient(train_scores) * len(train_loss.groups.events)
            if not np.any(residual):
                break
            tree = self._fit_tree(train_data, residual, training, iteration)
            trees.append(tree)
            train_scores += self.params["learning_rate"] * self._tree_scores(tree, train_data)
            tuning_scores += self.params["learning_rate"] * self._tree_scores(tree, tuning_data)
            self._record_iteration(iteration, train_loss.value(train_scores), tuning_loss.value(tuning_scores))
            if iteration - self.best_iteration_ >= self.params["early_stopping_rounds"]:
                break

        if self.best_iteration_:
            kept_trees = trees[: self.best_iteration_]
            # Every tree sees different residual labels, so categorical
            # statistics must remain separate to preserve the exact sum.
            self._model = sum_models(
                kept_trees,
                weights=[self.params["learning_rate"]] * len(kept_trees),
                ctr_merge_policy="KeepAllTables",
            )
        self._is_fitted = True
        return self

    def _loss(self, dataset: Dataset) -> Top5Loss:
        return Top5Loss.from_frame(dataset.all_data(), margin=self.params["margin"])

    def _fit_tree(
        self, data: pd.DataFrame, residual: np.ndarray, dataset: Dataset, iteration: int
    ) -> CatBoostRegressor:
        tree = CatBoostRegressor(
            iterations=1,
            learning_rate=1,
            loss_function="RMSE",
            depth=self.params["max_depth"],
            border_count=self.params["border_count"],
            l2_leaf_reg=self.params["l2_leaf_reg"],
            rsm=self.params["colsample_bylevel"],
            random_seed=self.params["random_seed"] + iteration - 1,
            thread_count=self.params["thread_count"],
            boost_from_average=False,
            random_strength=0,
            bootstrap_type="No",
            verbose=False,
            allow_writing_files=False,
        )
        tree.fit(data, residual, cat_features=dataset.feature_set.categorical_features)
        return tree

    @staticmethod
    def _tree_scores(tree: CatBoost, data: pd.DataFrame) -> np.ndarray:
        scores = np.asarray(tree.predict(data, prediction_type="RawFormulaVal"), dtype=float)
        if scores.shape != (len(data),) or not np.isfinite(scores).all():
            raise ValueError("Top5 tree produced nonfinite or misaligned scores")
        return scores

    def _start_history(self, training_loss: float, tuning_loss: float) -> None:
        self.best_iteration_ = 0
        self.best_score_ = tuning_loss
        self.training_history_ = []
        self._record_iteration(0, training_loss, tuning_loss)

    def _record_iteration(self, iteration: int, training_loss: float, tuning_loss: float) -> None:
        self.training_history_.append(
            {"iteration": iteration, "training_loss": training_loss, "tuning_loss": tuning_loss}
        )
        if tuning_loss < self.best_score_:
            self.best_iteration_ = iteration
            self.best_score_ = tuning_loss

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

    def predict(self, dataset: Dataset) -> np.ndarray:
        self._require_fitted()
        data = dataset.get_data()
        if list(data.columns) != self.feature_names_:
            raise ValueError("Prediction feature columns must match training features")
        if self._model is None:
            return np.zeros(len(data))
        return self._tree_scores(self._model, data)

    def rank(self, dataset: Dataset) -> np.ndarray:
        return self.predict(dataset)

    def get_feature_importances(self) -> pd.Series:
        self._require_fitted()
        importances = np.zeros(len(self.feature_names_))
        if self._model is not None:
            importances = self._model.get_feature_importance(type="PredictionValuesChange")
        return pd.Series(importances, index=self.feature_names_, name="feature_importance")
