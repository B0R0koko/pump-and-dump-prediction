from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from pumps_and_dumps.backtest.pipelines.base_model import BaseModel, require_fitted
from pumps_and_dumps.backtest.datasets.sample import Sample, DatasetType, Dataset


class CatboostRankerModel(BaseModel):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = dict(params)
        self._model: Optional[CatBoostRanker] = None

    def train(self, sample: Sample) -> "CatboostRankerModel":
        self._model = CatBoostRanker(**self.params)
        ptrain: Pool = sample.get_pool(DatasetType.TRAIN)
        ptuning: Pool = sample.get_pool(DatasetType.TUNING)
        self._model.fit(
            X=ptrain, eval_set=ptuning, early_stopping_rounds=50, use_best_model=True
        )
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        return require_fitted(self._model).predict(X=dataset.get_pool())

    def rank(self, dataset: Dataset) -> np.ndarray:
        return self.predict(dataset)

    def get_feature_importances(self) -> pd.Series:
        model = require_fitted(self._model)
        return pd.Series(
            model.feature_importances_,
            index=model.feature_names_,
            name="feature_importance",
        )
