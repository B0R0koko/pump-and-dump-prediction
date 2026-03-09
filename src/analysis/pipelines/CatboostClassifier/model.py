from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from analysis.pipelines.BaseModel import BaseModel
from analysis.utils.sample import Sample, DatasetType, Dataset


class CatboostClassifierModel(BaseModel):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = params
        self._model: Optional[CatBoostClassifier] = None

    def train(self, sample: Sample) -> "CatboostClassifierModel":
        self._model = CatBoostClassifier(**self.params)
        ptrain: Pool = sample.get_pool(DatasetType.TRAIN)
        pval: Pool = sample.get_pool(DatasetType.VALIDATION)
        self._model.fit(
            X=ptrain, eval_set=pval, early_stopping_rounds=50, use_best_model=True
        )
        return self

    def predict(self, dataset: Dataset) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(data=dataset.get_pool())

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.get_data())

    def rank(self, dataset: Dataset) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.get_pool())[:, 1]
