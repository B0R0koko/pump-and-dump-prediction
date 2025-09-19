from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from analysis.pipelines.BaseModel import BaseModel
from analysis.utils.sample import Sample, DatasetType, Dataset


class CatboostRankerModel(BaseModel):

    def __init__(self):
        self._model: Optional[CatBoostRanker] = None

    def train(self, sample: Sample) -> "CatboostRankerModel":
        self._model = CatBoostRanker(verbose=False)
        ptrain: Pool = sample.get_pool(DatasetType.TRAIN)
        pval: Pool = sample.get_pool(DatasetType.VALIDATION)
        self._model.fit(X=ptrain, eval_set=pval, early_stopping_rounds=50, use_best_model=True)
        return self

    def predict(self, dataset: Dataset) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(X=dataset.get_pool())

    def rank(self, dataset: Dataset) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(X=dataset.get_pool())
