import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from analysis.pipelines.BaseModel import BaseModel
from analysis.utils.sample import Dataset, Sample, DatasetType


class RandomForestModel(BaseModel):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = params
        self._model: Optional[RandomForestClassifier] = None

    def train(self, sample: Sample) -> "RandomForestModel":
        logging.info("Training model")
        self._model = RandomForestClassifier(**self.params)
        self._model.fit(
            X=sample.get_data(ds_type=DatasetType.TRAIN),
            y=sample.get_label(ds_type=DatasetType.TRAIN),
        )
        return self

    def predict(self, dataset: Dataset, *args, **kwargs) -> pd.Series:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict(X=dataset.get_data())

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.get_data())

    def rank(self, dataset: Dataset) -> np.ndarray:
        assert self._model is not None, "Model must be fitted first"
        return self._model.predict_proba(X=dataset.get_data())[:, 1]
