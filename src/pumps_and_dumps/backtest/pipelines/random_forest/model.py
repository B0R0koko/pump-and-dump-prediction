import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pumps_and_dumps.backtest.pipelines.base_model import BaseModel, require_fitted
from pumps_and_dumps.backtest.datasets.sample import Dataset, Sample, DatasetType


class RandomForestModel(BaseModel):

    def __init__(self, params: Dict[str, Any]):
        self.params: Dict[str, Any] = dict(params)
        self._model: Optional[RandomForestClassifier] = None

    def train(self, sample: Sample) -> "RandomForestModel":
        logging.info("Training model")
        self._model = RandomForestClassifier(**self.params)
        self._model.fit(
            X=sample.get_data(ds_type=DatasetType.TRAIN),
            y=sample.get_label(ds_type=DatasetType.TRAIN),
        )
        return self

    def predict(self, dataset: Dataset, *args, **kwargs) -> np.ndarray:
        return require_fitted(self._model).predict(X=dataset.get_data())

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        return require_fitted(self._model).predict_proba(X=dataset.get_data())

    def rank(self, dataset: Dataset) -> np.ndarray:
        return self.predict_proba(dataset)[:, 1]

    def get_feature_importances(self) -> pd.Series:
        model = require_fitted(self._model)
        return pd.Series(
            model.feature_importances_,
            index=model.feature_names_in_,
            name="feature_importance",
        )
