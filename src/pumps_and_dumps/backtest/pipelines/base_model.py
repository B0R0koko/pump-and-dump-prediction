from abc import ABC, abstractmethod
from typing import Protocol, Self, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.sample import Dataset, Sample


class BaseModelTrait(Protocol):
    def predict(self, dataset: Dataset) -> np.ndarray: ...

    def train(self, sample: Sample) -> Self: ...


@runtime_checkable
class ImplementsRank(Protocol):
    @abstractmethod
    def rank(self, dataset: Dataset) -> np.ndarray:
        """Return one ranking score per dataset row, in the same order."""
        ...


@runtime_checkable
class HasFeatureImportances(Protocol):
    def get_feature_importances(self) -> pd.Series: ...


class TrainingPipeline(Protocol):
    """The two operations needed to train an experiment."""

    def create_sample(self) -> Sample: ...

    def train(self, sample: Sample, tuned: bool = True) -> ImplementsRank: ...


class BaseModel(BaseModelTrait, ImplementsRank, ABC):
    @abstractmethod
    def predict(self, dataset: Dataset) -> np.ndarray: ...

    @abstractmethod
    def train(self, sample: Sample) -> Self: ...


Estimator = TypeVar("Estimator")


def require_fitted(model: Estimator | None) -> Estimator:
    if model is None:
        raise RuntimeError("Model must be fitted first")
    return model
