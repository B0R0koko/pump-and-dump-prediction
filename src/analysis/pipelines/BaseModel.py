from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from analysis.utils.sample import Dataset


class BaseModelTrait(Protocol):

    def predict(self, *args, **kwargs):
        ...

    def train(self, *args, **kwargs):
        ...


class ImplementsRank(ABC):

    @abstractmethod
    def rank(self, dataset: Dataset) -> np.ndarray:
        """
        Given data as pd.DataFrame returns Series with corresponding ranking or any values that can be sorted into ranking
        """
        ...


class BaseModel(BaseModelTrait, ImplementsRank, ABC):
    ...
