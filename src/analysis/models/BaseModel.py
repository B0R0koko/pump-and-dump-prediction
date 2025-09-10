from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd


class BaseModelTrait(Protocol):

    def predict(self, *args, **kwargs):
        ...

    def predict_proba(self, *args, **kwargs):
        ...


class ImplementsRank(ABC):

    @abstractmethod
    def rank(self, X: pd.DataFrame) -> pd.Series:
        """
        Given data as pd.DataFrame returns Series with corresponding ranking or any values that can be sorted into ranking
        """
        ...
