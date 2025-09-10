from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from analysis.utils.columns import COL_TARGET
from core.feature_type import FeatureType
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS


@dataclass
class FeatureSet:

    def __init__(
            self, numeric_features: List[str], target: str, categorical_features: Optional[List[str]] = None,
    ):
        self.numeric_features: List[str] = numeric_features
        self.categorical_features: Optional[List[str]] = categorical_features
        self.target: str = target

    def check_against(self, df: pd.DataFrame) -> None:
        categorical_features = self.categorical_features or []
        assert set(self.numeric_features).issubset(set(df.columns)), "Some numeric features are missing"
        assert set(categorical_features).issubset(set(df.columns)), "Some categorical features are missing"
        assert self.target in df.columns, "Target column is missing"

    @property
    def regressors(self) -> List[str]:
        return self.numeric_features + [] if self.categorical_features is None else self.categorical_features

    @classmethod
    def auto(cls) -> "FeatureSet":
        """
        Create feature set manually without dynamically inferring from the collection pipeline
        """
        feature_type: FeatureType
        numeric_features: List[str] = []

        for feature_type in list(FeatureType):
            numeric_features.extend(feature_type.col_names(offsets=REGRESSOR_OFFSETS))

        return cls(
            numeric_features=numeric_features,
            target=COL_TARGET,
            categorical_features=None
        )
