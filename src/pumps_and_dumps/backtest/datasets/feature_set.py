from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import COL_IS_PUMPED
from pumps_and_dumps.features.schema import regressor_columns


@dataclass
class FeatureSet:
    numeric_features: List[str]
    target: str
    categorical_features: Optional[List[str]] = None
    eval_fields: Optional[List[str]] = None

    def check_against(self, df: pd.DataFrame) -> None:
        missing = set(self.regressors + [self.target]) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    @property
    def regressors(self) -> List[str]:
        return self.numeric_features + (self.categorical_features or [])

    @property
    def all_columns(self) -> List[str]:
        return self.numeric_features + (self.categorical_features or []) + (self.eval_fields or [])

    @classmethod
    def auto(cls) -> "FeatureSet":
        """Select the ordered catalog of the original 81 predictors.

        Preserve a stable order across processes: unordered sets can change
        CatBoost's split tie-breaking. Outcome labels never enter this catalog.
        """
        return cls(
            numeric_features=regressor_columns(),
            target=COL_IS_PUMPED,
            categorical_features=None,
        )

    @classmethod
    def empty(cls) -> "FeatureSet":
        return cls(numeric_features=[], target=COL_IS_PUMPED, categorical_features=None)
