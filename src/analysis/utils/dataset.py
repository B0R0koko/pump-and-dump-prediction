from enum import Enum, auto
from typing import Optional, Dict

import pandas as pd

from analysis.utils.columns import COL_PUMP_HASH
from analysis.utils.feature_set import FeatureSet
from core.pump_event import PumpEvent


class DatasetType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class Dataset:

    def __init__(
            self,
            data: pd.DataFrame,
            feature_set: FeatureSet,
            ds_type: DatasetType,
    ):
        self._data: pd.DataFrame = data
        self.ds_type: DatasetType = ds_type
        self.feature_set: FeatureSet = feature_set

    def all_data(self, deepcopy: bool = True) -> pd.DataFrame:
        return self._data if not deepcopy else self._data.copy()

    def get_data(self) -> pd.DataFrame:
        return self._data[self.feature_set.regressors]

    def get_label(self) -> pd.Series:
        return self._data[self.feature_set.target]

    def get_eval_data(self) -> pd.DataFrame:
        assert self.feature_set.eval_fields is not None, "Data contains no evaluation fields"
        return self._data[self.feature_set.eval_fields]

    def get_cross_section(self, pump: PumpEvent, pump_col: str = COL_PUMP_HASH) -> "Dataset":
        cross_section: pd.DataFrame = self._data[self._data[pump_col] == pump.as_pump_hash()]
        return Dataset(data=cross_section, feature_set=self.feature_set, ds_type=self.ds_type)


class Sample:

    def __init__(self, datasets: Dict[DatasetType, Dataset]):
        self._datasets: Dict[DatasetType, Dataset] = datasets

    def get_dataset(self, ds_type: DatasetType) -> Dataset:
        assert ds_type in self._datasets
        return self._datasets[ds_type]

    def get_data(self, ds_type: DatasetType) -> pd.DataFrame:
        assert ds_type in self._datasets
        return self._datasets[ds_type].get_data()

    def get_label(self, ds_type: DatasetType) -> pd.Series:
        assert ds_type in self._datasets
        return self._datasets[ds_type].get_label()

    def get_eval_data(self, ds_type: DatasetType) -> Optional[pd.DataFrame]:
        assert ds_type in self._datasets
        return self._datasets[ds_type].get_eval_data()
