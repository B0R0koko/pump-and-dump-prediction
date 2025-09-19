from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Iterator, Tuple

import pandas as pd
from catboost import Pool

from analysis.utils.columns import COL_PUMP_HASH
from analysis.utils.feature_set import FeatureSet
from core.pump_event import PumpEvent


class DatasetType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


def split_by_time(
        df: pd.DataFrame, time_bins: List[datetime], names: List[DatasetType], time_col: str
) -> Dict[DatasetType, pd.DataFrame]:
    """Split df: pd.DataFrame by time_col and return Sample"""
    assert len(names) - 1 == len(time_bins), "There should be one name more than time_bins"
    df = df.sort_values(by=time_col)
    # first slice: < first bin
    datasets: Dict[DatasetType, pd.DataFrame] = {names[0]: df[df[time_col] < time_bins[0]]}

    # middle slices: between bins
    for i in range(1, len(time_bins)):
        start = time_bins[i - 1]
        end = time_bins[i]
        datasets[names[i]] = df[(df[time_col] >= start) & (df[time_col] < end)]

    # last slice: >= last bin
    datasets[names[-1]] = df[df[time_col] >= time_bins[-1]]

    return datasets


class Dataset:

    def __init__(self, data: pd.DataFrame, feature_set: FeatureSet, ds_type: DatasetType):
        self._data: pd.DataFrame = data
        self.ds_type: DatasetType = ds_type
        self.feature_set: FeatureSet = feature_set
        self._pool: Optional[Pool] = None

    def all_data(self) -> pd.DataFrame:
        return self._data

    def get_data(self) -> pd.DataFrame:
        return self._data[self.feature_set.regressors]

    def get_label(self) -> pd.Series:
        return self._data[self.feature_set.target]

    def get_eval_data(self) -> pd.DataFrame:
        return self._data[self.feature_set.eval_fields]

    def add_pool(self):
        """Adds simple Pool, use set_pool if Pool has any complex logic or requires a lot of additional configuration"""
        self._pool = Pool(
            data=self.get_data(),
            label=self.get_label(),
            cat_features=self.feature_set.categorical_features,
        )

    def set_pool(self, pool: Pool) -> None:
        self._pool = pool

    def get_pool(self) -> Pool:
        assert self._pool is not None, "You must set pool first"
        return self._pool

    def get_cross_section(self, pump: PumpEvent) -> "Dataset":
        df: pd.DataFrame = self.all_data()
        df_pump = df[df[COL_PUMP_HASH] == pump.as_pump_hash()].copy().reset_index(drop=True)
        dataset: Dataset = Dataset(data=df_pump, feature_set=self.feature_set, ds_type=self.ds_type)
        dataset.add_pool()
        return dataset

    def get_pumps(self) -> List[PumpEvent]:
        return [
            PumpEvent.from_pump_hash(pump_hash=pump_hash)
            for pump_hash in self.all_data()[COL_PUMP_HASH].unique()
        ]


class Sample:

    def __init__(self, datasets: Dict[DatasetType, Dataset]):
        self._datasets: Dict[DatasetType, Dataset] = datasets
        self._pools: Optional[Dict[DatasetType, Pool]] = None

    @classmethod
    def from_pandas(cls, datasets: Dict[DatasetType, pd.DataFrame], feature_set: FeatureSet) -> "Sample":
        return cls(
            datasets={
                ds_type: Dataset(data=data, feature_set=feature_set, ds_type=ds_type)
                for ds_type, data in datasets.items()
            },
        )

    def iter_datasets(self) -> Iterator[Tuple[DatasetType, Dataset]]:
        for ds_type, dataset in self._datasets.items():
            yield ds_type, dataset

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

    def get_pool(self, ds_type: DatasetType) -> Optional[Pool]:
        assert ds_type in self._datasets
        return self._datasets[ds_type].get_pool()
