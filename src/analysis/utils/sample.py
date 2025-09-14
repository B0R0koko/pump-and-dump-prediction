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
        self._pool: Optional[Pool] = None

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

    def to_pool(self) -> Pool:
        self._pool = Pool(
            data=self.get_data(),
            label=self.get_label(),
            cat_features=self.feature_set.categorical_features,
        )
        return self._pool

    def as_pool(self) -> Pool:
        return self._pool

    def get_pumps(self) -> List[PumpEvent]:
        unique_pump_hashes = self._data[COL_PUMP_HASH].unique()
        return [PumpEvent.from_pump_hash(pump_hash=pump_hash) for pump_hash in unique_pump_hashes]


class Sample:

    def __init__(self, datasets: Dict[DatasetType, Dataset]):
        self._datasets: Dict[DatasetType, Dataset] = datasets
        self._pools: Optional[Dict[DatasetType, Pool]] = None

    @classmethod
    def split_by_time(
            cls,
            df: pd.DataFrame,
            time_bins: List[datetime],
            names: List[DatasetType],
            time_col: str,
            feature_set: FeatureSet,
    ) -> "Sample":
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

        return cls.from_pandas(datasets=datasets, feature_set=feature_set)

    @classmethod
    def from_pandas(cls, datasets: Dict[DatasetType, pd.DataFrame], feature_set: FeatureSet) -> "Sample":
        """Use this method if you have already split up data into train/val/test sets"""
        return cls(
            datasets={
                ds_type: Dataset(data=dataset, feature_set=feature_set, ds_type=ds_type)
                for ds_type, dataset in datasets.items()
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

    def init_pools(self) -> None:
        self._pools = {
            ds_type: dataset.to_pool() for ds_type, dataset in self._datasets.items()
        }

    def get_pool(self, ds_type: DatasetType) -> Pool:
        assert self._pools is not None, "Call init_pools before calling get_pool"
        assert ds_type in self._pools
        return self._pools[ds_type]  # type:ignore
