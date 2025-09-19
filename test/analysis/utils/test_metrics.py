from typing import List

import numpy as np
import pandas as pd

from analysis.pipelines.BaseModel import ImplementsRank
from analysis.utils.columns import COl_PROBAS_PRED, COL_PUMP_HASH, COL_TARGET
from analysis.utils.sample import Dataset, DatasetType
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent, calculate_topk
from core.utils import configure_logging

_test_data: pd.DataFrame = pd.DataFrame(
    data=[
        # Cross-section 1
        (0.07, "pump#1", 0),
        (0.08, "pump#1", 0),
        (0.09, "pump#1", 0),
        (0.1, "pump#1", 1),
        (0.2, "pump#1", 0),
        (0.3, "pump#1", 0),
        # Cross-section 2
        (0.4, "pump#2", 0),
        (0.5, "pump#2", 0),
        (0.6, "pump#2", 0),
        (0.7, "pump#2", 0),
        (0.8, "pump#2", 1),
        (0.9, "pump#2", 0),
    ],
    columns=[COl_PROBAS_PRED, COL_PUMP_HASH, COL_TARGET]
)

feature_set: FeatureSet = FeatureSet(
    numeric_features=[],
    target=COL_TARGET,
    categorical_features=None,
    eval_fields=[COL_PUMP_HASH],
)


class DummyModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> pd.Series:
        return _test_data[COl_PROBAS_PRED].values


def test_calculate_topk():
    """
    Given the data above
    TOP1 = 0, TOP2 = 1/2, TOP5 = 1
    """
    configure_logging()
    topk_bins: List[int] = [1, 2, 5]

    dataset: Dataset = Dataset(data=_test_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()
    topks: pd.Series = calculate_topk(model=model, dataset=dataset, bins=topk_bins)

    assert np.allclose(topks.values, [0, 0.5, 1], atol=1e-5)


def test_calculate_topk_percent():
    """
    Given the data above
    TOP10% = 10% of the data is at least 1 element => TOP10% = TOP1 = 0
    TOP20% = 20% of the data is 2 elements => TOP20% = 0.5
    TOP50% = 50% of the data is 3 elements  => TOP50% = 1
    """
    topk_bins: List[float] = [0.1, 0.2, 0.5]

    dataset: Dataset = Dataset(data=_test_data, ds_type=DatasetType.TEST, feature_set=feature_set)
    model: DummyModel = DummyModel()
    topks: pd.Series = calculate_topk_percent(model=model, dataset=dataset, bins=topk_bins)

    assert np.allclose(topks.values, [0, 0.5, 1], atol=1e-5)
