from typing import Dict, Iterable

import numpy as np
import pandas as pd

from analysis.models.BaseModel import ImplementsRank
from analysis.utils.columns import COL_PROBAS_PRED, COL_PUMP_HASH, COL_TARGET
from analysis.utils.sample import Dataset


def calculate_topk(model: ImplementsRank, dataset: Dataset, bins: Iterable[float]) -> pd.Series:
    """
    :param bins: bins used to calculate topk
    :return: pd.Series with topk values. Which measures the chance of predicting the actual pump given we take a portfolio
    of size K
    """
    probas_pred: np.ndarray = model.rank(dataset=dataset)
    _df: pd.DataFrame = dataset.all_data()
    _df[COL_PROBAS_PRED] = probas_pred

    count_by_bins: Dict[float, int] = {}

    for pump_hash, df_cross_section in _df.groupby(COL_PUMP_HASH):
        df_cross_section = df_cross_section.sort_values(by=COL_PROBAS_PRED, ascending=False).reset_index(drop=True)
        for K in bins:
            contains_pump: bool = df_cross_section.iloc[:K][COL_TARGET].any()
            count_by_bins[K] = count_by_bins.get(K, 0) + contains_pump

    num_pumped: int = _df[_df[COL_TARGET] == True].shape[0]

    counts = np.array(list(count_by_bins.values()))

    return pd.Series(data=counts / num_pumped, index=bins)


def calculate_topk_percent(model: ImplementsRank, dataset: Dataset, bins: Iterable[float]) -> pd.Series:
    """
    :param bins: bins used to calculate topk. K measures the share of cross-section taken as a portfolio
    :return: pd.Series with topk% values. Which measures the chance of predicting the actual pump given we take a portfolio
    of size of K% of the whole cross-section
    """
    probas_pred: np.ndarray = model.rank(dataset=dataset)
    _df: pd.DataFrame = dataset.all_data()
    _df[COL_PROBAS_PRED] = probas_pred

    count_by_bins: Dict[float, int] = {}

    for pump_hash, df_cross_section in _df.groupby(COL_PUMP_HASH):
        df_cross_section = df_cross_section.sort_values(by=COL_PROBAS_PRED, ascending=False)
        n_rows = len(df_cross_section)

        for pct_bin in bins:
            k: int = max(1, int(np.ceil(n_rows * pct_bin)))
            contains_pump: bool = df_cross_section.iloc[:k][COL_TARGET].any()
            count_by_bins[pct_bin] = count_by_bins.get(pct_bin, 0) + contains_pump

    num_pumped: int = _df[_df[COL_TARGET] == True].shape[0]
    counts = np.array(list(count_by_bins.values()))

    return pd.Series(data=counts / num_pumped, index=bins)


from sklearn.metrics import auc


def calculate_topk_percent_auc(model: ImplementsRank, dataset: Dataset) -> float:
    """
    :return: If we iterate over all percentages from (0, 1) and compute TOPK% accuracy for each, we can measure overall
    performance using AUC approach
    """
    topk_percentages: pd.Series = calculate_topk_percent(model=model, dataset=dataset, bins=np.arange(0, 1.01, 0.005))
    return auc(x=topk_percentages.index, y=topk_percentages.values)
