from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd

from analysis.pipelines.BaseModel import ImplementsRank
from analysis.utils.columns import COL_PROBAS_PRED, COL_PUMP_HASH, COL_IS_PUMPED
from analysis.utils.sample import Dataset
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    f1_score,
    balanced_accuracy_score,
)


def calculate_topk(
    model: ImplementsRank, dataset: Dataset, bins: Iterable[float]
) -> pd.Series:
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
        df_cross_section = df_cross_section.sort_values(
            by=COL_PROBAS_PRED, ascending=False
        ).reset_index(drop=True)
        for K in bins:
            contains_pump: bool = df_cross_section.iloc[:K][COL_IS_PUMPED].any()
            count_by_bins[K] = count_by_bins.get(K, 0) + contains_pump

    num_pumped: int = _df[_df[COL_IS_PUMPED] == True].shape[0]

    counts = np.array(list(count_by_bins.values()))

    return pd.Series(data=counts / num_pumped, index=bins)


def calculate_topk_percent(
    model: ImplementsRank, dataset: Dataset, bins: Iterable[float]
) -> pd.Series:
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
        df_cross_section = df_cross_section.sort_values(
            by=COL_PROBAS_PRED, ascending=False
        )
        n_rows = len(df_cross_section)

        for pct_bin in bins:
            k: int = max(1, int(np.ceil(n_rows * pct_bin)))
            contains_pump: bool = df_cross_section.iloc[:k][COL_IS_PUMPED].any()
            count_by_bins[pct_bin] = count_by_bins.get(pct_bin, 0) + contains_pump

    num_pumped: int = _df[_df[COL_IS_PUMPED] == True].shape[0]
    counts = np.array(list(count_by_bins.values()))

    return pd.Series(data=counts / num_pumped, index=bins)


def calculate_topk_percent_auc(model: ImplementsRank, dataset: Dataset) -> float:
    """
    :return: If we iterate over all percentages from (0, 1) and compute TOPK% accuracy for each, we can measure overall
    performance using AUC approach
    """
    topk_percentages: pd.Series = calculate_topk_percent(
        model=model, dataset=dataset, bins=np.arange(0, 1.01, 0.005)
    )
    return auc(x=topk_percentages.index, y=topk_percentages.values)


def _with_scores(model: ImplementsRank, dataset: Dataset) -> pd.DataFrame:
    scores: np.ndarray = model.rank(dataset=dataset)
    # Ensure positional numpy indexing remains valid even when input dataframe has
    # non-contiguous or non-zero-based index values.
    df_scored: pd.DataFrame = dataset.all_data().copy().reset_index(drop=True)
    df_scored[COL_PROBAS_PRED] = scores
    return df_scored


def _predict_labels(
    df_scored: pd.DataFrame,
    decision_rule: Literal["top1_per_cross_section", "threshold"],
    threshold: float,
) -> np.ndarray:
    if decision_rule == "top1_per_cross_section":
        pred_labels = np.zeros(df_scored.shape[0], dtype=int)
        top_indices: np.ndarray = (
            df_scored.groupby(COL_PUMP_HASH, sort=False)[COL_PROBAS_PRED]
            .idxmax()
            .to_numpy(dtype=int)
        )
        pred_labels[top_indices] = 1
        return pred_labels

    if decision_rule == "threshold":
        return (df_scored[COL_PROBAS_PRED].to_numpy() >= threshold).astype(int)

    raise ValueError(f"Unknown decision_rule={decision_rule}")


def calculate_f1(
    model: ImplementsRank,
    dataset: Dataset,
    decision_rule: Literal[
        "top1_per_cross_section", "threshold"
    ] = "top1_per_cross_section",
    threshold: float = 0.5,
) -> float:
    df_scored: pd.DataFrame = _with_scores(model=model, dataset=dataset)
    y_true: np.ndarray = df_scored[COL_IS_PUMPED].to_numpy(dtype=int)
    y_pred: np.ndarray = _predict_labels(
        df_scored=df_scored, decision_rule=decision_rule, threshold=threshold
    )
    return float(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0))


def calculate_balanced_accuracy(
    model: ImplementsRank,
    dataset: Dataset,
    decision_rule: Literal[
        "top1_per_cross_section", "threshold"
    ] = "top1_per_cross_section",
    threshold: float = 0.5,
) -> float:
    df_scored: pd.DataFrame = _with_scores(model=model, dataset=dataset)
    y_true: np.ndarray = df_scored[COL_IS_PUMPED].to_numpy(dtype=int)
    y_pred: np.ndarray = _predict_labels(
        df_scored=df_scored, decision_rule=decision_rule, threshold=threshold
    )
    return float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))


def calculate_pr_auc(model: ImplementsRank, dataset: Dataset) -> float:
    df_scored: pd.DataFrame = _with_scores(model=model, dataset=dataset)
    y_true: np.ndarray = df_scored[COL_IS_PUMPED].to_numpy(dtype=int)
    y_score: np.ndarray = df_scored[COL_PROBAS_PRED].to_numpy(dtype=float)
    precision: np.ndarray
    recall: np.ndarray
    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_score)
    return float(auc(x=recall, y=precision))
