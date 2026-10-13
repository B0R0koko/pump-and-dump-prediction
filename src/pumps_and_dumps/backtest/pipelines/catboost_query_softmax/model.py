"""Raw event logits and event-normalized loss for pump identification."""

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import COL_IS_PUMPED, COL_PUMP_HASH, COL_PUMP_ID
from pumps_and_dumps.backtest.datasets.sample import Dataset
from pumps_and_dumps.backtest.pipelines.base_model import ImplementsRank, require_fitted
from pumps_and_dumps.backtest.pipelines.catboost_ranker.model import CatboostRankerModel


def event_group_column(data: pd.DataFrame) -> str:
    for column in (COL_PUMP_HASH, COL_PUMP_ID):
        if column in data:
            return column
    raise ValueError("QuerySoftMax requires pump_hash or pump_id event groups")


def validate_event_targets(data: pd.DataFrame) -> str:
    """Require one pumped asset, rather than future-return labels, in each event."""
    group_column = event_group_column(data)
    if data.empty:
        raise ValueError("QuerySoftMax requires a nonempty dataset")
    if data[group_column].isna().any():
        raise ValueError("QuerySoftMax event groups must not be missing")
    if COL_IS_PUMPED not in data or not data[COL_IS_PUMPED].isin([0, 1]).all():
        raise ValueError("QuerySoftMax requires binary is_pumped labels (0 or 1)")
    positives = data.groupby(group_column, sort=False)[COL_IS_PUMPED].sum()
    if not positives.eq(1).all():
        raise ValueError("QuerySoftMax requires exactly one pumped asset per event")
    return group_column


def query_softmax_loss(model: ImplementsRank, dataset: Dataset) -> float:
    """Mean event negative log likelihood, with one unit of weight per pump."""
    data = dataset.all_data()
    group_column = validate_event_targets(data)
    logits = np.asarray(model.rank(dataset), dtype=float)
    if logits.shape != (len(data),) or not np.isfinite(logits).all():
        raise ValueError("QuerySoftMax requires one finite logit per dataset row")
    labels = data[COL_IS_PUMPED].to_numpy(dtype=bool)
    losses = []
    for positions in data.groupby(group_column, sort=False).indices.values():
        event_logits = logits[positions]
        centered = event_logits - event_logits.max()
        loss = np.log(np.exp(centered).sum()) - centered[labels[positions]][0]
        losses.append(loss)
    return float(np.mean(losses))


class CatboostQuerySoftmaxModel(CatboostRankerModel):
    """Optimize the probability of the true pumped asset within its event group.

    Raw scores are logits for an event softmax, not independent binary
    probabilities. The portfolio sampler applies that softmax at temperature 1.
    """

    ranking_score_kind = "logit"

    def predict(self, dataset: Dataset) -> np.ndarray:
        # CatBoostRanker.predict always returns RawFormulaVal; unlike the
        # classifier API it does not accept a prediction_type argument.
        return np.asarray(require_fitted(self._model).predict(X=dataset.get_pool()))

    def get_feature_importances(self) -> pd.Series:
        # Ranking losses do not populate feature_importances_ automatically.
        # PredictionValuesChange uses fitted leaf weights without evaluation data.
        model = require_fitted(self._model)
        return pd.Series(
            model.get_feature_importance(type="PredictionValuesChange"),
            index=model.feature_names_,
            name="feature_importance",
        )
