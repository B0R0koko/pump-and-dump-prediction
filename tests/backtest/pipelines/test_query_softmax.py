"""QuerySoftMax trains on pump identity and preserves event/row alignment."""

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PUMP_HASH,
    COL_PUMP_ID,
)
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.pipelines.catboost_query_softmax.model import (
    CatboostQuerySoftmaxModel,
    query_softmax_loss,
)
from pumps_and_dumps.backtest.pipelines.catboost_query_softmax.pipeline import (
    CatboostQuerySoftmaxPipeline,
)


def _feature_set() -> FeatureSet:
    return FeatureSet(numeric_features=["signal"], target=COL_IS_PUMPED)


def _splits() -> dict[DatasetType, pd.DataFrame]:
    splits = {}
    for ds_type in DatasetType:
        rows = []
        for event_id in range(3):
            for asset_id in range(3):
                rows.append(
                    {
                        COL_PUMP_HASH: f"{ds_type.name}-{event_id}",
                        COL_PUMP_ID: event_id,
                        COL_CURRENCY_PAIR: f"ASSET{asset_id}-BTC",
                        COL_IS_PUMPED: asset_id == event_id,
                        "signal": float(asset_id == event_id) + 0.01 * asset_id,
                        "target_return@5MIN": float(asset_id - event_id),
                    }
                )
        splits[ds_type] = pd.DataFrame(rows).iloc[[6, 3, 0, 7, 4, 1, 8, 5, 2]].copy()
    return splits


def test_supplied_splits_group_rows_and_ignore_future_returns() -> None:
    feature_set = _feature_set()
    feature_set.target = "target_return@5MIN"
    pipeline = CatboostQuerySoftmaxPipeline(feature_set=feature_set)
    splits = _splits()
    original = {kind: frame.copy(deep=True) for kind, frame in splits.items()}
    sample = pipeline.create_sample_from_datasets(splits)
    changed_returns = {
        kind: frame.assign(**{"target_return@5MIN": -1000 * frame["target_return@5MIN"]})
        for kind, frame in splits.items()
    }
    changed = pipeline.create_sample_from_datasets(changed_returns)
    without_returns = pipeline.create_sample_from_datasets(
        {kind: frame.drop(columns="target_return@5MIN") for kind, frame in splits.items()}
    )

    assert pipeline.feature_set.target == COL_IS_PUMPED
    assert feature_set.target == "target_return@5MIN"
    for kind, dataset in sample.iter_datasets():
        data = dataset.all_data()
        event_ids = data[COL_PUMP_HASH]
        assert event_ids.ne(event_ids.shift()).sum() == event_ids.nunique()
        pd.testing.assert_frame_equal(splits[kind], original[kind])
        pd.testing.assert_frame_equal(
            data.sort_values([COL_PUMP_HASH, COL_CURRENCY_PAIR]).reset_index(drop=True),
            original[kind].sort_values([COL_PUMP_HASH, COL_CURRENCY_PAIR]).reset_index(drop=True),
        )
        np.testing.assert_array_equal(dataset.get_pool().get_label(), dataset.get_label())
        for alternative in (changed, without_returns):
            pd.testing.assert_frame_equal(dataset.get_data(), alternative.get_data(kind))
            pd.testing.assert_series_equal(dataset.get_label(), alternative.get_label(kind))


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 0], [0.25, 0.75, 0]])
def test_rejects_events_without_exactly_one_binary_pump_label(labels: list[float]) -> None:
    splits = _splits()
    train = splits[DatasetType.TRAIN]
    train[COL_IS_PUMPED] = train[COL_IS_PUMPED].astype(float)
    train.loc[train[COL_PUMP_ID] == 0, COL_IS_PUMPED] = labels

    with pytest.raises(ValueError):
        CatboostQuerySoftmaxPipeline(feature_set=_feature_set()).create_sample_from_datasets(splits)


def test_real_query_softmax_fit_uses_only_training_and_tuning_and_preserves_alignment() -> None:
    pipeline = CatboostQuerySoftmaxPipeline(feature_set=_feature_set())
    splits = {kind: data for kind, data in _splits().items() if kind in (DatasetType.TRAIN, DatasetType.TUNING)}
    sample = pipeline.create_sample_from_datasets(splits)
    model = CatboostQuerySoftmaxModel(
        params={
            "loss_function": "QuerySoftMax:beta=1",
            "iterations": 8,
            "depth": 2,
            "learning_rate": 0.2,
            "random_seed": 42,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
    ).train(sample)
    tuning = sample.get_dataset(DatasetType.TUNING)
    scores = model.rank(tuning)
    assert model.ranking_score_kind == "logit"
    assert scores.shape == (len(tuning.all_data()),)
    assert np.isfinite(scores).all()
    assert np.ptp(scores) > 0
    assert np.isfinite(query_softmax_loss(model, tuning))

    reordered = Dataset(
        data=tuning.all_data().iloc[[6, 3, 0, 7, 4, 1, 8, 5, 2]],
        feature_set=pipeline.feature_set,
        ds_type=DatasetType.TUNING,
    )
    reordered.add_pool()
    np.testing.assert_allclose(model.rank(reordered), scores[[6, 3, 0, 7, 4, 1, 8, 5, 2]])


def test_query_softmax_loss_averages_events_and_is_numerically_stable() -> None:
    data = pd.DataFrame(
        {
            COL_PUMP_HASH: ["a", "b", "a", "b", "b"],
            COL_IS_PUMPED: [1, 0, 0, 1, 0],
            "signal": [1000.0, -1000.0, 998.0, -999.0, -1001.0],
        }
    )
    dataset = Dataset(data=data, feature_set=_feature_set(), ds_type=DatasetType.VALIDATION)

    class ScoreModel:
        def rank(self, dataset: Dataset) -> np.ndarray:
            return dataset.get_data()["signal"].to_numpy()

    expected = (np.log1p(np.exp(-2)) + np.log1p(np.exp(-1) + np.exp(-2))) / 2
    assert query_softmax_loss(ScoreModel(), dataset) == pytest.approx(expected)
    data["signal"] += 1_000_000
    assert query_softmax_loss(ScoreModel(), dataset) == pytest.approx(expected)
