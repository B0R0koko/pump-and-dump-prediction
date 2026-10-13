"""Top-5 training uses event labels, exact boundary losses, and held-out stopping."""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_IS_PUMPED, COL_PUMP_HASH
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.pipelines.catboost_top5 import pipeline as top5_pipeline
from pumps_and_dumps.backtest.pipelines.catboost_top5.loss import Top5Loss
from pumps_and_dumps.backtest.pipelines.catboost_top5.model import CatboostTop5Model
from pumps_and_dumps.backtest.pipelines.catboost_top5.pipeline import CatboostTop5Pipeline


def _frame(events: list[str], labels: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_PUMP_HASH: events,
            COL_CURRENCY_PAIR: [f"ASSET{index}-BTC" for index in range(len(events))],
            COL_IS_PUMPED: labels,
        },
        index=np.arange(len(events)) * 3,
    )


def test_loss_uses_only_fifth_negative_and_weights_events_equally() -> None:
    # A has hinge 3, B has hinge 2, singleton C has no selection boundary.
    # Much larger first-four negative scores must not increase the loss.
    frame = _frame(["a"] * 7 + ["b"] * 6 + ["c"], [1] + [0] * 6 + [1] + [0] * 5 + [1])
    scores = np.array([0, 10, 9, 8, 7, 2, -4, 3, 8, 7, 6, 5, 4, -9], dtype=float)
    order = [0, 7, 13, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6]
    loss = Top5Loss.from_frame(frame.iloc[order])

    assert loss.value(scores[order]) == pytest.approx(5 / 3)
    assert loss.value(scores[order] + 1000) == pytest.approx(5 / 3)
    scores[1:5] += 100
    assert loss.value(scores[order]) == pytest.approx(5 / 3)


def test_negative_gradient_matches_finite_differences() -> None:
    frame = _frame(["a"] * 7 + ["b"] * 6, [1] + [0] * 6 + [1] + [0] * 5)
    scores = np.array([0, 5, 4, 3, 2, 0.5, -2, 2, 8, 7, 6, 5, 4], dtype=float)
    loss = Top5Loss.from_frame(frame)
    epsilon = 1e-6
    numerical = np.array(
        [
            (loss.value(scores + direction) - loss.value(scores - direction)) / (2 * epsilon)
            for direction in np.eye(len(scores)) * epsilon
        ]
    )
    gradient = loss.negative_gradient(scores)

    np.testing.assert_allclose(gradient, -numerical, atol=1e-9)
    for positions in frame.groupby(COL_PUMP_HASH).indices.values():
        assert gradient[positions].sum() == pytest.approx(0)


def test_cutoff_ties_share_gradient_independently_of_row_order_and_zero_hinge_stops() -> None:
    frame = _frame(["a"] * 8, [1] + [0] * 7)
    scores = np.array([0.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, -2.0])
    loss = Top5Loss.from_frame(frame)
    expected = np.array([1, 0, 0, 0, -1 / 3, -1 / 3, -1 / 3, 0])

    assert loss.value(scores) == pytest.approx(2)
    np.testing.assert_allclose(loss.negative_gradient(scores), expected)
    order = [4, 0, 5, 2, 1, 3, 7, 6]
    reordered = Top5Loss.from_frame(frame.iloc[order])
    np.testing.assert_allclose(reordered.negative_gradient(scores[order]), expected[order])
    assert reordered.value(scores[order]) == pytest.approx(loss.value(scores))

    scores[0] = 2.0
    assert loss.value(scores) == 0
    np.testing.assert_array_equal(loss.negative_gradient(scores), np.zeros(len(scores)))


@pytest.mark.parametrize("negative_count, expected_loss", [(0, 0), (4, 0), (5, 1)])
def test_small_events_have_no_loss_until_top5_excludes_an_asset(negative_count: int, expected_loss: float) -> None:
    frame = _frame(["a"] * (negative_count + 1), [1] + [0] * negative_count)
    scores = np.zeros(len(frame))
    loss = Top5Loss.from_frame(frame)

    assert loss.value(scores) == expected_loss
    expected_gradient = np.zeros(len(frame))
    if negative_count >= 5:
        expected_gradient[0] = 1
        expected_gradient[1:] = -1 / negative_count
    np.testing.assert_allclose(loss.negative_gradient(scores), expected_gradient)


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 0], [0.5, 0.5, 0]])
def test_requires_one_binary_positive_per_event(labels: list[float]) -> None:
    with pytest.raises(ValueError, match="binary|exactly one"):
        Top5Loss.from_frame(_frame(["a"] * 3, labels))


def _feature_set() -> FeatureSet:
    return FeatureSet(numeric_features=["signal"], target=COL_IS_PUMPED)


def _splits() -> dict[DatasetType, pd.DataFrame]:
    splits = {}
    for kind in (DatasetType.TRAIN, DatasetType.TUNING):
        frame = _frame([f"{kind.name}-{event}" for event in range(4) for _ in range(6)], [1, 0, 0, 0, 0, 0] * 4)
        frame["signal"] = frame[COL_IS_PUMPED] * 2.0 + np.arange(len(frame)) / 100
        frame["target_return@5MIN"] = np.arange(len(frame), dtype=float)
        splits[kind] = frame.iloc[np.arange(len(frame)).reshape(4, 6).T.flatten()].copy()
    return splits


def test_pipeline_preserves_rows_and_binary_labels_without_future_returns() -> None:
    feature_set = _feature_set()
    feature_set.target = "target_return@5MIN"
    pipeline = CatboostTop5Pipeline(feature_set=feature_set)
    splits = _splits()
    original = {kind: frame.copy(deep=True) for kind, frame in splits.items()}
    sample = pipeline.create_sample_from_datasets(splits)
    without_returns = pipeline.create_sample_from_datasets(
        {kind: frame.drop(columns="target_return@5MIN") for kind, frame in splits.items()}
    )

    assert pipeline.feature_set.target == COL_IS_PUMPED
    assert feature_set.target == "target_return@5MIN"
    for kind, dataset in sample.iter_datasets():
        pd.testing.assert_frame_equal(dataset.all_data(), original[kind])
        pd.testing.assert_frame_equal(splits[kind], original[kind])
        pd.testing.assert_frame_equal(dataset.get_data(), without_returns.get_data(kind))
        pd.testing.assert_series_equal(dataset.get_label(), without_returns.get_label(kind))


def test_real_tree_boosting_reduces_loss_and_preserves_prediction_alignment() -> None:
    pipeline = CatboostTop5Pipeline(feature_set=_feature_set())
    # No VALIDATION or TEST split is supplied: training and stopping must not read it.
    sample = pipeline.create_sample_from_datasets(_splits())
    model = CatboostTop5Model({"iterations": 8, "max_depth": 2, "learning_rate": 0.5, "thread_count": 1}).train(sample)
    tuning = sample.get_dataset(DatasetType.TUNING)
    scores = model.rank(tuning)
    loss = Top5Loss.from_frame(tuning.all_data())

    assert model.ranking_score_kind == "logit"
    assert scores.shape == (len(tuning.all_data()),)
    assert np.isfinite(scores).all()
    assert np.ptp(scores) > 0
    assert loss.value(scores) < loss.value(np.zeros(len(scores)))
    assert loss.value(scores) == pytest.approx(model.best_score_)
    assert model.training_history_[-1]["training_loss"] < model.training_history_[0]["training_loss"]

    order = np.arange(len(scores))[::-1]
    reordered = Dataset(tuning.all_data().iloc[order], pipeline.feature_set, DatasetType.TUNING)
    np.testing.assert_allclose(model.rank(reordered), scores[order])
    importances = model.get_feature_importances()
    assert list(importances.index) == ["signal"]
    assert np.isfinite(importances).all()


@pytest.mark.parametrize("updates, expected_iteration", [([0.2, -0.4, -0.4], 1), ([-0.2, -0.2, -0.2], 0)])
def test_early_stopping_restores_best_tuning_prefix_including_zero(
    monkeypatch: pytest.MonkeyPatch, updates: list[float], expected_iteration: int
) -> None:
    # Controlled trees make tuning improve then deteriorate, or deteriorate
    # immediately. This checks prediction of the selected prefix, not just its log.
    monkeypatch.setattr(
        CatboostTop5Model,
        "_fit_tree",
        lambda self, data, residual, dataset, iteration: updates[iteration - 1],
    )
    monkeypatch.setattr(
        CatboostTop5Model,
        "_tree_scores",
        staticmethod(lambda tree, data: tree * data["signal"].to_numpy()),
    )
    monkeypatch.setattr(
        "pumps_and_dumps.backtest.pipelines.catboost_top5.model.sum_models",
        lambda models, weights, **kwargs: float(np.dot(models, weights)),
    )
    pipeline = CatboostTop5Pipeline(feature_set=_feature_set())
    sample = pipeline.create_sample_from_datasets(_splits())
    model = CatboostTop5Model({"iterations": 3, "learning_rate": 1.0, "early_stopping_rounds": 2}).train(sample)
    tuning = sample.get_dataset(DatasetType.TUNING)
    expected_scores = sum(updates[:expected_iteration]) * tuning.get_data()["signal"].to_numpy()

    assert model.best_iteration_ == expected_iteration
    assert len(model.training_history_) == expected_iteration + 3
    np.testing.assert_allclose(model.rank(tuning), expected_scores)
    loss = Top5Loss.from_frame(tuning.all_data())
    assert model.best_score_ == pytest.approx(loss.value(expected_scores))


def test_tuning_uses_separate_minimizing_study_and_reloads_best_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage = f"sqlite:///{tmp_path / 'top5-study.db'}"
    previous = optuna.create_study(study_name="CatboostMultiTopKPipelineStudy", storage=storage)
    previous.set_user_attr("preserve", True)
    pipeline = CatboostTop5Pipeline(feature_set=_feature_set(), study_storage=storage)
    sample = pipeline.create_sample_from_datasets(_splits())
    monkeypatch.setattr(pipeline, "create_sample", lambda: sample)
    monkeypatch.setitem(top5_pipeline._BASE_PARAMS, "iterations", 3)
    monkeypatch.setitem(top5_pipeline._BASE_PARAMS, "thread_count", 1)
    study = pipeline.optimize_parameters(n_trials=1)
    model = pipeline.train(sample, tuned=True)
    tuning = sample.get_dataset(DatasetType.TUNING)
    loss = Top5Loss.from_frame(tuning.all_data())

    assert study.study_name == "CatboostTop5PipelineStudy"
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert study.user_attrs["objective"] == "equal_event_top5_hinge_v1"
    assert study.user_attrs["top_k"] == 5
    assert model.best_score_ == pytest.approx(study.best_value)
    assert loss.value(model.rank(tuning)) == pytest.approx(study.best_value)
    assert optuna.load_study(study_name=previous.study_name, storage=storage).user_attrs == {"preserve": True}
