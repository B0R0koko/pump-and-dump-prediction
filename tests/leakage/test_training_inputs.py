"""Tiny fit-boundary examples: train on TRAIN; early-stop only on TUNING."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import DatasetType


@pytest.mark.parametrize(
    "pipeline_module, pipeline_name, estimator_module, estimator_name",
    [
        ("catboost_classifier", "CatboostClassifierPipeline", "catboost_classifier", "CatBoostClassifier"),
        ("catboost_query_softmax", "CatboostQuerySoftmaxPipeline", "catboost_ranker", "CatBoostRanker"),
    ],
)
def test_fit_receives_exact_training_rows_and_tuning_labels_only(
    pipeline_module: str,
    pipeline_name: str,
    estimator_module: str,
    estimator_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit both classifier and QuerySoftMax pipelines using only allowed splits.

    Give each split two uniquely recognizable signals: TRAIN [1, 2], TUNING
    [11, 12], comparison [101, 102], and TEST [1001, 1002]. A recording
    estimator captures the actual training call without fitting a real model.
    Expect one feature column, TRAIN labels [1, 0], and TUNING labels [0, 1]
    in the evaluation pool, with 50-round early stopping and best-model use.
    Both held-out dataset access and their data/label/pool accessors fail
    immediately if read. Extra return, rank, and identifier columns must not
    enter either feature pool. This catches held-out fitting, early stopping
    on comparison/test data, swapped labels, and target-derived predictors.
    """
    module = import_module(f"pumps_and_dumps.backtest.pipelines.{pipeline_module}.pipeline")
    pipeline = getattr(module, pipeline_name)(feature_set=FeatureSet(numeric_features=["signal"], target="is_pumped"))
    splits = {}
    for kind, signals, labels in [
        (DatasetType.TRAIN, [1.0, 2.0], [1, 0]),
        (DatasetType.TUNING, [11.0, 12.0], [0, 1]),
        (DatasetType.VALIDATION, [101.0, 102.0], [1, 0]),
        (DatasetType.TEST, [1001.0, 1002.0], [0, 1]),
    ]:
        splits[kind] = pd.DataFrame(
            {
                "pump_hash": [kind.name, kind.name],
                "currency_pair": ["ABTC", "BBTC"],
                "signal": signals,
                "is_pumped": labels,
                "asset_return_rank": [0.5, 1.0],
                "net_return": [99.0, -0.9],
            }
        )
    sample = pipeline.create_sample_from_datasets(splits)
    get_dataset = sample.get_dataset
    reads = []

    def forbidden_read():
        pytest.fail("Held-out data read during fit")

    for kind in (DatasetType.VALIDATION, DatasetType.TEST):
        dataset = get_dataset(kind)
        for accessor in ("all_data", "get_data", "get_label", "get_eval_data", "get_pool"):
            monkeypatch.setattr(dataset, accessor, forbidden_read)

    def guarded_dataset(ds_type):
        assert ds_type in (DatasetType.TRAIN, DatasetType.TUNING), "Held-out data read during fit"
        reads.append(ds_type)
        return get_dataset(ds_type)

    monkeypatch.setattr(sample, "get_dataset", guarded_dataset)
    calls = []

    class RecordingEstimator:
        def fit(self, **kwargs):
            calls.append(kwargs)
            return self

    estimator = RecordingEstimator()
    model_module = import_module(f"pumps_and_dumps.backtest.pipelines.{estimator_module}.model")
    monkeypatch.setattr(model_module, estimator_name, lambda **kwargs: estimator)

    pipeline.train(sample, tuned=False)

    assert reads == [DatasetType.TRAIN, DatasetType.TUNING]
    assert len(calls) == 1
    fit = calls[0]
    # These literal expectations catch mixing partitions, swapping labels, and
    # leaking labels, future returns, or event identifiers into the feature X.
    assert fit["X"].get_feature_names() == ["signal"]
    np.testing.assert_array_equal(fit["X"].get_features(), [[1.0], [2.0]])
    np.testing.assert_array_equal(fit["X"].get_label(), [1, 0])
    assert fit["eval_set"].get_feature_names() == ["signal"]
    np.testing.assert_array_equal(fit["eval_set"].get_features(), [[11.0], [12.0]])
    np.testing.assert_array_equal(fit["eval_set"].get_label(), [0, 1])
    assert fit["early_stopping_rounds"] == 50
    assert fit["use_best_model"] is True
