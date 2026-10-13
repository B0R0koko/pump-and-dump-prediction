"""Every search fits and scores successfully without either held-out partition."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from optuna.trial import FixedTrial

from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PUMP_HASH,
    COL_PUMP_ID,
)
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import DatasetType


@pytest.mark.parametrize(
    "module_name, pipeline_name",
    [
        ("logistic_regression", "LogisticRegressionPipeline"),
        ("random_forest", "RandomForestPipeline"),
        ("catboost_classifier", "CatboostClassifierPipeline"),
        ("catboost_smote", "CatboostClassifierSMOTEPipeline"),
        ("catboost_ranker", "CatboostRankerPipeline"),
        ("catboost_topk_auc", "CatboostClassifierTOPKAUCPipeline"),
        ("catboost_query_softmax", "CatboostQuerySoftmaxPipeline"),
        ("catboost_top5", "CatboostTop5Pipeline"),
    ],
)
def test_objective_never_requires_comparison_or_test(
    module_name: str, pipeline_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(f"pumps_and_dumps.backtest.pipelines.{module_name}.pipeline")
    pipeline = getattr(module, pipeline_name)(
        feature_set=FeatureSet(numeric_features=["signal"], target=COL_IS_PUMPED)
    )
    splits = {}
    for kind in (DatasetType.TRAIN, DatasetType.TUNING):
        labels = np.tile([1, 0, 0, 0, 0, 0], 6)
        splits[kind] = pd.DataFrame(
            {
                COL_PUMP_HASH: [f"{kind.name}-{event}" for event in range(6) for _ in range(6)],
                COL_PUMP_ID: np.repeat(np.arange(6), 6),
                COL_CURRENCY_PAIR: [f"ASSET{index % 6}-BTC" for index in range(36)],
                COL_IS_PUMPED: labels,
                "asset_return_rank": np.tile(np.arange(1, 7) / 6, 6),
                "signal": labels + np.arange(36) / 100,
            }
        )
    sample = pipeline.create_sample_from_datasets(splits)
    if module_name.startswith("catboost"):
        iteration_key = "num_boost_round" if "num_boost_round" in module._BASE_PARAMS else "iterations"
        monkeypatch.setitem(module._BASE_PARAMS, iteration_key, 3)
        monkeypatch.setitem(module._BASE_PARAMS, "thread_count", 1)
    if module_name == "random_forest":
        monkeypatch.setitem(module._BASE_PARAMS, "n_jobs", 1)
    trial = FixedTrial(
        {
            "class_weight": 10.0,
            "lambda": 10.0,
            "max_features": 0.5,
            "max_samples": 0.5,
            "max_depth": 2,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "colsample_bylevel": 0.7,
            "subsample": 0.7,
            "iterations": 10,
            "l2_leaf_reg": 1.0,
        }
    )

    assert np.isfinite(module._objective(trial, sample))
