from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.research.portfolio_simulation import (
    cached_outcomes_for_dataset,
    evaluate_validation_portfolio_simulation,
    train_simulation_candidates,
)


def _experiment(name="model", reverse=False):
    frame = pd.DataFrame(
        {
            "pump_hash": ["event"] * 3,
            "currency_pair": ["AAA-BTC", "BBB-BTC", "CCC-BTC"],
            "pump_time": [pd.Timestamp("2021-01-01")] * 3,
            "is_pumped": [True, False, False],
            "signal": [1.0, 0.0, 0.0],
        }
    )
    if reverse:
        frame = frame.iloc[::-1].reset_index(drop=True)
    dataset = Dataset(frame, FeatureSet(["signal"], "is_pumped"), DatasetType.VALIDATION)

    def validation_only(split):
        assert split is DatasetType.VALIDATION
        return dataset

    model = SimpleNamespace(
        ranking_score_kind="probability", rank=lambda dataset: dataset.get_data()["signal"].to_numpy()
    )
    return Experiment(name, model, SimpleNamespace(get_dataset=validation_only))


def _outcomes():
    return pd.DataFrame(
        {
            "pump_hash": ["event"] * 3,
            "currency_pair": ["CCC-BTC", "AAA-BTC", "BBB-BTC"],
            "pump_time": [pd.Timestamp("2021-01-01")] * 3,
            "net_return": [-0.2, 0.5, 0.0],
            "executed": [True, True, False],
        }
    )


def test_cached_outcomes_subset_keeps_every_candidate_and_rejects_missing_keys(tmp_path):
    path = tmp_path / "outcomes.parquet"
    dataset = _experiment().get_sample().get_dataset(DatasetType.VALIDATION)
    assert cached_outcomes_for_dataset(path, dataset) is None
    old_event = _outcomes().assign(pump_hash="earlier_event")
    pd.concat([old_event, _outcomes()], ignore_index=True).to_parquet(path)

    subset = cached_outcomes_for_dataset(path, dataset)
    assert subset.pump_hash.tolist() == ["event"] * 3
    assert subset.currency_pair.tolist() == ["AAA-BTC", "BBB-BTC", "CCC-BTC"]
    assert subset.net_return.tolist() == [0.5, 0.0, -0.2]
    assert subset.executed.tolist() == [True, False, True]

    subset.iloc[:2].to_parquet(path)
    with pytest.raises(ValueError, match="missing 1 requested candidates"):
        cached_outcomes_for_dataset(path, dataset)

    pd.concat([subset, subset.iloc[:1]], ignore_index=True).to_parquet(path)
    with pytest.raises(ValueError, match="duplicate candidate identities"):
        cached_outcomes_for_dataset(path, dataset)


def test_validation_simulation_aligns_preprocessing_orders_and_exports(tmp_path):
    result = evaluate_validation_portfolio_simulation(
        [_experiment("forward"), _experiment("reverse", reverse=True)],
        outcomes=_outcomes(),
        portfolio_size=1,
        n_simulations=30,
        output_dir=tmp_path,
    )
    model_paths = result.paths[result.paths.strategy.ne("uniform")]
    np.testing.assert_allclose(model_paths.final_return, 0.5)
    np.testing.assert_allclose(model_paths.max_drawdown, 0.0)
    assert len(result.paths) == 90
    assert (tmp_path / "run.json").is_file()
    assert len(pd.read_parquet(tmp_path / "validation_scores.parquet")) == 6
    assert len(pd.read_parquet(tmp_path / "event_returns.parquet")) == 90
    assert result.summary.set_index("strategy").loc["forward", "top1_inclusion_rate"] == 1


def test_validation_simulation_rejects_extra_outcomes_instead_of_dropping_them():
    outcomes = _outcomes()
    extra = outcomes.iloc[[0]].assign(currency_pair="EXTRA-BTC")
    with pytest.raises(ValueError):
        evaluate_validation_portfolio_simulation(
            [_experiment()],
            outcomes=pd.concat([outcomes, extra]),
            portfolio_size=1,
            n_simulations=2,
        )


def test_validation_simulation_rejects_models_with_different_candidates():
    different = _experiment("different")
    different.get_sample().get_dataset(DatasetType.VALIDATION).all_data().loc[0, "currency_pair"] = "OTHER-BTC"
    with pytest.raises(ValueError, match="candidate universe differs"):
        evaluate_validation_portfolio_simulation([_experiment(), different], outcomes=_outcomes())


def test_simulation_candidates_use_all_eight_tuned_pipelines_with_clean_names(monkeypatch):
    calls = []

    def train(pipeline, name, *, tuned):
        experiment = SimpleNamespace(pipeline=type(pipeline).__name__, name=name, tuned=tuned)
        calls.append(experiment)
        return experiment

    monkeypatch.setattr("pumps_and_dumps.backtest.research.training.train_experiment", train)
    experiments = train_simulation_candidates()
    assert experiments == calls
    assert len(experiments) == 8
    assert {experiment.pipeline: experiment.name for experiment in experiments} == {
        "LogisticRegressionPipeline": "Logistic regression",
        "RandomForestPipeline": "Random forest",
        "CatboostClassifierPipeline": "CatBoost classifier",
        "CatboostClassifierSMOTEPipeline": "CatBoost classifier + SMOTE",
        "CatboostRankerPipeline": "CatBoost ranker",
        "CatboostClassifierTOPKAUCPipeline": "CatBoost classifier + Top@k%AUC early stopping",
        "CatboostQuerySoftmaxPipeline": "CatBoost + QuerySoftMax",
        "CatboostTop5Pipeline": "CatBoost + Top-5",
    }
    assert all(experiment.tuned is True for experiment in experiments)
    assert all("tuned" not in experiment.name.lower() for experiment in experiments)
