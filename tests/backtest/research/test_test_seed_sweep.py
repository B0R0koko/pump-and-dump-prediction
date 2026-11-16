"""Check saved-parameter refits and isolation of the test sample."""

import numpy as np
import pandas as pd
import pytest

from research_snapshots import seed_sweep as validation
from research_snapshots import test_seed_sweep as evaluation
from tests.backtest.research.test_seed_sweep import miniature_data, restore_model_defaults


@pytest.mark.parametrize("key", list(validation.MODELS))
def test_saved_parameters_reproduce_validation_and_test_labels_cannot_change_scores(key, tmp_path, monkeypatch):
    import optuna
    import backtest.pipelines.BasePipeline as base
    from backtest.utils.feature_set import FeatureSet

    original_configure = validation.configure_model
    original_suggest = optuna.Trial.suggest_int

    def small_int(trial, name, low, high, **kwargs):
        if name in ("iterations", "n_estimators"):
            low, high = 6, 9
        if name == "max_depth":
            high = min(high, 3)
        return original_suggest(trial, name, low, high, **kwargs)

    def configure(key, seed):
        module, pipeline = original_configure(key, seed)
        pipeline.feature_set = FeatureSet(["signal", "noise"], "asset_return_rank" if key == "ranker" else "is_pumped")
        pipeline.preprocess_data = lambda df: base.fillna_with_median_by_cross_section(df, pipeline.feature_set)
        if key not in ("logistic", "forest", "ranker"):
            module._BASE_PARAMS["num_boost_round"] = 8
        return module, pipeline

    monkeypatch.setattr(optuna.Trial, "suggest_int", small_int)
    monkeypatch.setattr(validation, "configure_model", configure)
    raw = miniature_data()
    raw.to_parquet(tmp_path / "development.parquet", index=False)
    validation.write_json(tmp_path / "config.json", {"trials_per_study": 2})
    validation.write_json(tmp_path / "data.json", {"inner_cutoff": "2020-01-13"})
    record = validation.run_study(key, 7, tmp_path)
    saved = pd.read_parquet(tmp_path / "studies" / key / "seed_007/validation_predictions.parquet")
    model, pipeline, validation_data = evaluation.fit_saved_configuration(key, 7, record, raw, "2020-01-13")
    np.testing.assert_array_equal(model.rank(validation_data), saved.score.to_numpy())
    if key not in ("logistic", "forest"):
        assert model._model.tree_count_ == record["tree_count"]
        assert not model._model.get_evals_result().get("validation")
    test = raw.loc[raw.pump_time >= pd.Timestamp(validation.TRAIN_END)].copy()
    test.pump_time = test.pump_time + pd.DateOffset(years=1)
    test.pump_hash = "test-" + test.pump_hash
    before = evaluation.build_test_dataset(key, pipeline, raw, test)
    scores = model.rank(before)
    test.is_pumped = test.currency_pair.eq("ASSET11BTC")
    test.asset_return_rank = 1 - test.asset_return_rank
    after = evaluation.build_test_dataset(key, pipeline, raw, test)
    np.testing.assert_array_equal(scores, model.rank(after))


def test_test_imputation_uses_training_priors_and_excludes_validation():
    import backtest.pipelines.BasePipeline as base
    from backtest.utils.feature_set import FeatureSet

    _, pipeline = validation.configure_model("logistic", 7)
    pipeline.feature_set = FeatureSet(["signal", "noise"], "is_pumped")
    pipeline.preprocess_data = lambda df: base.fillna_with_median_by_cross_section(df, pipeline.feature_set)
    development = miniature_data()
    test = development.iloc[:24].copy()
    test.pump_time = test.pump_time + pd.DateOffset(years=2)
    test.pump_hash = "test-" + test.pump_hash
    test["signal"] = np.nan
    expected = development.loc[development.pump_time < pd.Timestamp(validation.TRAIN_END), "signal"].median()
    first = evaluation.build_test_dataset("logistic", pipeline, development, test)
    np.testing.assert_allclose(first.get_data().signal, expected)
    development.loc[development.pump_time >= pd.Timestamp(validation.TRAIN_END), "signal"] = 1e20
    test.loc[test.pump_hash == test.pump_hash.iloc[12], "signal"] = -1e20
    second = evaluation.build_test_dataset("logistic", pipeline, development, test)
    np.testing.assert_allclose(second.get_data().signal.iloc[:12], expected)
    assert base.TRAIN_END == pd.Timestamp(validation.TRAIN_END)
