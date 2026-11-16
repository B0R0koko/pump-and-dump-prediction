"""Check split/seed isolation with real searches and tiny model fits."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from research_snapshots import seed_sweep as sweep


@pytest.fixture(autouse=True)
def restore_model_defaults():
    import importlib

    modules = [importlib.import_module(f"backtest.pipelines.{name}.pipe") for name, _ in sweep.MODELS.values()]
    defaults = [(module, module._BASE_PARAMS.copy(), module.RANDOM_SEED) for module in modules]
    yield
    for module, parameters, seed in defaults:
        module._BASE_PARAMS = parameters
        module.RANDOM_SEED = seed


def test_inner_boundary_keeps_simultaneous_events_together():
    events = pd.DataFrame({"pump_time": pd.to_datetime([
        "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-04",
    ])})
    boundary = sweep.inner_boundary(events)
    assert boundary == pd.Timestamp("2020-01-04")
    assert (events.pump_time < boundary).sum() == 3


def miniature_data():
    rng = np.random.default_rng(98)
    frames = []
    for i, time in enumerate(pd.date_range("2020-01-01", periods=16).tolist()
                             + [pd.Timestamp("2020-09-02"), pd.Timestamp("2020-09-03")]):
        labels = np.arange(12) == 0
        frames.append(pd.DataFrame({
            "pump_hash": f"event-{i}", "pump_time": time, "pump_id": i,
            "currency_pair": [f"ASSET{j}BTC" for j in range(12)],
            "is_pumped": labels, "asset_return_rank": np.arange(1, 13) / 12,
            "signal": rng.normal(size=12) + labels, "noise": rng.normal(size=12),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("key", list(sweep.MODELS))
def test_outer_validation_cannot_change_search_or_final_fit(key, tmp_path, monkeypatch):
    import optuna
    from backtest.utils.feature_set import FeatureSet
    from backtest.utils.sample import DatasetType
    import backtest.pipelines.BasePipeline as base

    original_configure = sweep.configure_model
    suggest_int = optuna.Trial.suggest_int

    def small_int(trial, name, low, high, **kwargs):
        if name in ("iterations", "n_estimators"):
            low, high = 6, 9
        if name == "max_depth":
            high = min(high, 3)
        return suggest_int(trial, name, low, high, **kwargs)

    def configure(key, seed):
        module, pipeline = original_configure(key, seed)
        target = "asset_return_rank" if key == "ranker" else "is_pumped"
        pipeline.feature_set = FeatureSet(["signal", "noise"], target)
        pipeline.preprocess_data = lambda df: base.fillna_with_median_by_cross_section(df, pipeline.feature_set)
        if key not in ("logistic", "forest", "ranker"):
            module._BASE_PARAMS["num_boost_round"] = 8
        return module, pipeline

    monkeypatch.setattr(optuna.Trial, "suggest_int", small_int)
    monkeypatch.setattr(sweep, "configure_model", configure)
    raw = miniature_data()
    results, models = [], []
    for version in ("baseline", "changed_validation"):
        output = tmp_path / version
        output.mkdir()
        frame = raw.copy()
        if version == "changed_validation":
            mask = frame.pump_time >= pd.Timestamp(sweep.TRAIN_END)
            frame.loc[mask, ["signal", "noise"]] = 1e12
            # Keep one target per event, but change which asset is the target.
            frame.loc[mask, "is_pumped"] = frame.loc[mask, "currency_pair"].eq("ASSET11BTC")
        frame.to_parquet(output / "development.parquet", index=False)
        sweep.write_json(output / "config.json", {"trials_per_study": 2})
        sweep.write_json(output / "data.json", {"inner_cutoff": "2020-01-13"})
        result = sweep.run_study(key, 7, output)
        assert result["parameters"].get("random_state", result["parameters"].get("random_seed")) == 7
        results.append(result)
        with (output / "studies" / key / "seed_007/model.pickle").open("rb") as handle:
            models.append(pickle.load(handle))
    for field in ("parameters", "inner_tuning_auc", "best_trial", "tree_count"):
        assert results[0][field] == results[1][field]
    _, pipeline = configure(key, 7)
    comparison = sweep.build_samples(pipeline, raw, "2020-01-13", "full").get_dataset(DatasetType.VALIDATION)
    np.testing.assert_array_equal(models[0].rank(comparison), models[1].rank(comparison))
    if key not in ("logistic", "forest"):
        assert models[0]._model.tree_count_ == results[0]["tree_count"]
        assert not models[0]._model.get_evals_result().get("validation")


def test_inner_preprocessing_does_not_use_tuning_priors():
    from backtest.utils.feature_set import FeatureSet
    from backtest.utils.sample import DatasetType
    import backtest.pipelines.BasePipeline as base

    _, pipeline = sweep.configure_model("logistic", 17)
    pipeline.feature_set = FeatureSet(["signal", "noise"], "is_pumped")
    pipeline.preprocess_data = lambda df: base.fillna_with_median_by_cross_section(df, pipeline.feature_set)
    raw = miniature_data()
    raw.loc[raw.pump_time == pd.Timestamp("2020-01-01"), "signal"] = np.nan
    before = sweep.build_samples(pipeline, raw, "2020-01-13", "inner").get_data(DatasetType.TRAIN)
    raw.loc[raw.pump_time >= pd.Timestamp("2020-01-13"), "signal"] = -1e20
    after = sweep.build_samples(pipeline, raw, "2020-01-13", "inner").get_data(DatasetType.TRAIN)
    pd.testing.assert_frame_equal(before, after)
    assert base.TRAIN_END == pd.Timestamp(sweep.TRAIN_END)


def test_seeds_reach_sampler_estimators_and_smote():
    import optuna

    for key in sweep.MODELS:
        module, _ = sweep.configure_model(key, 123)
        assert module.RANDOM_SEED == 123
        seed_field = "random_state" if key in ("logistic", "forest") else "random_seed"
        assert module._BASE_PARAMS[seed_field] == 123
    values = []
    for seed in (7, 7, 8):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        trial = study.ask()
        values.append(trial.suggest_float("value", 0, 1))
    assert values[0] == values[1] and values[0] != values[2]


def test_selection_waits_for_all_seeds_and_uses_mean_not_best_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "plot_distributions", lambda *args: None)
    config = {"seeds": list(range(1, 101)), "trials_per_study": 25, "selection": "Highest mean"}
    sweep.write_json(tmp_path / "config.json", config)
    config_hash = sweep.digest(tmp_path / "config.json")
    records = []
    for key in sweep.MODELS:
        for seed in config["seeds"]:
            score = .65 if key == "catboost" else .60
            if key == "forest" and seed == 1:
                score = .99  # Best single run, but not the best mean.
            directory = tmp_path / "studies" / key / f"seed_{seed:03d}"
            directory.mkdir(parents=True)
            (directory / "model.pickle").write_bytes(f"{key}-{seed}".encode())
            records.append((directory / "result.json", {
                "model": key, "seed": seed, "validation_auc": score,
                "completed_trials": 25, "config_sha256": config_hash,
                "parameters": {}, "trial_seconds": [1.0] * 25, "elapsed_seconds": 26.0,
            }))
    for path, result in records[:-1]:
        sweep.write_json(path, result)
    assert sweep.summarize(tmp_path, 12)["completed_studies"] == 599
    assert not (tmp_path / "selection.json").exists()
    sweep.write_json(*records[-1])
    assert sweep.summarize(tmp_path, 12)["selected_model"] == "catboost"
    assert (tmp_path / "selected_model.pickle").read_bytes() == b"catboost-42"


def test_all_workers_respects_available_cpu_affinity(monkeypatch):
    monkeypatch.setattr(sweep.os, "sched_getaffinity", lambda pid: set(range(32)))
    assert sweep.worker_count("all") == 32
    assert sweep.worker_count("12") == 12


def test_import_preserves_completed_artifacts_and_excludes_seed_zero(tmp_path):
    import shutil
    import sqlite3

    source, destination = tmp_path / "previous", tmp_path / "current"
    for directory, seeds in [(source, list(range(100))), (destination, list(range(1, 101)))]:
        directory.mkdir()
        shutil.copy2(sweep.__file__, directory / "runner_snapshot.py")
        sweep.write_json(directory / "config.json", {
            "seeds": seeds, "trials_per_study": 25, "models": list(sweep.MODELS),
            "runner_sha256": sweep.digest(sweep.__file__), "feature_sha256": {"data": "unchanged"},
        })
        sweep.write_json(directory / "data.json", {"inner_cutoff": "2020-03-18T19:00:00"})
    original_hashes = {}
    for seed in [0, 1]:
        directory = source / "studies/catboost" / f"seed_{seed:03d}"
        directory.mkdir(parents=True)
        (directory / "model.pickle").write_bytes(f"model-{seed}".encode())
        (directory / "validation_predictions.parquet").write_bytes(f"predictions-{seed}".encode())
        with sqlite3.connect(directory / "studies.db") as connection:
            connection.execute("CREATE TABLE trials (state TEXT)")
            connection.executemany("INSERT INTO trials VALUES (?)", [("COMPLETE",)] * 25)
        sweep.write_json(directory / "result.json", {
            "model": "catboost", "seed": seed, "completed_trials": 25, "validation_auc": .65,
            "model_sha256": sweep.digest(directory / "model.pickle"),
            "prediction_sha256": sweep.digest(directory / "validation_predictions.parquet"),
            "config_sha256": sweep.digest(source / "config.json"),
        })
        original_hashes[seed] = {p.name: sweep.digest(p) for p in directory.iterdir()}
    report = sweep.import_completed(source, destination)
    assert [r["seed"] for r in report["imported"]] == [1]
    assert not (destination / "studies/catboost/seed_000").exists()
    new = destination / "studies/catboost/seed_001"
    for name in ["model.pickle", "validation_predictions.parquet", "studies.db"]:
        assert sweep.digest(new / name) == original_hashes[1][name]
    assert json.loads((new / "result.json").read_text())["origin_config_sha256"] == sweep.digest(source / "config.json")
    assert sweep.digest(new / "origin/result.json") == original_hashes[1]["result.json"]
    for seed, expected in original_hashes.items():
        assert {p.name: sweep.digest(p) for p in (source / "studies/catboost" / f"seed_{seed:03d}").iterdir()} == expected
    wrong = json.loads((destination / "config.json").read_text())
    wrong["feature_sha256"] = {"data": "modified"}
    sweep.write_json(destination / "config.json", wrong)
    with pytest.raises(ValueError, match="inputs"):
        sweep.import_completed(source, destination)
