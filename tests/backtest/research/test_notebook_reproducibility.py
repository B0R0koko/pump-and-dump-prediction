"""Repeat real notebook searches and fits in independent Python processes."""

import json
import os
from pathlib import Path
import subprocess
import sys


def _replay_models(directory):
    from contextlib import ExitStack
    from importlib import import_module
    from unittest.mock import patch

    import numpy as np
    import optuna
    import pandas as pd

    from backtest.pipelines import BasePipeline, study as studies
    from backtest.utils.feature_set import FeatureSet
    from backtest.utils.sample import DatasetType
    from research_snapshots.reproducibility import initialize_reproducibility
    from pumps_and_dumps.backtest.pipelines.logistic_regression.statsmodels import StatsmodelsLogitModel
    from tests.notebook_helpers import analysis

    settings = initialize_reproducibility()
    assert settings["random_seed"] == 42
    assert settings["optuna_n_jobs"] == settings["model_threads"] == 1
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(123)
    frames = {}
    for kind, events in ((DatasetType.TRAIN, 12), (DatasetType.VALIDATION, 8)):
        count = events * 16
        labels = np.tile(np.arange(16) == 0, events)
        frames[kind] = pd.DataFrame({
            "pump_hash": np.repeat([f"{kind.name}-{i}" for i in range(events)], 16),
            "pump_id": np.repeat(np.arange(events), 16),
            "currency_pair": np.tile([f"ASSET{i}-BTC" for i in range(16)], events),
            "is_pumped": labels,
            "asset_return_rank": np.tile(np.arange(1, 17) / 16, events),
            "signal": rng.normal(size=count) + labels * 0.7,
            "noise": rng.normal(size=count),
        })

    # Exercise TPE beyond its ten startup trials, with tiny tree budgets so the
    # real objectives and final selected-parameter fits remain cheap to run.
    suggest_int = optuna.Trial.suggest_int
    optimize = optuna.study.Study.optimize

    def small_int(trial, name, low, high, **kwargs):
        if name in {"iterations", "n_estimators"}:
            low, high = 6, 9
        if name == "max_depth":
            high = min(high, 3)
        return suggest_int(trial, name, low, high, **kwargs)

    def sequential(study, *args, **kwargs):
        assert kwargs["n_jobs"] == 1
        return optimize(study, *args, **kwargs)

    results = {}
    names = ("LogisticRegression", "RandomForest", "CatboostClassifier",
             "CatboostClassifierSMOTE", "CatboostRanker", "CatboostClassifierTOPKAUC")
    for name in names:
        module = import_module(f"backtest.pipelines.{name}.pipe")
        pipeline = getattr(module, f"{name}Pipeline")()
        target = "asset_return_rank" if name == "CatboostRanker" else "is_pumped"
        pipeline.feature_set = FeatureSet(["signal", "noise"], target)
        pipeline.build_datasets = lambda: {kind: data.copy(deep=True) for kind, data in frames.items()}
        params = dict(module._BASE_PARAMS)
        if name.startswith("Catboost") and name != "CatboostRanker":
            params["num_boost_round"] = 8
        params["verbose"] = False
        storage = f"sqlite:///{directory / (name + '.db')}"
        with ExitStack() as stack:
            for target_module in (studies, BasePipeline, module):
                stack.enter_context(patch.object(target_module, "SQLITE_URL", storage, create=True))
            stack.enter_context(patch.object(module, "_BASE_PARAMS", params))
            stack.enter_context(patch.object(optuna.Trial, "suggest_int", small_int))
            stack.enter_context(patch.object(optuna.study.Study, "optimize", sequential))
            study = pipeline.optimize_parameters(n_trials=12)
            assert study.user_attrs["reproducibility"]["random_seed"] == 42
            sample = pipeline.create_sample()
            model = pipeline.train(sample, tuned=True)
            seed_key = "random_seed" if name.startswith("Catboost") else "random_state"
            assert model.params[seed_key] == 42
            validation = sample.get_dataset(DatasetType.VALIDATION)
            results[name] = {
                "trials": [(trial.params, trial.value) for trial in study.trials],
                "best_trial": study.best_trial.number,
                "predictions": model.rank(validation).tolist(),
            }
            if name == "LogisticRegression":
                # Override a conflicting inherited seed: L2 must use its own
                # explicit policy, for every candidate in its fixed grid.
                l2, path = analysis.fit_l2_logistic_path(
                    sample, model.params | {"random_state": 999}, [0.001, 0.01, 0.1],
                )
                assert l2.params["random_state"] == 42
                results["L2"] = {"path": path.to_dict("list"), "predictions": l2.rank(validation).tolist()}
                diagnostic = StatsmodelsLogitModel().train(sample, train_type=DatasetType.TRAIN)
                results["diagnostic"] = diagnostic.rank(validation).tolist()
    return results


def test_fresh_searches_and_training_repeat_across_processes(tmp_path):
    """All seven predictive candidates repeat despite different Python hash seeds.

    Uses fresh SQLite studies and actual estimators, including SMOTE and the
    custom CatBoost metric. Only miniature training/validation data are supplied;
    production databases, features and held-out test events are never read.
    """
    outputs = []
    root = Path(__file__).resolve().parents[3]
    for hash_seed in ("1", "98765"):
        directory = tmp_path / hash_seed
        directory.mkdir()
        process = subprocess.run(
            [sys.executable, "-m", "tests.backtest.research.test_notebook_reproducibility", str(directory)],
            cwd=root,
            env=os.environ | {"PYTHONHASHSEED": hash_seed, "PUMPS_AND_DUMPS_MODEL_THREADS": "1"},
            capture_output=True, text=True, timeout=180,
        )
        assert process.returncode == 0, process.stderr[-6000:]
        outputs.append(json.loads((directory / "results.json").read_text()))
    for name in outputs[0]:
        assert outputs[0][name] == outputs[1][name], f"Search or predictions changed for {name}"


if __name__ == "__main__":
    output = Path(sys.argv[1])
    (output / "results.json").write_text(json.dumps(_replay_models(output), allow_nan=False))
