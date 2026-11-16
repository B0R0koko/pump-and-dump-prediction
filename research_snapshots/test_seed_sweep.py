"""Refit saved seed-specific configurations and evaluate the fixed test events.

No hyperparameter search or model selection occurs here. Each fresh training fit
must reproduce its saved validation predictions exactly before test evaluation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

from research_snapshots import seed_sweep as validation

ROOT = validation.ROOT
MODELS = validation.MODELS
DEFAULT_OUTPUT = ROOT / "notebooks/analysis_outputs/test_seed_sweep_100x25"
digest = validation.digest
write_json = validation.write_json
PREDICTION_COLUMNS = ["pump_hash", "pump_time", "currency_pair", "is_pumped"]


def prepare(source, output):
    import pandas as pd
    from backtest.utils.build_dataset import load_pumps, _read_cross_section, FEATURE_DIR
    from research_snapshots.reproducibility import reproducibility_settings

    source_config = json.loads((source / "config.json").read_text())
    assert source_config["seeds"] == list(range(1, 101))
    assert source_config["models"] == list(MODELS)
    assert source_config["trials_per_study"] == 25
    assert not source_config["test_features_loaded"]
    assert source_config["runner_sha256"] == digest(validation.__file__)
    assert source_config["runtime_sha256"] == digest(ROOT / "research_snapshots/73ba716/runtime.zip")
    assert source_config["versions"] == reproducibility_settings()["versions"]
    source_data = json.loads((source / "data.json").read_text())
    assert source_data["development_sha256"] == digest(source / "development.parquet")
    assert source_data["event_manifest_sha256"] == digest(source / "events.csv")
    selection = json.loads((source / "selection.json").read_text())
    assert selection["final_seed"] == validation.FINAL_SEED
    source_records = {}
    for key in MODELS:
        for seed in source_config["seeds"]:
            directory = source / "studies" / key / f"seed_{seed:03d}"
            result = json.loads((directory / "result.json").read_text())
            assert result["model"] == key and result["seed"] == seed
            assert result["completed_trials"] == 25
            assert result["config_sha256"] == digest(source / "config.json")
            assert result["model_sha256"] == digest(directory / "model.pickle")
            assert result["prediction_sha256"] == digest(directory / "validation_predictions.parquet")
            source_records[f"{key}/{seed:03d}"] = digest(directory / "result.json")
    manifest = ROOT / "research_snapshots/paper_0167ddd/pumps.json"
    assert digest(manifest) == source_config["manifest_sha256"]
    events = [p for p in load_pumps(manifest) if p.time >= pd.Timestamp(validation.VALIDATION_END)]
    assert len(events) == 80
    features = {str(FEATURE_DIR / "pumps" / f"{event}.parquet"):
                digest(FEATURE_DIR / "pumps" / f"{event}.parquet") for event in events}
    config = {
        "protocol": "saved_100_seed_parameters_train_290_test_80_v1",
        "source": str(source), "source_config_sha256": digest(source / "config.json"),
        "source_data_sha256": digest(source / "data.json"),
        "development_sha256": source_data["development_sha256"],
        "source_result_sha256": source_records,
        "source_selection_sha256": digest(source / "selection.json"),
        "selected_model": selection["model"], "selected_seed": selection["final_seed"],
        "seeds": source_config["seeds"], "models": source_config["models"],
        "model_threads": 1, "new_search_trials": 0, "test_used_for_selection": False,
        "training_events": 290, "validation_events": 103, "test_events": 80,
        "validation_prediction_check": "Exact array equality before test evaluation",
        "test_feature_sha256": features, "auc": source_config["auc"],
        "versions": source_config["versions"], "runtime_sha256": source_config["runtime_sha256"],
        "runner_sha256": digest(__file__), "validation_runner_sha256": digest(validation.__file__),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Inputs or implementation changed; use a new output directory")
        data = json.loads((output / "data.json").read_text())
        assert data["test_sha256"] == digest(output / "test.parquet")
        assert data["events_sha256"] == digest(output / "events.csv")
        return config
    frames = []
    for event in events:
        _, frame = _read_cross_section(event)
        if frame is None:
            raise ValueError(f"Missing test event: {event}")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["is_pumped"] = raw.currency_pair == raw.pumped_currency_pair
    assert raw.pump_hash.nunique() == 80
    assert raw.groupby("pump_hash").is_pumped.sum().eq(1).all()
    event_frame = raw[["pump_hash", "pump_time"]].drop_duplicates()
    development_events = pd.read_csv(source / "events.csv")
    assert set(event_frame.pump_hash).isdisjoint(development_events.pump_hash)
    raw.to_parquet(output / "test.parquet", index=False)
    event_frame.to_csv(output / "events.csv", index=False)
    write_json(output / "data.json", {
        "test_sha256": digest(output / "test.parquet"), "events_sha256": digest(output / "events.csv"),
        "test_events": 80, "test_rows": len(raw),
    })
    shutil.copy2(source / "selection.json", output / "validation_selection.json")
    shutil.copy2(__file__, output / "runner_snapshot.py")
    write_json(output / "config.json", config)
    return config


def fit_saved_configuration(key, seed, record, development, inner_cutoff):
    """Repeat exactly the final training step; validation never controls the fit."""
    from backtest.utils.sample import DatasetType

    module, pipeline = validation.configure_model(key, seed)
    params = record["parameters"].copy()
    seed_field = "random_state" if key in ("logistic", "forest") else "random_seed"
    assert params[seed_field] == seed
    if isinstance(params.get("class_weight"), dict):
        params["class_weight"] = {int(k): v for k, v in params["class_weight"].items()}
    sample = validation.build_samples(pipeline, development, inner_cutoff, "full")
    symbol = ("LogisticRegressionModel" if key == "logistic" else "RandomForestModel" if key == "forest"
              else "CatboostRankerModel" if key == "ranker" else "CatboostClassifierModel")
    model = getattr(module, symbol)(params)
    if key in ("logistic", "forest"):
        model.train(sample)
    else:
        from catboost import CatBoostClassifier, CatBoostRanker

        assert params["iterations"] == record["tree_count"]
        assert params["use_best_model"] is False and "eval_metric" not in params
        model._model = (CatBoostRanker if key == "ranker" else CatBoostClassifier)(**params)
        model._model.fit(sample.get_pool(DatasetType.TRAIN))
    return model, pipeline, sample.get_dataset(DatasetType.VALIDATION)


def build_test_dataset(key, pipeline, development, test):
    """Apply the same preprocessing, with fallback priors from training only."""
    import pandas as pd
    import backtest.pipelines.BasePipeline as base
    from backtest.utils.sample import Dataset, DatasetType

    train_end = pd.Timestamp(validation.TRAIN_END)
    test_start = pd.Timestamp(validation.VALIDATION_END)
    training = development.loc[development.pump_time < train_end].copy()
    assert test.pump_time.ge(test_start).all()
    assert set(training.pump_hash).isdisjoint(test.pump_hash)
    previous = base.TRAIN_END
    try:
        base.TRAIN_END = train_end
        data = pipeline.preprocess_data(pd.concat([training, test], ignore_index=True))
    finally:
        base.TRAIN_END = previous
    data = data.loc[data.pump_time >= test_start].copy()
    dataset = Dataset(data, pipeline.feature_set, DatasetType.TEST)
    if key not in ("logistic", "forest"):
        from catboost import Pool

        kwargs = {"group_id": data["pump_id"]} if key == "ranker" else {}
        dataset.set_pool(Pool(dataset.get_data(), label=dataset.get_label(),
                              cat_features=pipeline.feature_set.categorical_features, **kwargs))
    return dataset


def verify_completed(directory, config_hash):
    result = json.loads((directory / "result.json").read_text())
    assert result["config_sha256"] == config_hash
    assert result["validation_predictions_identical"] is True
    for name, field in [("model.pickle", "model_sha256"),
                        ("test_predictions.parquet", "test_prediction_sha256"),
                        ("validation_predictions.parquet", "validation_prediction_sha256")]:
        assert digest(directory / name) == result[field]
    return result


def run_one(key, seed, output):
    import logging
    import numpy as np
    import pandas as pd
    from backtest.utils.metrics import calculate_topk_percent_auc

    logging.getLogger().setLevel(logging.ERROR)
    config = json.loads((output / "config.json").read_text())
    config_hash = digest(output / "config.json")
    assert digest(__file__) == config["runner_sha256"]
    assert digest(validation.__file__) == config["validation_runner_sha256"]
    source = Path(config["source"])
    source_directory = source / "studies" / key / f"seed_{seed:03d}"
    assert digest(source_directory / "result.json") == config["source_result_sha256"][f"{key}/{seed:03d}"]
    record = json.loads((source_directory / "result.json").read_text())
    assert digest(source_directory / "validation_predictions.parquet") == record["prediction_sha256"]
    directory = output / "fits" / key / f"seed_{seed:03d}"
    if (directory / "result.json").exists():
        return verify_completed(directory, config_hash)
    if directory.exists():
        archive = output / "interrupted" / key
        archive.mkdir(parents=True, exist_ok=True)
        directory.rename(archive / f"seed_{seed:03d}_{time.time_ns()}")
    directory.mkdir(parents=True)
    started = time.monotonic()
    assert digest(source / "development.parquet") == config["development_sha256"]
    development = pd.read_parquet(source / "development.parquet")
    source_data = json.loads((source / "data.json").read_text())
    model, pipeline, validation_data = fit_saved_configuration(key, seed, record, development, source_data["inner_cutoff"])
    fit_seconds = time.monotonic() - started
    predictions = validation_data.all_data()[PREDICTION_COLUMNS].copy().reset_index(drop=True)
    predictions["score"] = model.rank(validation_data)
    saved = pd.read_parquet(source_directory / "validation_predictions.parquet")
    pd.testing.assert_frame_equal(predictions[PREDICTION_COLUMNS], saved[PREDICTION_COLUMNS].reset_index(drop=True))
    np.testing.assert_array_equal(predictions.score.to_numpy(), saved.score.to_numpy())
    validation_auc = float(calculate_topk_percent_auc(model, validation_data))
    assert validation_auc == record["validation_auc"]
    data_record = json.loads((output / "data.json").read_text())
    assert digest(output / "test.parquet") == data_record["test_sha256"]
    test_data = build_test_dataset(key, pipeline, development, pd.read_parquet(output / "test.parquet"))
    test_predictions = test_data.all_data()[PREDICTION_COLUMNS].copy().reset_index(drop=True)
    test_predictions["score"] = model.rank(test_data)
    assert test_predictions.pump_hash.nunique() == config["test_events"]
    test_auc = float(calculate_topk_percent_auc(model, test_data))
    predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
    test_predictions.to_parquet(directory / "test_predictions.parquet", index=False)
    with (directory / "model.pickle").open("wb") as stream:
        pickle.dump(model, stream)
    with (directory / "test_predictions.parquet").open("rb") as stream:
        os.fsync(stream.fileno())
    result = {
        "model": key, "label": MODELS[key][1], "seed": seed,
        "parameters": record["parameters"], "tree_count": record["tree_count"],
        "validation_auc": validation_auc, "test_auc": test_auc,
        "validation_predictions_identical": True, "test_events": config["test_events"],
        "fit_seconds": fit_seconds, "elapsed_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_result_sha256": digest(source_directory / "result.json"),
        "config_sha256": config_hash, "model_sha256": digest(directory / "model.pickle"),
        "validation_prediction_sha256": digest(directory / "validation_predictions.parquet"),
        "test_prediction_sha256": digest(directory / "test_predictions.parquet"),
    }
    write_json(directory / "result.json", result)
    print(f"COMPLETE {key} seed {seed}: test AUC={test_auc:.6f}; validation reproduced exactly", flush=True)
    return result


def summarize(output):
    import numpy as np
    import pandas as pd

    config = json.loads((output / "config.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(output.glob("fits/*/seed_*/result.json"))]
    expected = len(config["seeds"]) * len(config["models"])
    status = {"status": "complete" if len(records) == expected else "running",
              "completed_fits": len(records), "expected_fits": expected,
              "validation_selected_model": config["selected_model"], "selected_seed": config["selected_seed"]}
    if records:
        frame = pd.DataFrame(records)
        assert not frame.duplicated(["model", "seed"]).any()
        assert frame.config_sha256.eq(digest(output / "config.json")).all()
        assert frame.validation_predictions_identical.all()
        status["completed_by_model"] = frame.groupby("model").size().to_dict()
        frame.drop(columns="parameters").to_csv(output / "test_results.csv", index=False)
        if len(records) == expected:
            matrix = frame.pivot(index="seed", columns="model", values="test_auc").reindex(index=config["seeds"], columns=config["models"])
            assert not matrix.isna().any().any()
            matrix.to_csv(output / "test_auc_by_seed.csv")
            summary = matrix.describe(percentiles=[.025, .25, .5, .75, .975]).T
            summary["win_fraction"] = matrix.eq(matrix.max(axis=1), axis=0).mean()
            summary.index.name = "model"
            summary.sort_values("mean", ascending=False).to_csv(output / "model_distributions.csv")
            draws = np.random.default_rng(42).integers(0, len(matrix), size=(10000, len(matrix)))
            pairs = []
            for i, a in enumerate(config["models"]):
                for b in config["models"][i + 1:]:
                    difference = (matrix[a] - matrix[b]).to_numpy()
                    lo, hi = np.quantile(difference[draws].mean(axis=1), [.025, .975])
                    pairs.append({"model_a": a, "model_b": b, "mean_difference": float(difference.mean()),
                                  "ci_lower": float(lo), "ci_upper": float(hi),
                                  "scope": "Paired seed variation conditional on fixed test events; not event-sampling uncertainty"})
            pd.DataFrame(pairs).to_csv(output / "paired_seed_differences.csv", index=False)
            source_summary = pd.read_csv(Path(config["source"]) / "model_distributions.csv", index_col=0)
            comparison = source_summary[["mean", "std"]].add_prefix("validation_").join(summary[["mean", "std"]].add_prefix("test_"))
            comparison["test_minus_validation_mean"] = comparison.test_mean - comparison.validation_mean
            comparison.to_csv(output / "validation_test_comparison.csv")
            selected = frame.loc[(frame.model == config["selected_model"]) & (frame.seed == config["selected_seed"])].iloc[0]
            status["selected_model_test_auc"] = float(selected.test_auc)
            status["selected_model_validation_auc"] = float(selected.validation_auc)
            selected_dir = output / "fits" / config["selected_model"] / f"seed_{config['selected_seed']:03d}"
            shutil.copy2(selected_dir / "model.pickle", output / "validation_selected_model.pickle")
            plot_distributions(matrix, output, config["selected_model"])
    write_json(output / "status.json", status)
    return status


def plot_distributions(matrix, output, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = matrix.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4.6))
    values = [matrix[key].to_numpy() for key in order]
    labels = [MODELS[key][1] + (" *" if key == selected else "") for key in order]
    ax.boxplot(values, orientation="horizontal", tick_labels=labels, showmeans=True)
    for i, value in enumerate(values, 1):
        ax.scatter(value, i + np.linspace(-.15, .15, len(value)), s=10, alpha=.4)
    ax.invert_yaxis()
    ax.set_xlabel("Test Top@k%AUC (0–20%, normalized)")
    ax.set_title("100 seeds per model · 80 fixed test events\nSaved tuning parameters · 290 training events · * selected on validation", fontsize=10)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"test_seed_distributions.{extension}", dpi=180)
    plt.close(fig)


def run_all(source, output, workers):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(source, output)
        config_hash = digest(output / "config.json")
        (output / "logs").mkdir(exist_ok=True)
        jobs = [(key, seed) for seed in config["seeds"] for key in config["models"]]
        write_json(output / "launch.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                                           "pid": os.getpid(), "workers": workers, "config_sha256": config_hash})

        def launch(job):
            key, seed = job
            directory = output / "fits" / key / f"seed_{seed:03d}"
            if (directory / "result.json").exists():
                verify_completed(directory, config_hash)
                return key, seed
            env = os.environ | {name: "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "POLARS_MAX_THREADS", "PUMPS_AND_DUMPS_MODEL_THREADS")}
            env.update(PYTHONHASHSEED=str(seed), MPLBACKEND="Agg")
            with (output / "logs" / f"{key}_{seed:03d}.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "research_snapshots.test_seed_sweep", "worker",
                                "--output", str(output), "--model", key, "--seed", str(seed)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            return key, seed

        summarize(output)
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(launch, job): job for job in jobs}
            for future in as_completed(pending):
                try:
                    key, seed = future.result()
                    status = summarize(output)
                    print(f"Completed {key} seed {seed}: {status['completed_fits']}/{len(jobs)} fits", flush=True)
                except Exception as exc:
                    errors.append({"job": pending[future], "error": repr(exc)})
                    write_json(output / "errors.json", errors)
                    print(f"ERROR {pending[future]}: {exc}", flush=True)
        status = summarize(output)
        if errors:
            status["status"] = "failed"
            status["errors"] = errors
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
        print(json.dumps(status, indent=2), flush=True)
        if errors:
            raise RuntimeError(f"{len(errors)} fits failed; inspect per-fit logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "worker", "summarize"])
    parser.add_argument("--source", type=Path, default=validation.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=validation.worker_count, default="all")
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "run":
        run_all(args.source.resolve(), args.output.resolve(), args.workers)
    elif args.action == "worker":
        run_one(args.model, args.seed, args.output.resolve())
    else:
        print(json.dumps(summarize(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
