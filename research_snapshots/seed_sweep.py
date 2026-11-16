"""Independent seeded searches on inner training events, compared on validation.

Run from the repository root with ``python -m research_snapshots.seed_sweep``.
Each (model, seed) is a separate process and a separate Optuna database. Completed
studies are reused; an interrupted study is archived and restarted from its seed.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks/analysis_outputs/validation_seed_sweep_100x25"
MODELS = {
    "logistic": ("LogisticRegression", "Logistic regression"),
    "forest": ("RandomForest", "Random forest"),
    "catboost": ("CatboostClassifier", "CatBoost"),
    "smote": ("CatboostClassifierSMOTE", "CatBoost + SMOTE"),
    "ranker": ("CatboostRanker", "CatBoost ranker"),
    "catboost_es": ("CatboostClassifierTOPKAUC", "CatBoost + ES"),
}
TRAIN_END = "2020-09-01"
VALIDATION_END = "2021-05-01"
FINAL_SEED = 42  # Chosen in advance, never selected by the largest validation score.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    # The result marker is committed only after the corresponding artifacts.
    if path.name == "result.json":
        for name in ("model.pickle", "validation_predictions.parquet", "trials.csv", "studies.db"):
            artifact = path.parent / name
            if artifact.exists():
                with artifact.open("rb") as stream:
                    os.fsync(stream.fileno())
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def worker_count(value):
    count = available_cpus() if value == "all" else int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("Workers must be positive or 'all'")
    return count


def fit_source_signature(source):
    """Identify numerical code independently of scheduling and file persistence."""
    names = {"inner_boundary", "configure_model", "build_samples", "run_study"}
    constants = {"MODELS", "TRAIN_END", "VALIDATION_END", "FINAL_SEED"}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            nodes.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(ast.dump(node, include_attributes=False))
    if len(nodes) != len(names) + len(constants):
        raise ValueError("Incomplete model-fitting source fingerprint")
    return hashlib.sha256("\n".join(nodes).encode()).hexdigest()


def import_completed(source, output):
    """Adopt unchanged studies after a seed-schedule change, retaining provenance."""
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source / "config.json").read_text())
    new = json.loads((output / "config.json").read_text())
    excluded = {"seeds", "runner_sha256"}
    if {k: v for k, v in old.items() if k not in excluded} != {k: v for k, v in new.items() if k not in excluded}:
        raise ValueError("Study protocol, inputs or software versions differ")
    snapshot = source / "runner_snapshot.py"
    if digest(snapshot) != old["runner_sha256"]:
        raise ValueError("Source runner snapshot does not match its recorded hash")
    signature = fit_source_signature(Path(__file__).read_text())
    if fit_source_signature(snapshot.read_text()) != signature:
        raise ValueError("Model-fitting code differs")
    if json.loads((source / "data.json").read_text()) != json.loads((output / "data.json").read_text()):
        raise ValueError("Prepared data or inner split differs")
    imported = []
    for result_file in sorted(source.glob("studies/*/seed_*/result.json")):
        result = json.loads(result_file.read_text())
        key, seed = result["model"], result["seed"]
        if seed not in new["seeds"]:
            continue
        assert key in new["models"] and result["completed_trials"] == new["trials_per_study"]
        assert result["config_sha256"] == digest(source / "config.json")
        assert result["model_sha256"] == digest(result_file.parent / "model.pickle")
        assert result["prediction_sha256"] == digest(result_file.parent / "validation_predictions.parquet")
        with sqlite3.connect(f"file:{result_file.parent / 'studies.db'}?mode=ro", uri=True) as connection:
            assert connection.execute("SELECT count(*) FROM trials WHERE state='COMPLETE'").fetchone()[0] == new["trials_per_study"]
        destination = output / "studies" / key / f"seed_{seed:03d}"
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing study: {destination}")
        shutil.copytree(result_file.parent, destination)
        provenance = destination / "origin"
        provenance.mkdir()
        for name in ("config.json", "data.json", "runner_snapshot.py"):
            shutil.copy2(source / name, provenance / name)
        shutil.copy2(result_file, provenance / "result.json")
        result["origin_config_sha256"] = result["config_sha256"]
        result["config_sha256"] = digest(output / "config.json")
        result["imported_from"] = str(result_file.parent)
        result["fit_source_sha256"] = signature
        write_json(destination / "result.json", result)
        imported.append({"model": key, "seed": seed, "model_sha256": result["model_sha256"],
                         "prediction_sha256": result["prediction_sha256"]})
    report = {"source": str(source), "destination": str(output), "fit_source_sha256": signature,
              "imported": imported, "imported_at": datetime.now(timezone.utc).isoformat()}
    reports = output / "imports"
    reports.mkdir(exist_ok=True)
    write_json(reports / f"{source.name}.json", report)
    return report


def inner_boundary(events, fraction=0.2):
    """Closest chronological 80/20 cut without splitting simultaneous events."""
    counts = events.groupby("pump_time", sort=True).size()
    if len(counts) < 2:
        raise ValueError("At least two training-event timestamps are required")
    distances = (counts.cumsum().iloc[:-1] - len(events) * (1 - fraction)).abs()
    return counts.index[int(distances.argmin()) + 1]


def prepare(output, seeds, trials, workers):
    import pandas as pd
    from backtest.utils.build_dataset import load_pumps, _read_cross_section, FEATURE_DIR
    from research_snapshots.reproducibility import reproducibility_settings

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "research_snapshots/paper_0167ddd/pumps.json"
    manifest = load_pumps(manifest_path)
    events = [p for p in manifest if p.time < pd.Timestamp(VALIDATION_END)]
    # Held-out test feature files are never opened by this experiment.
    feature_paths = [FEATURE_DIR / "pumps" / f"{p}.parquet" for p in events]
    inputs = {str(p): digest(p) for p in feature_paths}
    config = {
        "protocol": "paper_290_train_103_validation_inner_chronological_80_20_v1",
        "seeds": seeds, "trials_per_study": trials, "models": list(MODELS),
        "model_threads": 1, "optuna_jobs_per_study": 1,
        "sampler": "TPESampler", "selection": "Highest mean validation Top@k%AUC across all seeds; ties: lower SD, then model key",
        "final_model_seed": FINAL_SEED, "validation_used_for_tuning_or_early_stopping": False,
        "test_features_loaded": False, "test_used_for_selection": False,
        "manifest_sha256": digest(manifest_path), "feature_sha256": inputs,
        "runtime_sha256": digest(ROOT / "research_snapshots/73ba716/runtime.zip"),
        "runner_sha256": digest(__file__), "versions": reproducibility_settings()["versions"],
        "auc": {"range": [0, 0.2], "step": 0.005, "normalization": 0.2},
    }
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing experiment has different inputs or settings; choose a new output directory")
        cached = json.loads((output / "data.json").read_text())
        if digest(output / "development.parquet") != cached["development_sha256"]:
            raise ValueError("Cached development data changed")
        return config
    frames = []
    for event in events:
        _, frame = _read_cross_section(event)
        if frame is None:
            raise ValueError(f"Missing requested event: {event}")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["is_pumped"] = raw.currency_pair == raw.pumped_currency_pair
    event_frame = raw[["pump_hash", "pump_time"]].drop_duplicates()
    training = event_frame.loc[event_frame.pump_time < pd.Timestamp(TRAIN_END)]
    validation = event_frame.loc[event_frame.pump_time >= pd.Timestamp(TRAIN_END)]
    assert len(training) == 290 and len(validation) == 103
    cutoff = inner_boundary(training)
    event_frame["split"] = "validation"
    event_frame.loc[event_frame.pump_time < pd.Timestamp(TRAIN_END), "split"] = "inner_tuning"
    event_frame.loc[event_frame.pump_time < cutoff, "split"] = "inner_training"
    event_frame.to_csv(output / "events.csv", index=False)
    raw.to_parquet(output / "development.parquet", index=False)
    write_json(output / "data.json", {
        "inner_cutoff": cutoff.isoformat(), "events": event_frame.split.value_counts().to_dict(),
        "development_sha256": digest(output / "development.parquet"),
        "event_manifest_sha256": digest(output / "events.csv"),
    })
    write_json(config_path, config)
    shutil.copy2(__file__, output / "runner_snapshot.py")
    return config


def configure_model(key, seed):
    import random
    import numpy as np
    from research_snapshots.reproducibility import initialize_reproducibility

    initialize_reproducibility()
    random.seed(seed)
    np.random.seed(seed)
    module_name, _ = MODELS[key]
    module = importlib.import_module(f"backtest.pipelines.{module_name}.pipe")
    module.RANDOM_SEED = seed  # Also controls SMOTE's independent fit_resample.
    module._BASE_PARAMS = dict(module._BASE_PARAMS)
    module._BASE_PARAMS["random_seed" if key not in ("logistic", "forest") else "random_state"] = seed
    module._BASE_PARAMS["verbose"] = False
    if key == "forest":
        module._BASE_PARAMS["n_jobs"] = 1
    elif key != "logistic":
        module._BASE_PARAMS.update(thread_count=1, task_type="CPU", allow_writing_files=False)
    return module, getattr(module, module_name + "Pipeline")()


def build_samples(pipeline, raw, cutoff, phase):
    """Fit imputation priors on the phase's training events, never outer validation."""
    import pandas as pd
    import backtest.pipelines.BasePipeline as base
    from backtest.utils.sample import DatasetType

    training_end = pd.Timestamp(cutoff if phase == "inner" else TRAIN_END)
    source = raw.loc[raw.pump_time < pd.Timestamp(TRAIN_END)].copy() if phase == "inner" else raw.copy()
    previous = base.TRAIN_END
    try:
        base.TRAIN_END = training_end
        data = pipeline.preprocess_data(source)
    finally:
        base.TRAIN_END = previous
    frames = {
        DatasetType.TRAIN: data.loc[data.pump_time < training_end].copy(),
        DatasetType.VALIDATION: data.loc[data.pump_time >= training_end].copy(),
    }
    original = pipeline.build_datasets
    try:
        pipeline.build_datasets = lambda: {k: v.copy(deep=True) for k, v in frames.items()}
        return pipeline.create_sample()
    finally:
        pipeline.build_datasets = original


def run_study(key, seed, output):
    import logging
    import numpy as np
    import optuna
    import pandas as pd
    from backtest.utils.sample import DatasetType
    from backtest.utils.metrics import calculate_topk_percent_auc

    logging.getLogger().setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = json.loads((output / "config.json").read_text())
    config_hash = digest(output / "config.json")
    directory = output / "studies" / key / f"seed_{seed:03d}"
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["config_sha256"] == config_hash
        assert digest(directory / "model.pickle") == result["model_sha256"]
        return result
    if directory.exists():
        archive = output / "interrupted" / key
        archive.mkdir(parents=True, exist_ok=True)
        directory.rename(archive / f"seed_{seed:03d}_{time.time_ns()}")
    directory.mkdir(parents=True)
    started = time.monotonic()
    module, pipeline = configure_model(key, seed)
    raw = pd.read_parquet(output / "development.parquet")
    metadata = json.loads((output / "data.json").read_text())
    inner = build_samples(pipeline, raw, metadata["inner_cutoff"], "inner")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f"sqlite:///{directory / 'studies.db'}", study_name=f"{key}_seed_{seed}",
    )
    study.set_user_attr("config_sha256", config_hash)
    study.set_user_attr("seed", seed)
    study.set_user_attr("optimization_split", "inner_tuning")
    study.set_user_attr("outer_validation_used", False)
    symbol = "LogisticRegressionModel" if key == "logistic" else "RandomForestModel" if key == "forest" else "CatboostRankerModel" if key == "ranker" else "CatboostClassifierModel"
    original_model_class = getattr(module, symbol)

    def objective(trial):
        # Reuse the paper's objective/search space, recording the fitted tree
        # count so the final fit never needs validation-controlled stopping.
        class RecordedModel(original_model_class):
            def train(self, sample):
                super().train(sample)
                params = {k: v for k, v in self.params.items() if k != "eval_metric"}
                trial.set_user_attr("model_parameters", params)
                trial.set_user_attr("tree_count", getattr(self._model, "tree_count_", None))
                return self

        setattr(module, symbol, RecordedModel)
        try:
            return module._objective(trial, inner)
        finally:
            setattr(module, symbol, original_model_class)

    def progress(study, trial):
        print(f"{key} seed {seed}: trial {trial.number + 1}/{config['trials_per_study']} AUC={trial.value:.5f}", flush=True)
        write_json(directory / "progress.json", {"completed_trials": len(study.trials), "elapsed_seconds": time.monotonic() - started})

    study.optimize(objective, n_trials=config["trials_per_study"], n_jobs=1, callbacks=[progress])
    assert len(study.trials) == config["trials_per_study"]
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    search_seconds = time.monotonic() - started
    best = study.best_trial
    params = best.user_attrs["model_parameters"].copy()
    if isinstance(params.get("class_weight"), dict):
        params["class_weight"] = {int(k): v for k, v in params["class_weight"].items()}
    full = build_samples(pipeline, raw, metadata["inner_cutoff"], "full")
    model = original_model_class(params)
    if key in ("logistic", "forest"):
        model.train(full)
    else:
        from catboost import CatBoostClassifier, CatBoostRanker

        for parameter in ("iterations", "num_boost_round", "eval_metric"):
            params.pop(parameter, None)
        params["iterations"] = int(best.user_attrs["tree_count"])
        params["use_best_model"] = False
        model.params = params
        model._model = (CatBoostRanker if key == "ranker" else CatBoostClassifier)(**params)
        model._model.fit(full.get_pool(DatasetType.TRAIN))  # No eval_set.
    validation = full.get_dataset(DatasetType.VALIDATION)
    score = float(calculate_topk_percent_auc(model, validation))
    predictions = validation.all_data()[["pump_hash", "pump_time", "currency_pair", "is_pumped"]].copy()
    predictions["score"] = model.rank(validation)
    predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
    with (directory / "model.pickle").open("wb") as stream:
        pickle.dump(model, stream)
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
    result = {
        "model": key, "label": MODELS[key][1], "seed": seed,
        "completed_trials": len(study.trials), "best_trial": best.number,
        "inner_tuning_auc": float(best.value), "validation_auc": score,
        "parameters": params, "tree_count": best.user_attrs["tree_count"],
        "search_seconds": search_seconds, "elapsed_seconds": time.monotonic() - started,
        "trial_seconds": [t.duration.total_seconds() for t in study.trials],
        "validation_events": int(predictions.pump_hash.nunique()),
        "config_sha256": config_hash, "model_sha256": digest(directory / "model.pickle"),
        "prediction_sha256": digest(directory / "validation_predictions.parquet"),
    }
    write_json(result_path, result)
    print(f"COMPLETE {key} seed {seed}: validation AUC={score:.6f}; {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def summarize(output, workers):
    import numpy as np
    import pandas as pd

    config = json.loads((output / "config.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((output / "studies").glob("*/seed_*/result.json"))]
    expected = len(config["seeds"]) * len(MODELS)
    status = {"completed_studies": len(records), "expected_studies": expected, "workers": workers}
    if not records:
        write_json(output / "status.json", status)
        return status
    frame = pd.DataFrame(records)
    assert set(frame.seed).issubset(config["seeds"])
    assert set(frame.model).issubset(MODELS)
    assert frame.completed_trials.eq(config["trials_per_study"]).all()
    assert frame.config_sha256.eq(digest(output / "config.json")).all()
    assert not frame.duplicated(["seed", "model"]).any()
    frame.drop(columns=["parameters", "trial_seconds"]).to_csv(output / "validation_results.csv", index=False)
    timing = frame.groupby("model").elapsed_seconds.agg(["count", "mean", "min", "max"])
    timing.to_csv(output / "timing_by_model.csv")
    if set(timing.index) == set(MODELS):
        serial = float(timing["mean"].sum() * len(config["seeds"]))
        remaining = sum(max(0, len(config["seeds"]) - int(row["count"])) * row["mean"] for _, row in timing.iterrows())
        status.update(estimated_serial_hours=serial / 3600,
                      estimated_parallel_hours=serial / workers / 3600,
                      estimated_remaining_hours=remaining / workers / 3600)
    if len(records) == expected:
        assert not frame.duplicated(["seed", "model"]).any()
        matrix = frame.pivot(index="seed", columns="model", values="validation_auc").reindex(config["seeds"])
        assert not matrix.isna().any().any()
        matrix.to_csv(output / "validation_auc_by_seed.csv")
        summary = matrix.describe(percentiles=[.025, .25, .5, .75, .975]).T
        summary["win_fraction"] = matrix.eq(matrix.max(axis=1), axis=0).mean()
        summary.index.name = "model"
        summary = summary.reset_index().sort_values(["mean", "std", "model"], ascending=[False, True, True]).set_index("model")
        summary.to_csv(output / "model_distributions.csv")
        if len(config["seeds"]) >= 2:
            rng = np.random.default_rng(42)
            draws = rng.integers(0, len(matrix), size=(10000, len(matrix)))
            pairs = []
            for i, a in enumerate(MODELS):
                for b in list(MODELS)[i + 1:]:
                    difference = (matrix[a] - matrix[b]).to_numpy()
                    lo, hi = np.quantile(difference[draws].mean(axis=1), [.025, .975])
                    pairs.append({"model_a": a, "model_b": b, "mean_difference": float(difference.mean()),
                                  "ci_lower": float(lo), "ci_upper": float(hi),
                                  "scope": "Paired seed variation conditional on these fixed events; not event-sampling uncertainty"})
            pd.DataFrame(pairs).to_csv(output / "paired_seed_differences.csv", index=False)
        if len(config["seeds"]) == 100 and config["trials_per_study"] == 25:
            winner = str(summary.index[0])
            selected = output / "studies" / winner / f"seed_{FINAL_SEED:03d}"
            assert FINAL_SEED in config["seeds"]
            shutil.copy2(selected / "model.pickle", output / "selected_model.pickle")
            write_json(output / "selection.json", {
                "model": winner, "label": MODELS[winner][1], "rule": config["selection"],
                "mean_validation_auc": float(summary.loc[winner, "mean"]),
                "final_seed": FINAL_SEED, "final_seed_result": json.loads((selected / "result.json").read_text()),
            })
            status["selected_model"] = winner
        plot_distributions(matrix, output, config["trials_per_study"])
    status["status"] = "complete" if len(records) == expected else "running"
    write_json(output / "status.json", status)
    return status


def plot_distributions(matrix, output, trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = matrix.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [matrix[key].to_numpy() for key in order]
    ax.boxplot(values, orientation="horizontal", tick_labels=[MODELS[k][1] for k in order], showmeans=True)
    for i, value in enumerate(values, 1):
        offsets = np.linspace(-.15, .15, len(value))
        ax.scatter(value, i + offsets, s=12, alpha=.45)
    ax.invert_yaxis()
    ax.set_xlabel("Validation Top@k%AUC (0–20%, normalized)")
    ax.set_title(f"{len(matrix)} seeds per model · {trials}-trial inner searches\nFinal fits on all training events; fixed outer validation events")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "validation_seed_distributions.png", dpi=180)
    fig.savefig(output / "validation_seed_distributions.pdf")
    plt.close(fig)


def run_all(output, workers):
    config = json.loads((output / "config.json").read_text())
    (output / "logs").mkdir(exist_ok=True)
    # Seed-major order exposes all model timings early and keeps progress balanced.
    jobs = [(key, seed) for seed in config["seeds"] for key in MODELS]

    def launch(job):
        key, seed = job
        path = output / "studies" / key / f"seed_{seed:03d}" / "result.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record["config_sha256"] == digest(output / "config.json")
            assert record["completed_trials"] == config["trials_per_study"]
            assert record["model_sha256"] == digest(path.parent / "model.pickle")
            assert record["prediction_sha256"] == digest(path.parent / "validation_predictions.parquet")
            return key, seed
        env = os.environ | {name: "1" for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "POLARS_MAX_THREADS", "PUMPS_AND_DUMPS_MODEL_THREADS")}
        env["PYTHONHASHSEED"] = str(seed)
        with (output / "logs" / f"{key}_{seed:03d}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "research_snapshots.seed_sweep", "worker",
                            "--output", str(output), "--model", key, "--seed", str(seed)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return key, seed

    write_json(output / "status.json", {"status": "running", "pid": os.getpid(), "workers": workers,
                                       "started_at": datetime.now(timezone.utc).isoformat(),
                                       "expected_studies": len(jobs)})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(launch, job): job for job in jobs}
        for future in as_completed(pending):
            key, seed = future.result()
            status = summarize(output, workers)
            print(f"Completed {key} seed {seed}: {status['completed_studies']}/{status['expected_studies']} studies", flush=True)
    print(json.dumps(summarize(output, workers), indent=2), flush=True)


def estimate_runtime(output, workers, seeds=100, trials=25):
    """Project measured trial costs; the range is a planning buffer, not a CI."""
    import numpy as np

    rows = []
    for key in MODELS:
        durations, refits = [], []
        for database in (output / "studies" / key).glob("seed_*/studies.db"):
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                records = connection.execute("SELECT datetime_start, datetime_complete FROM trials WHERE state='COMPLETE'").fetchall()
            durations.extend((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
                             for start, end in records)
            result_file = database.parent / "result.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                refits.append(result["elapsed_seconds"] - result["search_seconds"])
        if not durations:
            rows.append({"model": key, "completed_pilot_trials": 0})
            continue
        mean = float(np.mean(durations))
        refit = float(np.mean(refits)) if refits else mean
        rows.append({"model": key, "completed_pilot_trials": len(durations),
                     "mean_trial_seconds": mean, "median_trial_seconds": float(np.median(durations)),
                     "maximum_trial_seconds": max(durations), "estimated_final_fit_seconds": refit,
                     "projected_cpu_hours": seeds * (trials * mean + refit) / 3600})
    report = {"measured_at": datetime.now(timezone.utc).isoformat(), "target_seeds": seeds,
              "target_trials": trials, "workers": workers, "models": rows,
              "caveat": "Pilot-based projection. Search paths, tree depths and CPU contention change runtime; the range is a planning buffer, not a statistical interval."}
    if all(row["completed_pilot_trials"] for row in rows):
        serial = sum(row["projected_cpu_hours"] for row in rows)
        ideal = serial / workers
        report.update(projected_serial_hours=serial, ideal_parallel_hours=ideal,
                      planning_range_hours=[1.15 * ideal, 2.0 * ideal])
    write_json(output / "runtime_estimate.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "worker", "summarize", "estimate", "import-completed"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--workers", type=worker_count, default="all")
    parser.add_argument("--source", type=Path, help="Completed experiment to import after verifying unchanged fits")
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if min(args.seeds, args.trials, args.workers) < 1 or args.start_seed < 0:
        parser.error("Seed count, trial count and worker count must be positive; start seed must be nonnegative")
    output = args.output.resolve()
    if args.action == "worker":
        if args.model is None or args.seed is None:
            parser.error("worker needs --model and --seed")
        run_study(args.model, args.seed, output)
    elif args.action == "summarize":
        print(json.dumps(summarize(output, args.workers), indent=2))
    elif args.action == "estimate":
        print(json.dumps(estimate_runtime(output, args.workers, args.seeds, args.trials), indent=2))
    elif args.action == "import-completed":
        if args.source is None:
            parser.error("import-completed needs --source")
        seeds = list(range(args.start_seed, args.start_seed + args.seeds))
        prepare(output, seeds, args.trials, args.workers)
        print(json.dumps(import_completed(args.source, output), indent=2))
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.seeds))
        if args.seeds == 100 and FINAL_SEED not in seeds:
            parser.error("The 100-seed experiment must include the preselected final-model seed 42")
        prepare(output, seeds, args.trials, args.workers)
        run_all(output, args.workers)


if __name__ == "__main__":
    main()
