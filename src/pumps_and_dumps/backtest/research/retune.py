"""Run fresh, isolated model searches using only the chronological tuning split."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import time


# CPU budgets keep independent searches within this machine's capacity.
CANDIDATES = {
    "logistic_regression": ("LogisticRegressionPipeline", "Logistic regression", 1),
    "random_forest": ("RandomForestPipeline", "Random forest", 8),
    "catboost_classifier": ("CatboostClassifierPipeline", "CatBoost classifier", 1),
    "catboost_smote": ("CatboostClassifierSMOTEPipeline", "CatBoost classifier + SMOTE", 8),
    "catboost_ranker": ("CatboostRankerPipeline", "CatBoost ranker", 4),
    "catboost_topk_auc": ("CatboostClassifierTOPKAUCPipeline", "CatBoost classifier + Top@k%AUC early stopping", 1),
    "catboost_top5": ("CatboostTop5Pipeline", "CatBoost + Top-5", 4),
}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _split_manifest(sample) -> dict:
    result = {}
    for split, dataset in sample.iter_datasets():
        data = dataset.all_data()
        events = data[["pump_hash", "pump_time"]].drop_duplicates() if "pump_hash" in data else None
        result[split.name] = {
            "rows": len(data),
            "events": None if events is None else len(events),
            "start": None if events is None else events.pump_time.min().isoformat(),
            "end": None if events is None else events.pump_time.max().isoformat(),
            "event_ids": None if events is None else sorted(events.pump_hash.tolist()),
        }
    return result


def run_candidate(key: str, storage: str, output: Path, trials: int, threads: int) -> None:
    import optuna

    from pumps_and_dumps.backtest.datasets.sample import DatasetType
    from pumps_and_dumps.backtest.pipelines.base_pipeline import MODEL_SELECTION_SPLIT_POLICY

    module = importlib.import_module(f"pumps_and_dumps.backtest.pipelines.{key}.pipeline")
    class_name, display_name, _ = CANDIDATES[key]
    if key == "random_forest":
        module._BASE_PARAMS["n_jobs"] = threads
    elif key.startswith("catboost"):
        module._BASE_PARAMS["thread_count"] = threads
        if key != "catboost_top5":
            module._BASE_PARAMS["verbose"] = False
    pipeline = getattr(module, class_name)(study_storage=storage)
    started = datetime.now(timezone.utc).isoformat()
    print(f"Starting {display_name}: {trials} fresh trials, {threads} threads", flush=True)
    study = pipeline.optimize_parameters(n_trials=trials)
    complete = [t for t in study.trials if t.state is optuna.trial.TrialState.COMPLETE]
    if len(complete) != trials:
        raise RuntimeError(f"{display_name}: only {len(complete)}/{trials} trials completed")
    sample = pipeline.create_sample()
    splits = _split_manifest(sample)
    for name, value in {
        "split_policy": MODEL_SELECTION_SPLIT_POLICY,
        "optimization_dataset": DatasetType.TUNING.name,
        "early_stopping_dataset": DatasetType.TUNING.name,
        "split_manifest": splits,
    }.items():
        study.set_user_attr(name, value)
    model = pipeline.train(sample, tuned=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    with (model_dir / f"{class_name}.pickle").open("wb") as stream:
        pickle.dump(model, stream)
    metadata = {
        "status": "complete", "model": display_name, "pipeline": class_name,
        "study_name": study.study_name, "storage": storage,
        "completed_trials": len(complete), "best_trial": study.best_trial.number,
        "best_value": study.best_value, "best_parameters": study.best_params,
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "splits": splits, "regressors": pipeline.feature_set.regressors,
        "threads": threads,
    }
    _write_json(output / f"{key}.json", metadata)
    print(f"Completed {display_name}; best trial {study.best_trial.number}: {study.best_value}", flush=True)


def run_searches(storage: str, output: Path, trials: int) -> None:
    import optuna

    output.mkdir(parents=True, exist_ok=True)
    database = Path(storage.removeprefix("sqlite:///"))
    if database.exists():
        raise FileExistsError(f"Fresh searches require a new study database: {database}")
    database.parent.mkdir(parents=True, exist_ok=True)
    optuna.storages.RDBStorage(storage)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    processes = {}
    cpu_count = len(os.sched_getaffinity(0))
    scale = min(1.0, cpu_count / sum(item[2] for item in CANDIDATES.values()))
    for key, (_, _, budget) in CANDIDATES.items():
        threads = max(1, int(budget * scale))
        env = os.environ | {name: str(threads) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"
        )}
        command = [sys.executable, "-m", "pumps_and_dumps.backtest.research.retune", "--candidate", key, "--storage", storage,
                   "--output-dir", str(output), "--trials", str(trials), "--threads", str(threads)]
        log = (output / f"{key}.log").open("w")
        processes[key] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
    while processes:
        for key, process in list(processes.items()):
            if process.poll() is None:
                continue
            print(f"{key}: exit {process.returncode}", flush=True)
            if process.returncode:
                raise RuntimeError(f"Search failed for {key}; inspect {output / (key + '.log')}")
            del processes[key]
        time.sleep(1)
    _write_json(output / "complete.json", {"models": len(CANDIDATES), "trials_per_model": trials,
                                          "storage": storage, "completed_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    from pumps_and_dumps.core.paths import SQLITE_URL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=SQLITE_URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error("trials and threads must be positive")
    if args.candidate:
        run_candidate(args.candidate, args.storage, args.output_dir, args.trials, args.threads)
    else:
        run_searches(args.storage, args.output_dir, args.trials)


if __name__ == "__main__":
    main()
