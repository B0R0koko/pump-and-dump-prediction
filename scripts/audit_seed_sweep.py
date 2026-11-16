"""Verify all saved validation seed-sweep results without touching test data."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_snapshots.seed_sweep import DEFAULT_OUTPUT, MODELS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = DEFAULT_OUTPUT
    config = json.loads((output / "config.json").read_text())
    assert config["seeds"] == list(range(1, 101))
    assert config["trials_per_study"] == 25 and config["models"] == list(MODELS)
    assert config["model_threads"] == config["optuna_jobs_per_study"] == 1
    assert not config["test_features_loaded"] and not config["validation_used_for_tuning_or_early_stopping"]
    events = pd.read_csv(output / "events.csv")
    validation_ids = set(events.loc[events.split == "validation", "pump_hash"])
    assert len(validation_ids) == 103
    expected_counts = {"inner_training": 232, "inner_tuning": 58, "validation": 103}
    assert events.split.value_counts().to_dict() == expected_counts
    assert events.groupby("pump_time").split.nunique().eq(1).all()
    config_hash = sha(output / "config.json")
    paths = sorted(output.glob("studies/*/seed_*/result.json"))
    assert len(paths) == 600
    matrix = pd.DataFrame(index=config["seeds"], columns=config["models"], dtype=float)
    bins = np.arange(0, .2001, .005)
    total_trials = 0
    for index, path in enumerate(paths, 1):
        result = json.loads(path.read_text())
        key, seed = result["model"], result["seed"]
        assert key in config["models"] and seed in config["seeds"]
        assert pd.isna(matrix.loc[seed, key]), "Duplicate model/seed"
        assert result["config_sha256"] == config_hash and result["completed_trials"] == 25
        assert result["parameters"].get("random_seed", result["parameters"].get("random_state")) == seed
        assert sha(path.parent / "model.pickle") == result["model_sha256"]
        assert sha(path.parent / "validation_predictions.parquet") == result["prediction_sha256"]
        with sqlite3.connect(f"file:{path.parent / 'studies.db'}?mode=ro", uri=True) as connection:
            states = dict(connection.execute("SELECT state,count(*) FROM trials GROUP BY state"))
            assert states == {"COMPLETE": 25}
            attrs = {k: json.loads(v) for k, v in connection.execute("SELECT key,value_json FROM study_user_attributes")}
            assert attrs["seed"] == seed and attrs["optimization_split"] == "inner_tuning"
            assert attrs["outer_validation_used"] is False
            assert attrs["config_sha256"] == result.get("origin_config_sha256", config_hash)
            best = connection.execute("SELECT value FROM trial_values JOIN trials USING(trial_id) WHERE number=?", (result["best_trial"],)).fetchone()[0]
            np.testing.assert_allclose(best, result["inner_tuning_auc"], rtol=1e-12, atol=1e-12)
        total_trials += 25
        data = pd.read_parquet(path.parent / "validation_predictions.parquet")
        assert set(data.pump_hash) == validation_ids
        assert not data.duplicated(["pump_hash", "currency_pair"]).any()
        assert data.groupby("pump_hash").is_pumped.sum().eq(1).all()
        ordered = data.sort_values("score", ascending=False, kind="stable")
        ordered["rank"] = ordered.groupby("pump_hash", sort=False).cumcount() + 1
        targets = ordered.loc[ordered.is_pumped].set_index("pump_hash")["rank"]
        sizes = data.groupby("pump_hash").size().reindex(targets.index)
        curves = targets.to_numpy()[:, None] <= np.ceil(sizes.to_numpy()[:, None] * bins)
        score = np.trapezoid(curves.astype(float), bins, axis=1).mean() / .2
        np.testing.assert_allclose(score, result["validation_auc"], rtol=1e-12, atol=1e-12)
        matrix.loc[seed, key] = score
        if index % 100 == 0:
            print(f"Verified {index}/600 studies and their saved predictions", flush=True)
    assert total_trials == 15000 and not matrix.isna().any().any()
    recorded = pd.read_csv(output / "validation_auc_by_seed.csv", index_col=0)
    np.testing.assert_allclose(matrix, recorded.reindex(index=matrix.index, columns=matrix.columns), rtol=1e-12, atol=1e-12)
    summary = matrix.describe(percentiles=[.025, .25, .5, .75, .975]).T
    summary["win_fraction"] = matrix.eq(matrix.max(axis=1), axis=0).mean()
    reported_summary = pd.read_csv(output / "model_distributions.csv", index_col=0)
    np.testing.assert_allclose(summary, reported_summary.reindex(summary.index), rtol=1e-12, atol=1e-12)
    draws = np.random.default_rng(42).integers(0, 100, size=(10000, 100))
    pairs = pd.read_csv(output / "paired_seed_differences.csv")
    assert len(pairs) == 15
    for row in pairs.itertuples():
        difference = (matrix[row.model_a] - matrix[row.model_b]).to_numpy()
        np.testing.assert_allclose(row.mean_difference, difference.mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose([row.ci_lower, row.ci_upper], np.quantile(difference[draws].mean(axis=1), [.025, .975]), rtol=1e-12, atol=1e-12)
    selection = json.loads((output / "selection.json").read_text())
    assert selection["model"] == summary["mean"].idxmax() == "catboost"
    assert selection["final_seed"] == 42
    assert sha(output / "selected_model.pickle") == selection["final_seed_result"]["model_sha256"]
    launch = json.loads((output / "launch.json").read_text())
    start = datetime.fromisoformat(launch["started_at"])
    finish = datetime.fromtimestamp(max(path.stat().st_mtime for path in paths), tz=timezone.utc)
    report = {
        "status": "PASS", "experiment_status": "complete", "seeds": [1, 100],
        "models": 6, "completed_studies": len(paths), "completed_trials": total_trials,
        "validation_predictions_replayed": 600, "model_and_prediction_checksums_verified": 600,
        "split_event_counts": expected_counts, "paired_seed_comparisons_replayed": len(pairs),
        "selected_model": selection["model"], "selected_seed": selection["final_seed"],
        "selection": selection, "summary": reported_summary.to_dict("index"),
        "paired_comparisons": pairs.to_dict("records"), "workers": launch["workers"],
        "started_at": start.isoformat(), "last_result_saved_at": finish.isoformat(),
        "elapsed_hours_after_relaunch": (finish - start).total_seconds() / 3600,
        "config_sha256": config_hash, "test_evaluated": False,
        "uncertainty_scope": "Tuning/training seed variation conditional on the fixed validation events",
    }
    (ROOT / "docs/seed_sweep_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all 600 validation scores, 15,000 trials, model files and 15 paired comparisons verified.")


if __name__ == "__main__":
    main()
