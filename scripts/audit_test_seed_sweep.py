"""Replay test metrics, compare saved-model predictions, and verify seed provenance."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_snapshots import seed_sweep as validation
from research_snapshots import test_seed_sweep as evaluation


def audit(output):
    digest = validation.digest
    config = json.loads((output / "config.json").read_text())
    config_hash = digest(output / "config.json")
    source = Path(config["source"])
    assert config["seeds"] == list(range(1, 101)) and config["models"] == list(validation.MODELS)
    assert config["new_search_trials"] == 0 and config["test_used_for_selection"] is False
    assert digest(source / "config.json") == config["source_config_sha256"]
    assert digest(source / "selection.json") == config["source_selection_sha256"]
    assert digest(output / "validation_selection.json") == config["source_selection_sha256"]
    assert digest(source / "development.parquet") == config["development_sha256"]
    assert digest(output / "runner_snapshot.py") == config["runner_sha256"]
    assert digest(evaluation.__file__) == config["runner_sha256"]
    assert digest(validation.__file__) == config["validation_runner_sha256"]
    assert digest(ROOT / "research_snapshots/73ba716/runtime.zip") == config["runtime_sha256"]
    data = json.loads((output / "data.json").read_text())
    assert digest(output / "test.parquet") == data["test_sha256"]
    assert digest(output / "events.csv") == data["events_sha256"]
    for path, checksum in config["test_feature_sha256"].items():
        assert digest(path) == checksum
    development = pd.read_parquet(source / "development.parquet")
    raw_test = pd.read_parquet(output / "test.parquet")
    test_events = pd.read_csv(output / "events.csv")
    test_ids = set(test_events.pump_hash)
    assert len(test_ids) == 80 and test_ids.isdisjoint(development.pump_hash)
    assert development.loc[development.pump_time < pd.Timestamp(validation.TRAIN_END)].pump_hash.nunique() == 290
    assert development.loc[development.pump_time >= pd.Timestamp(validation.TRAIN_END)].pump_hash.nunique() == 103
    assert raw_test.pump_time.ge(pd.Timestamp(validation.VALIDATION_END)).all()
    paths = sorted(output.glob("fits/*/seed_*/result.json"))
    assert len(paths) == 600
    matrix = pd.DataFrame(index=config["seeds"], columns=config["models"], dtype=float)
    datasets = {}
    bins = np.arange(0, .2001, .005)
    event_metrics = []
    for index, path in enumerate(paths, 1):
        result = evaluation.verify_completed(path.parent, config_hash)
        key, seed = result["model"], result["seed"]
        assert key in config["models"] and seed in config["seeds"] and pd.isna(matrix.loc[seed, key])
        original_dir = source / "studies" / key / f"seed_{seed:03d}"
        expected_hash = config["source_result_sha256"][f"{key}/{seed:03d}"]
        assert result["source_result_sha256"] == digest(original_dir / "result.json") == expected_hash
        original = json.loads((original_dir / "result.json").read_text())
        assert original["parameters"] == result["parameters"]
        assert original["tree_count"] == result["tree_count"]
        assert result["parameters"].get("random_seed", result["parameters"].get("random_state")) == seed
        assert result["validation_auc"] == original["validation_auc"]
        assert digest(original_dir / "model.pickle") == original["model_sha256"]
        assert digest(original_dir / "validation_predictions.parquet") == original["prediction_sha256"]
        previous_predictions = pd.read_parquet(original_dir / "validation_predictions.parquet").reset_index(drop=True)
        new_predictions = pd.read_parquet(path.parent / "validation_predictions.parquet")
        pd.testing.assert_frame_equal(previous_predictions, new_predictions, check_exact=True)
        predictions = pd.read_parquet(path.parent / "test_predictions.parquet")
        assert set(predictions.pump_hash) == test_ids
        assert not predictions.duplicated(["pump_hash", "currency_pair"]).any()
        assert predictions.groupby("pump_hash").is_pumped.sum().eq(1).all()
        assert np.isfinite(predictions.score).all()
        if key not in datasets:
            _, pipeline = validation.configure_model(key, 42)
            datasets[key] = evaluation.build_test_dataset(key, pipeline, development, raw_test)
        dataset = datasets[key]
        pd.testing.assert_frame_equal(predictions[evaluation.PREDICTION_COLUMNS],
                                      dataset.all_data()[evaluation.PREDICTION_COLUMNS].reset_index(drop=True))
        # Reload both checkpoints. Equality on every test asset verifies that the
        # new fits preserve the original models beyond the validation sample.
        for model_path in (path.parent / "model.pickle", original_dir / "model.pickle"):
            with model_path.open("rb") as stream:
                model = pickle.load(stream)
            np.testing.assert_array_equal(model.rank(dataset), predictions.score.to_numpy())
        ordered = predictions.sort_values("score", ascending=False, kind="stable")
        ordered["rank"] = ordered.groupby("pump_hash", sort=False).cumcount() + 1
        target_ranks = ordered.loc[ordered.is_pumped].set_index("pump_hash")["rank"]
        sizes = predictions.groupby("pump_hash").size().reindex(target_ranks.index)
        curves = target_ranks.to_numpy()[:, None] <= np.ceil(sizes.to_numpy()[:, None] * bins)
        aucs = np.trapezoid(curves.astype(float), bins, axis=1) / .2
        np.testing.assert_allclose(aucs.mean(), result["test_auc"], rtol=1e-12, atol=1e-12)
        # After independent verification, preserve the reported floating-point
        # values when replaying exact ties in the win-fraction calculation.
        matrix.loc[seed, key] = result["test_auc"]
        event_metrics.append(pd.DataFrame({"model": key, "seed": seed, "pump_hash": target_ranks.index,
                                           "target_rank": target_ranks.to_numpy(), "eligible_assets": sizes.to_numpy(),
                                           "topkp_auc": aucs}))
        if index % 100 == 0:
            print(f"Verified {index}/600 fits, source checkpoints and test predictions", flush=True)
    recorded = pd.read_csv(output / "test_auc_by_seed.csv", index_col=0)
    np.testing.assert_allclose(matrix, recorded.reindex(index=matrix.index, columns=matrix.columns), rtol=1e-12, atol=1e-12)
    summary = matrix.describe(percentiles=[.025, .25, .5, .75, .975]).T
    summary["win_fraction"] = matrix.eq(matrix.max(axis=1), axis=0).mean()
    reported = pd.read_csv(output / "model_distributions.csv", index_col=0)
    np.testing.assert_allclose(summary, reported.reindex(summary.index), rtol=1e-12, atol=1e-12)
    draws = np.random.default_rng(42).integers(0, 100, size=(10000, 100))
    pairs = pd.read_csv(output / "paired_seed_differences.csv")
    assert len(pairs) == 15
    for row in pairs.itertuples():
        difference = (matrix[row.model_a] - matrix[row.model_b]).to_numpy()
        np.testing.assert_allclose(row.mean_difference, difference.mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose([row.ci_lower, row.ci_upper], np.quantile(difference[draws].mean(axis=1), [.025, .975]), rtol=1e-12, atol=1e-12)
    selection = json.loads((source / "selection.json").read_text())
    assert config["selected_model"] == selection["model"]
    assert config["selected_seed"] == selection["final_seed"] == 42
    selected_dir = output / "fits" / selection["model"] / "seed_042"
    assert digest(output / "validation_selected_model.pickle") == digest(selected_dir / "model.pickle")
    selected = json.loads((selected_dir / "result.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    start = datetime.fromisoformat(launch["started_at"])
    finish = max(datetime.fromisoformat(json.loads(path.read_text())["completed_at"]) for path in paths)
    pd.concat(event_metrics, ignore_index=True).to_parquet(output / "test_event_metrics.parquet", index=False)
    report = {
        "status": "PASS", "completed_fits": 600, "seeds": [1, 100], "models": 6,
        "training_events": 290, "validation_events": 103, "test_events": 80,
        "new_search_trials": 0, "test_used_for_selection": False,
        "validation_predictions_exactly_reproduced": 600,
        "test_predictions_reproduced_by_new_and_source_checkpoints": 600,
        "test_metrics_independently_replayed": 600, "paired_seed_comparisons_replayed": 15,
        "selected_model": selection["model"], "selected_seed": 42,
        "selected_model_test_auc": selected["test_auc"],
        "selected_model_validation_auc": selected["validation_auc"],
        "summary": reported.to_dict("index"), "paired_comparisons": pairs.to_dict("records"),
        "config_sha256": config_hash, "source_config_sha256": config["source_config_sha256"],
        "workers": launch["workers"], "started_at": start.isoformat(), "last_result_saved_at": finish.isoformat(),
        "elapsed_minutes": (finish - start).total_seconds() / 60,
        "uncertainty_scope": "Tuning/training seed variation conditional on the fixed test events; not event-sampling uncertainty",
    }
    validation.write_json(output / "audit.json", report)
    validation.write_json(ROOT / "docs/test_seed_sweep_results.json", report)
    print("PASS: all 600 refits, test predictions, AUC scores and paired-seed intervals verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=evaluation.DEFAULT_OUTPUT)
    audit(parser.parse_args().output.resolve())
