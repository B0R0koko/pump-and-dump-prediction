"""Check the historical TEST bridge, paired resampling, and executable notebook export."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest.utils.experiment import Experiment
from backtest.utils.sample import DatasetType
from pumps_and_dumps.backtest.robustness.auc_bootstrap import evaluate_test_auc_bootstrap
from pumps_and_dumps.backtest.robustness.significance import compare_models_bootstrap_topk_percent_auc

from tests.notebook_helpers import analysis
from research_snapshots import input_provenance
from research_snapshots.reproducibility import reproducibility_settings

ROOT = Path(__file__).resolve().parents[3]


def experiments():
    rows = []
    # Unequal group sizes and input order expose row weighting / ordering errors.
    for event, size, rank in (("z", 10, 2), ("a", 20, 0), ("m", 30, 8)):
        for asset in range(size):
            rows.append(
                {
                    "pump_hash": event,
                    "currency_pair": f"ASSET{asset}",
                    "is_pumped": asset == 0,
                    "mixed": float(size - (rank if asset == 0 else 0 if asset == rank else asset)),
                    "good": float(size - asset),
                }
            )

    class Sample:
        def __init__(self):
            self.frame = pd.DataFrame(rows)
            self.requests = []

        def get_dataset(self, kind):
            assert kind is DatasetType.TEST, "Bootstrap must not read training or validation"
            self.requests.append(kind)
            return SimpleNamespace(all_data=lambda: self.frame)

    class Model:
        def __init__(self, column):
            self.column = column
            self.calls = 0
            self.params = {}

        def rank(self, dataset):
            self.calls += 1
            return dataset.all_data()[self.column].to_numpy()

    return [
        Experiment(name, Model(column), Sample())
        for name, column in (("mixed", "mixed"), ("strong", "good"), ("strong copy", "good"))
    ]


def test_shared_draws_match_explicitly_rebuilt_curves_and_existing_pairwise_tests():
    candidates = experiments()
    result = evaluate_test_auc_bootstrap(candidates, dataset_type=DatasetType.TEST, n_bootstrap=41, random_state=9)
    events = result.event_aucs.index.to_numpy()
    draws = events[result.sampled_indices]
    assert any(len(set(draw)) < len(events) for draw in draws)
    for candidate in candidates:
        assert candidate.model.calls == 1
        assert candidate.sample.requests == [DatasetType.TEST]
        frame = candidate.sample.frame
        for replicate, draw in enumerate(draws):
            curves = []
            for event in draw:
                group = frame.loc[frame.pump_hash == event].sort_values(
                    candidate.model.column, ascending=False, kind="stable"
                )
                curves.append([group.iloc[: int(np.ceil(len(group) * k))].is_pumped.any() for k in result.bins])
            expected = np.trapezoid(np.mean(curves, axis=0), x=result.bins) / 0.2
            assert result.bootstrap_aucs.loc[replicate, candidate.experiment_name] == pytest.approx(expected)
    _, expected_pairwise = compare_models_bootstrap_topk_percent_auc(
        result.scored_datasets, n_bootstrap=41, random_state=9
    )
    pd.testing.assert_frame_equal(result.pairwise[expected_pairwise.columns], expected_pairwise)
    identical = result.pairwise.query('model_a == "strong" and model_b == "strong copy"').iloc[0]
    assert identical.observed_diff == 0
    assert identical.p_value_holm == 1
    assert not identical.reject_h0


def test_rejects_different_assets_even_when_event_counts_match():
    candidates = experiments()
    candidates[1].sample.frame.loc[0, "currency_pair"] = "different"
    with pytest.raises(ValueError, match="same event assets and true labels"):
        evaluate_test_auc_bootstrap(candidates, dataset_type=DatasetType.TEST)


def test_notebook_uses_validation_winner_for_primary_model_without_test_access(monkeypatch):
    from backtest.utils.evaluation import select_best_experiment_on_validation

    notebook = json.loads((ROOT / "notebooks/research_notebook.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "838aa37064257d7e")
    validation = object()
    requests = []

    def get_dataset(kind):
        requests.append(kind)
        assert kind is DatasetType.VALIDATION, "Selection must never request TEST"
        return validation

    candidates = [
        Experiment(name, SimpleNamespace(score=score), SimpleNamespace(get_dataset=get_dataset))
        for name, score in [("L1", 0.60), ("L2", 0.61), ("CatBoost + ES", 0.65)]
    ]

    def score(model, dataset):
        assert dataset is validation
        return model.score

    monkeypatch.setattr("backtest.utils.evaluation.calculate_topk_percent_auc", score)
    namespace = {
        "experiments": candidates, "experiment_lr_tuned": candidates[0], "experiment_lr_l2": candidates[1],
        "DatasetType": DatasetType, "cols": [], "IMAGES": Path("unused"),
        "evaluate_experiments_topk_percent_curves": lambda *a, **kw: (pd.DataFrame(), pd.Series(dtype=float)),
        "plot_topk_percent_curves": lambda *a, **kw: None,
        "select_best_experiment_on_validation": select_best_experiment_on_validation,
        "highlight_max": lambda column: [""] * len(column),
    }
    exec(compile(source, "notebook:validation-selection", "exec"), namespace)
    assert namespace["selected_experiment"] is candidates[2]
    assert namespace["selected_experiment"] is namespace["validation_selected_experiment"]
    assert namespace["validation_auc_ranking"].index[0] == "CatBoost + ES"
    assert requests == [DatasetType.VALIDATION] * 3


@pytest.mark.parametrize("kind", [DatasetType.TRAIN, DatasetType.VALIDATION, "TEST"])
def test_rejects_non_test_partitions(kind):
    with pytest.raises(ValueError, match="Only the TEST"):
        evaluate_test_auc_bootstrap(experiments(), dataset_type=kind)


def test_notebook_bootstrap_cell_exports_replayable_draws_and_keeps_selected_model(tmp_path):
    notebook = json.loads((ROOT / "notebooks/research_notebook.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "test-auc-comparison")
    candidates = experiments()
    selected = candidates[1]
    namespace = {
        **vars(analysis),
        "experiments": candidates,
        "DatasetType": DatasetType,
        "PROJECT_ROOT": tmp_path,
        "selected_experiment": selected,
        "validation_selected_experiment": candidates[0],
        "input_provenance": input_provenance,
        "pump_manifest_sha256": "manifest",
        "l2_selection": {"test_used_for_tuning": False},
        "REPRODUCIBILITY": reproducibility_settings(),
        "snapshot_provenance": {"commit": "73ba716", "active_pumps_commit": "73ba716", "runtime_sha256": "runtime"},
        "split_counts": {"TEST": 3},
        "TRAIN_END": pd.Timestamp("2020-09-01"),
        "VALIDATION_END": pd.Timestamp("2021-05-01"),
        "pump_events": pd.DataFrame({"pump_hash": ["a", "m", "z"], "pump_time": pd.Timestamp("2022-01-01")}),
        "validation_auc_ranking": pd.Series([0.7, 0.6, 0.6], index=[c.experiment_name for c in candidates]),
        "plt": plt,
        "json": json,
        "display": lambda value: None,
    }
    try:
        exec(compile(source, "notebook:test-auc-comparison", "exec"), namespace)
    finally:
        plt.close("all")
    assert namespace["selected_experiment"] is selected
    output = namespace["bootstrap_output_dir"]
    result = namespace["test_auc_bootstrap"]
    figure = analysis.plot_test_auc_distributions(result)
    try:
        assert len(figure.axes) == 1
        assert len(figure.axes[0].get_yticklabels()) == len(candidates)
    finally:
        plt.close(figure)
    with np.load(output / "resampled_events.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sampled_indices"], result.sampled_indices)
        np.testing.assert_array_equal(data["event_ids"], result.event_aucs.index)
    draws = pd.read_csv(output / "auc_distributions.csv", index_col="replicate")
    np.testing.assert_allclose(draws, result.bootstrap_aucs)
    assert len(draws) == 10_000
    assert len(pd.read_parquet(output / "test_predictions.parquet")) == 180
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["provenance"]["selected_on_validation"] == candidates[0].experiment_name
    assert metadata["provenance"]["primary_model"] == selected.experiment_name
    assert metadata["provenance"]["reproducibility"]["random_seed"] == 42
    assert metadata["provenance"]["reproducibility"]["optuna_n_jobs"] == 1
    assert all(candidate.model.calls == 1 for candidate in candidates)
    for name in ("auc_distributions.png", "auc_distributions.svg", "auc_comparisons.png"):
        assert (output / name).stat().st_size > 1000


def test_snapshot_uses_exact_historical_paper_cohort(monkeypatch):
    from backtest.utils import build_dataset
    from datetime import datetime

    provenance = json.loads((ROOT / "research_snapshots/73ba716/provenance.json").read_text())
    manifest_path = ROOT / "research_snapshots/paper_0167ddd/pumps.json"
    manifest = manifest_path.read_bytes()
    cohort = provenance["paper_cohort"]
    assert cohort["paper_commit"] == "0167dddfa6b55819e276812836ac7537c5d3a00b"
    assert cohort["paper_pdf_git_blob"] == "862c708896830497e87e88f0069027c1e3c4424a"
    assert provenance["active_pumps_sha256"] == hashlib.sha256(manifest).hexdigest()
    source = (ROOT / cohort["source_manifest_path"]).read_bytes()
    dropped = (ROOT / cohort["historical_exclusions_path"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == "3f0cab25093a2c304d15b071b7b40344a7ce58c03ef79dd7b1e9b3a29b319467"
    assert hashlib.sha256(dropped).hexdigest() == "ff16500dce73763527f06cbee5fb32cec90572445e0fb1b206ef28440bab7500"

    def key(event):
        return (
            event.get("symbol", event.get("currency_pair")),
            event["exchange"],
            datetime.fromisoformat(event["time"]),
        )

    excluded = {key(event) for event in json.loads(dropped)}
    expected = [event for event in json.loads(source) if key(event) not in excluded]
    assert json.loads(manifest) == expected
    times = pd.to_datetime([event["time"] for event in expected])
    assert len(expected) == 473
    assert (times < "2020-09-01").sum() == 290
    assert ((times >= "2020-09-01") & (times < "2021-05-01")).sum() == 103
    assert (times >= "2021-05-01").sum() == 80

    class ManifestRead(Exception):
        pass

    def capture_manifest(path):
        assert path == manifest_path
        raise ManifestRead

    monkeypatch.setattr(build_dataset, "load_pumps", capture_manifest)
    with pytest.raises(ManifestRead):
        build_dataset.create_dataset()
