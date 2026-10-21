"""Run the actual test-bootstrap notebook cells without training or market data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import COL_CURRENCY_PAIR, COL_IS_PUMPED, COL_PUMP_HASH
from pumps_and_dumps.backtest.datasets.sample import DatasetType
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.research import graphs, reports


def test_test_bootstrap_notebook_reuses_frozen_models_and_exports_paired_metric_distributions(
    tmp_path, monkeypatch, capsys
):
    """Execute the compact notebook bootstrap section with fitted stand-ins.

    Three models rank four test pumps with six candidate assets each. One is
    always correct, one always wrong, and one correct on alternate pumps.
    Each sample rejects any request outside TEST and model counters require
    one inference call despite 40 bootstrap replicates. The intentionally
    weaker comparison-selected model must stay selected after all test views.

    Execute the actual notebook source, changing only the replicate count
    from 10,000 to 40, to catch broken imports, cell dependencies or variable
    names that isolated helper tests miss. Check the paired model/replicate
    table, all 12 pair/metric hypotheses, saved duplicate pump draws, all six
    CSVs and metadata. Check the Matplotlib comparison and its identical paper
    export without duplicate inline rendering or interactive HTML payloads.
    """
    notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / "research_notebook.ipynb"
    cells = {cell["id"]: cell for cell in json.loads(notebook_path.read_text())["cells"]}
    n_events, n_assets, n_models, n_bootstrap = 4, 6, 3, 40
    requested_splits = []

    class Dataset:
        def __init__(self):
            self.frame = pd.DataFrame(
                {
                    COL_PUMP_HASH: np.repeat([f"test_pump_{i}" for i in range(n_events)], n_assets),
                    COL_CURRENCY_PAIR: np.tile([f"ASSET{i}-BTC" for i in range(n_assets)], n_events),
                    COL_IS_PUMPED: np.tile(np.arange(n_assets) == 0, n_events),
                }
            )

        def all_data(self):
            return self.frame

    class Sample:
        def __init__(self):
            self.dataset = Dataset()

        def get_dataset(self, dataset_type):
            requested_splits.append(dataset_type)
            assert dataset_type is DatasetType.TEST
            return self.dataset

    class Model:
        def __init__(self, correct_events):
            self.correct_events = set(correct_events)
            self.rank_calls = 0

        def rank(self, dataset):
            self.rank_calls += 1
            scores = np.tile(-np.arange(n_assets, dtype=float), n_events)
            for event in range(n_events):
                scores[event * n_assets] = 1 if event in self.correct_events else -n_assets
            return scores

    experiments = [
        Experiment("strong", Model(range(n_events)), Sample()),
        Experiment("mixed", Model((0, 2)), Sample()),
        Experiment("weak", Model(()), Sample()),
    ]
    original_frames = [experiment.sample.dataset.all_data().copy(deep=True) for experiment in experiments]
    displayed = []
    monkeypatch.setattr(reports, "display", displayed.append)
    paper_images = tmp_path / "paper" / "images"
    output_dir = tmp_path / "analysis_outputs" / "deterministic_selection"
    namespace = {
        "OUTPUT_DIR": output_dir,
        "PAPER_IMAGES": paper_images,
        "graphs": graphs,
        "reports": reports,
        "experiments": experiments,
        "selected_experiment": experiments[-1],
        "display": displayed.append,
    }
    for cell_id in (
        "test-bootstrap-estimates",
        "test-bootstrap-figure",
        "test-bootstrap-pairwise",
    ):
        source = "".join(cells[cell_id]["source"])
        if cell_id == "test-bootstrap-estimates":
            assert source.count("N_TEST_BOOTSTRAP = 10_000") == 1
            source = source.replace("N_TEST_BOOTSTRAP = 10_000", f"N_TEST_BOOTSTRAP = {n_bootstrap}")
        exec(compile(source, cell_id, "exec"), namespace)

    result = namespace["test_bootstrap"]
    assert requested_splits == [DatasetType.TEST] * n_models
    assert all(experiment.model.rank_calls == 1 for experiment in experiments)
    for experiment, original in zip(experiments, original_frames):
        pd.testing.assert_frame_equal(experiment.sample.dataset.all_data(), original)
    assert namespace["selected_experiment"] is experiments[-1]
    assert len(result.bootstrap_metrics) == n_bootstrap * n_models
    assert result.bootstrap_metrics.groupby("replicate").size().eq(n_models).all()
    assert len(result.metric_summary) == n_models * 4
    assert len(result.pairwise) == 12
    assert result.pairwise.groupby("metric").size().eq(3).all()
    assert result.pairwise.attrs["n_hypotheses"] == 12
    assert result.sampled_events.shape == (n_bootstrap, n_events)
    assert any(len(set(row)) < n_events for row in result.sampled_events.to_numpy())
    assert (result.bootstrap_metrics["f1"] == result.bootstrap_metrics["precision"]).all()
    assert (result.bootstrap_metrics["f1"] == result.bootstrap_metrics["recall"]).all()
    assert result.metric_summary["dataset_type"].eq("TEST").all()
    assert result.metric_summary["n_bootstrap"].eq(n_bootstrap).all()
    assert len(displayed) == 2 + int(result.pairwise["reject_h0"].any())

    comparison = namespace["test_bootstrap_comparison"]
    assert not plt.fignum_exists(comparison.number)  # Explicit display must not duplicate the inline figure.
    assert comparison._suptitle.get_text().startswith("Test model comparison: Top@k%AUC")
    forest_ax, matrix_ax = comparison.axes
    auc_summary = result.metric_summary.query("metric == 'topk_percent_auc'").set_index("model")
    ordered = auc_summary.sort_values("point_estimate", ascending=False, kind="stable")
    np.testing.assert_allclose(forest_ax.collections[1].get_offsets()[:, 0], ordered["point_estimate"])
    decisions = matrix_ax.images[0].get_array()
    for row in result.pairwise.query("metric == 'topk_percent_auc'").itertuples(index=False):
        i, j = ordered.index.get_loc(row.model_a), ordered.index.get_loc(row.model_b)
        expected = (1 if row.better_model == row.model_a else -1) if row.reject_h0 else 0
        assert decisions[i, j] == expected
        assert decisions[j, i] == -expected

    destination = output_dir / "test_pump_bootstrap"
    assert (destination / "test_bootstrap_model_comparison.png").stat().st_size > 0
    expected_tables = {
        "bootstrap_metrics",
        "sampled_events",
        "event_metrics",
        "metric_summary",
        "pairwise",
        "scored_predictions",
    }
    assert {path.stem for path in destination.glob("*.csv")} == expected_tables
    assert set(namespace["test_bootstrap_files"]) == expected_tables | {"metadata"}
    exported_draws = pd.read_csv(destination / "sampled_events.csv").set_index("replicate")
    pd.testing.assert_frame_equal(exported_draws, result.sampled_events)
    exported_metrics = pd.read_csv(destination / "bootstrap_metrics.csv")
    pd.testing.assert_frame_equal(exported_metrics, result.bootstrap_metrics, check_exact=False)
    assert len(pd.read_csv(destination / "scored_predictions.csv")) == n_models * n_events * n_assets
    assert len(pd.read_csv(destination / "pairwise.csv")) == 12
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["dataset_type"] == "TEST"
    assert metadata["n_bootstrap"] == n_bootstrap
    assert metadata["n_cross_sections"] == n_events
    assert metadata["classification_top_k"] == 1
    assert metadata["pairwise_inference"]["n_hypotheses"] == 12
    assert metadata["pairwise_inference"]["p_value_adjustment"] == "Holm"
    assert metadata["models"] == ["strong", "mixed", "weak"]

    image_path = destination / "test_bootstrap_model_comparison.png"
    assert (paper_images / image_path.name).read_bytes() == image_path.read_bytes()
    assert displayed[1].data == image_path.read_bytes()
    assert not list(destination.glob("*.html"))
    output = capsys.readouterr().out
    assert "40 paired resamples × 4 test pumps per sample × 3 fixed models" in output
    assert "Holm correction uses all 12 model-pair × metric tests together" in output
    assert "Model frozen on comparison validation: weak" in output
