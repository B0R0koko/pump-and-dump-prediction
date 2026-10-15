"""Execute the validation section with fitted stand-ins, without accessing market data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.columns import COL_IS_PUMPED, COL_PUMP_HASH
from pumps_and_dumps.backtest.research import reports
from pumps_and_dumps.backtest.research import graphs
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.datasets.sample import DatasetType


@pytest.mark.parametrize("identical_models", [False, True])
def test_validation_bootstrap_notebook_compares_all_candidates_without_test_access(
    tmp_path, capsys, monkeypatch, identical_models: bool
) -> None:
    """Execute the real validation notebook cells using fitted deterministic stand-ins.

    Eleven candidates evaluate the same 12 events and 30 assets. Either one
    model ranks every true asset first or every model ties. TEST access fails
    immediately, and rank-call counters require each model to be scored once.
    All 55 paired comparisons, Holm decisions, threshold intervals, CSVs and
    the comparison plot must be produced without changing the selected model.
    Reversing the chosen-model view must reverse signs but preserve the
    original multiple-comparison correction without rescoring any model.
    """
    notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / "research_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text())
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell_id in ("validation-bootstrap-estimates", "test-bootstrap-estimates"):
        source = "".join(cells[cell_id]["source"])
        assert "n_portfolios" not in source
        assert "portfolio_random_state" not in source
        assert "N_PORTFOLIOS" not in source
        assert "PORTFOLIO_SAMPLE_SEED" not in source
    requested_splits = []
    n_events, n_assets = 12, 30

    class Dataset:
        def __init__(self):
            self.frame = pd.DataFrame(
                {
                    COL_PUMP_HASH: np.repeat([f"event_{i}" for i in range(n_events)], n_assets),
                    COL_IS_PUMPED: np.tile(np.arange(n_assets) == 0, n_events),
                    "currency_pair": np.tile([f"asset_{i:02d}" for i in range(n_assets)], n_events),
                }
            )

        def all_data(self):
            return self.frame

    class Sample:
        def __init__(self):
            self.dataset = Dataset()

        def get_dataset(self, ds_type):
            requested_splits.append(ds_type)
            assert ds_type is DatasetType.VALIDATION
            return self.dataset

    class Model:
        def __init__(self, winning: bool):
            self.winning = winning
            self.rank_calls = 0

        def rank(self, dataset):
            self.rank_calls += 1
            scores = np.tile(-np.arange(n_assets, dtype=float), n_events)
            scores[::n_assets] = -0.5 if self.winning else -n_assets
            return scores

    experiments = [Experiment(f"candidate {i}", Model(identical_models or i == 10), Sample()) for i in range(11)]
    notebook_dir = tmp_path / "notebooks"
    output_dir = notebook_dir / "analysis_outputs" / "split_60_10_20_10_v1" / "deterministic_selection"
    bootstrap_output = output_dir / "significance"
    images = bootstrap_output / "images"
    images.mkdir(parents=True)
    namespace = {
        "NOTEBOOK_DIR": notebook_dir,
        "OUTPUT_DIR": output_dir,
        "reports": reports,
        "graphs": graphs,
        "experiments": experiments,
        "selected_experiment": experiments[-1],
        "DatasetType": DatasetType,
        "pd": pd,
        "display": lambda value: None,
    }

    # Exercise the current notebook's estimates, exports, intervals and plot.
    for cell_id in (
        "validation-bootstrap-estimates",
        "validation-bootstrap-curves",
        "validation-bootstrap-figure",
    ):
        exec(compile("".join(cells[cell_id]["source"]), cell_id, "exec"), namespace)
    plt.close("all")

    assert requested_splits == [DatasetType.VALIDATION] * 11
    assert all(experiment.get_model().rank_calls == 1 for experiment in experiments)
    bootstrap = namespace["validation_bootstrap"]
    summary = bootstrap.auc_summary
    pairwise = bootstrap.pairwise
    # The optional chosen-model view is now a helper, not a notebook cell.
    chosen = reports.report_chosen_comparisons(
        pairwise,
        experiments[-1].get_experiment_name(),
        notebook_dir,
        output_dir=bootstrap_output / "validation",
    )
    assert len(summary) == 11
    assert len(pairwise) == 55
    assert len(chosen) == 10
    assert summary["n_cross_sections"].eq(n_events).all()
    assert summary["metric"].eq("topk_percent_auc").all()
    assert pairwise["metric"].eq("topk_percent_auc").all()
    assert chosen["metric"].eq("topk_percent_auc").all()
    for table in (summary, pairwise, chosen, bootstrap.topk_ci, bootstrap.topk_percent_ci):
        assert "n_portfolios" not in table
        assert "portfolio_random_state" not in table
    assert len(bootstrap.topk_ci) == 11 * 7
    assert len(bootstrap.topk_percent_ci) == 11 * 5
    assert namespace["selected_experiment"] is experiments[-1]

    output = capsys.readouterr().out
    if identical_models:
        assert chosen["conclusion"].eq("Inconclusive").all()
        assert chosen["p_value_holm"].eq(1).all()
        assert "No statistically significant superiority established" in output
    else:
        # The chosen model is model_b in every pair: intervals and signs must reverse together.
        assert chosen["conclusion"].eq("Chosen model better").all()
        assert (chosen["observed_diff"] > 0).all()
        assert (chosen["ci_lower"] > 0).all()
        for row in pairwise.loc[pairwise["model_b"] == "candidate 10"].itertuples():
            result = chosen.loc[row.model_a]
            assert result["observed_diff"] == pytest.approx(-row.observed_diff)
            assert result["ci_lower"] == pytest.approx(-row.ci_upper)
            assert result["ci_upper"] == pytest.approx(-row.ci_lower)
            assert result["p_value_holm"] == row.p_value_holm
        assert "Validation evidence of higher Top@k%AUC" in output

    assert "shared event resamples" in output
    assert "weighted draws" not in output
    export_dir = bootstrap_output / "validation"
    assert len(list(export_dir.glob("validation_*.csv"))) == 5
    exported_pairs = pd.read_csv(export_dir / "validation_topk_percent_auc_pairwise.csv")
    assert len(exported_pairs) == 55
    assert exported_pairs["dataset_type"].eq("VALIDATION").all()
    assert not (export_dir / "validation_event_hit_probabilities.csv").exists()
    assert (images / "validation_bootstrap_model_comparison.png").is_file()
    assert not (notebook_dir / "analysis_outputs" / "significance").exists()

    # Changing the chosen model reuses the full comparison family and does not rescore.
    alternate = reports.report_chosen_comparisons(pairwise, "candidate 5", notebook_dir, output_dir=export_dir)
    assert len(alternate) == 10
    assert all(experiment.get_model().rank_calls == 1 for experiment in experiments)
    if not identical_models:
        assert alternate.loc["candidate 10", "conclusion"] == "Comparator better"
        assert (alternate.drop(index="candidate 10")["conclusion"] == "Inconclusive").all()
