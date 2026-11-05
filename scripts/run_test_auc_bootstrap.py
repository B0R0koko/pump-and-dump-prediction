#!/usr/bin/env python3
"""Run model fits and test AUC bootstrap; use --full for the full research notebook."""

import argparse
import os
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per model (default: 1 for repeatability)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Run the full notebook, including portfolios and robustness")
    mode.add_argument("--portfolio", action="store_true", help="Run the 1,000 USDT slippage notebook after --full")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    # These must be set before the new kernel imports numerical libraries.
    os.environ.update({name: str(args.threads) for name in (
        "PUMPS_AND_DUMPS_MODEL_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS",
    )})
    os.environ["PYTHONHASHSEED"] = "42"
    root = Path(__file__).resolve().parents[1]
    if args.portfolio:
        notebook = nbformat.read(root / "notebooks/portfolio_1000usdt_slippage.ipynb", as_version=4)
        output = root / "notebooks/analysis_outputs/paper_cohort_473_l2/portfolio_1000usdt_slippage"
        output.mkdir(parents=True, exist_ok=True)
        target = output / "portfolio_1000usdt_slippage.executed.ipynb"

        def portfolio_progress(cell, cell_index, **kwargs):
            nbformat.write(notebook, target)
            print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:100]}", flush=True)

        try:
            NotebookClient(notebook, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(root)}},
                           on_cell_start=portfolio_progress).execute()
        finally:
            nbformat.write(notebook, target)
        print(f"Completed: {output}")
        return
    source = nbformat.read(root / "notebooks/research_notebook.ipynb", as_version=4)
    output = root / "notebooks/analysis_outputs/paper_cohort_473_l2/test_auc_bootstrap"
    output.mkdir(parents=True, exist_ok=True)
    cells = [c for c in source.cells if "test-auc-setup" in c.metadata.get("tags", [])]
    if args.full:
        cells = list(source.cells)
        output = root / "notebooks/analysis_outputs/paper_cohort_473_l2/paper_run"
        output.mkdir(parents=True, exist_ok=True)
        snapshot = output / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        for relative in (
            "notebooks/research_notebook.ipynb", "notebooks/analysis_helpers.ipynb",
            "notebooks/portfolio_1000usdt_slippage.ipynb", "research_snapshots/73ba716/runtime.zip",
            "research_snapshots/reproducibility.py", "research_snapshots/artifacts.py",
            "scripts/run_test_auc_bootstrap.py",
        ):
            shutil.copy2(root / relative, snapshot / Path(relative).name)
    # Suppress fit logging; seed and thread settings come from the shared policy.
    cells.insert(
        1,
        nbformat.v4.new_code_cell(f"""
import importlib
for pipeline_name in ("CatboostClassifier", "CatboostClassifierSMOTE", "CatboostRanker", "CatboostClassifierTOPKAUC"):
    module = importlib.import_module(f"backtest.pipelines.{{pipeline_name}}.pipe")
    module._BASE_PARAMS = module._BASE_PARAMS | {{"verbose": False, "allow_writing_files": False}}
"""),
    )
    if not args.full:
        cells.append(nbformat.v4.new_code_cell("""
validation_selected_experiment, validation_auc_ranking = select_best_experiment_on_validation(experiments)
selected_experiment = validation_selected_experiment
print(f"Validation winner: {validation_selected_experiment.get_experiment_name()}")
print(f"Primary model (validation selection): {selected_experiment.get_experiment_name()}")
display(validation_auc_ranking.to_frame("Top@k%AUC"))
"""))
        cells.extend(c for c in source.cells if c.id in ("test-auc-explanation", "test-auc-comparison"))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=source.metadata)

    def progress(cell, cell_index, **kwargs):
        name = "research_notebook.executed.ipynb" if args.full else "test_auc_bootstrap.executed.ipynb"
        nbformat.write(notebook, output / name)
        print(f"Executing cell {cell_index + 1}/{len(cells)}: {cell.source.splitlines()[0][:100]}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        on_cell_start=progress,
    )
    try:
        client.execute()
    finally:
        name = "research_notebook.executed.ipynb" if args.full else "test_auc_bootstrap.executed.ipynb"
        nbformat.write(notebook, output / name)
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
