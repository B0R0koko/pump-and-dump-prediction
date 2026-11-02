#!/usr/bin/env python3
"""Run model fits and test AUC bootstrap; use --full for the full research notebook."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8, help="CatBoost worker threads (default: 8)")
    parser.add_argument("--full", action="store_true", help="Run the full notebook, including portfolios and robustness")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = Path(__file__).resolve().parents[1]
    source = nbformat.read(root / "notebooks/research_notebook.ipynb", as_version=4)
    output = root / "notebooks/analysis_outputs/paper_cohort_473_l2/test_auc_bootstrap"
    output.mkdir(parents=True, exist_ok=True)
    cells = [c for c in source.cells if "test-auc-setup" in c.metadata.get("tags", [])]
    if args.full:
        cells = list(source.cells)
        output = root / "notebooks/analysis_outputs/paper_cohort_473_l2/paper_run"
        output.mkdir(parents=True, exist_ok=True)
    # Only resource/logging options change; statistical settings stay as in the notebook.
    cells.insert(
        1,
        nbformat.v4.new_code_cell(f"""
import importlib
for pipeline_name in ("CatboostClassifier", "CatboostClassifierSMOTE", "CatboostRanker", "CatboostClassifierTOPKAUC"):
    module = importlib.import_module(f"backtest.pipelines.{{pipeline_name}}.pipe")
    module._BASE_PARAMS = module._BASE_PARAMS | {{"thread_count": {args.threads}, "verbose": False, "allow_writing_files": False}}
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
