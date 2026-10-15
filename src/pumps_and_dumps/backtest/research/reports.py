"""Assemble notebook tables and exports without changing evaluation or model selection."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from IPython.display import display

from pumps_and_dumps.backtest import evaluation
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.datasets.sample import DatasetType
from pumps_and_dumps.backtest.robustness.validation_bootstrap import (
    ValidationBootstrapResults,
    comparisons_for_model,
)


def validation_metrics_table(
    auc_ranking: pd.Series,
    topk: pd.DataFrame,
    topk_percent: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    auc_label: str = "Top@k%AUC",
) -> pd.DataFrame:
    """Align the already computed validation metrics by model name."""
    table = auc_ranking.rename(auc_label).to_frame()
    for k in (1, 5, 10):
        table[f"Top@{k}"] = topk.T[k]
    for fraction, label in ((0.01, "Top@1%"), (0.05, "Top@5%"), (0.10, "Top@10%")):
        table[label] = topk_percent.T[fraction]
    for metric in (
        "PR-AUC",
        "F1 (Top@1 per cross-section)",
        "Balanced Accuracy (Top@1 per cross-section)",
    ):
        table[metric] = classification[metric]
    return table.rename(
        columns={
            "F1 (Top@1 per cross-section)": "F1",
            "Balanced Accuracy (Top@1 per cross-section)": "Balanced accuracy",
        }
    )


def report_bootstrap(
    results: ValidationBootstrapResults,
    notebook_dir: Path,
    *,
    n_bootstrap: int,
    dataset_type: DatasetType = DatasetType.VALIDATION,
    output_dir: Path | None = None,
) -> None:
    """Export the full comparison family and display the AUC summary without rescoring."""
    split = dataset_type.name.lower()
    if not results.auc_summary["dataset_type"].eq(dataset_type.name).all():
        raise ValueError("Bootstrap results must match the requested report split")
    metric_label = "Top@k%AUC"
    folder = "significance" if dataset_type is DatasetType.VALIDATION else f"significance_{split}"
    output = output_dir if output_dir is not None else notebook_dir / "analysis_outputs" / folder
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in {
        f"{split}_topk_percent_auc_ci.csv": results.auc_summary,
        f"{split}_topk_ci.csv": results.topk_ci,
        f"{split}_topk_percent_ci.csv": results.topk_percent_ci,
    }.items():
        table.to_csv(output / filename)
    results.pairwise.to_csv(output / f"{split}_topk_percent_auc_pairwise.csv", index=False)
    print(
        f"{len(results.scored_datasets)} models; {len(results.pairwise)} paired comparisons; "
        f"{n_bootstrap:,} shared event resamples."
    )
    columns = [
        "point_estimate",
        "mean_bootstrap",
        "std_bootstrap",
        "ci_lower",
        "ci_upper",
        "n_cross_sections",
    ]
    display(
        results.auc_summary[columns]
        .style.format(precision=4)
        .set_caption(f"{split.capitalize()} {metric_label}: pointwise bootstrap intervals")
    )


def report_validation_bootstrap(
    results: ValidationBootstrapResults,
    notebook_dir: Path,
    *,
    n_bootstrap: int,
    output_dir: Path | None = None,
) -> None:
    """Preserve the validation report API and its existing output directory."""
    report_bootstrap(results, notebook_dir, n_bootstrap=n_bootstrap, output_dir=output_dir)


def report_chosen_comparisons(
    pairwise: pd.DataFrame,
    model_name: str,
    notebook_dir: Path,
    *,
    alpha: float = 0.05,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Report one view of the existing Holm-adjusted comparisons, without retesting."""
    metric_label = "Top@k%AUC"
    chosen = comparisons_for_model(pairwise, model_name)
    for column in ("metric", "dataset_type"):
        if column in pairwise:
            chosen[column] = pairwise[column].iloc[0]
    output = output_dir if output_dir is not None else notebook_dir / "analysis_outputs" / "significance"
    output.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(output / "validation_chosen_model_comparisons.csv")
    display(
        chosen.style.format(precision=4).set_caption(
            f"{metric_label}: chosen model minus comparator; " "decisions use Holm-adjusted p-values"
        )
    )
    wins = chosen.index[chosen["conclusion"] == "Chosen model better"].tolist()
    print(f"Chosen model: {model_name}")
    if wins:
        print(f"Validation evidence of higher {metric_label} (Holm-adjusted p < {alpha:g}) than: " + "; ".join(wins))
    else:
        print("No statistically significant superiority established for the chosen model on validation.")
    print(
        "These comparisons condition on fitted models. Tuning used a separate partition in this run; "
        "earlier research inspected these periods, so this is a retrospective comparison."
    )
    return chosen


def portfolio_benchmark(
    experiment: Experiment, *, portfolio_sizes: Sequence[int] = (1, 2, 5, 10, 20, 30)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed-size TEST portfolios with the time-aligned BTC benchmark."""
    curves = {
        k: evaluation.get_equity_curve_for_experiment(experiment=experiment, portfolio_size=k) for k in portfolio_sizes
    }
    returns = pd.concat(curves, axis=1)
    returns.columns = list(portfolio_sizes)
    statistics = evaluation.compute_portfolio_statistics(returns)
    dataset = experiment.get_sample().get_dataset(DatasetType.TEST)
    btc = evaluation.get_btc_buy_and_hold_baseline(dataset)
    btc_statistics = evaluation.compute_btc_buy_and_hold_statistics(btc)
    aligned = evaluation.align_portfolio_returns_with_btc(returns.rename(columns=lambda k: f"K={k}"), btc)
    statistics.insert(1, "cumulative return", returns.sum())
    statistics.loc[btc_statistics.name] = {
        "average event return": float("nan"),
        **btc_statistics.to_dict(),
    }
    print(f"Portfolio model selected on validation: {experiment.get_experiment_name()}")
    print(f"BTC benchmark interval: {btc.attrs['start_time']} to {btc.attrs['end_time']}")
    return aligned, statistics
