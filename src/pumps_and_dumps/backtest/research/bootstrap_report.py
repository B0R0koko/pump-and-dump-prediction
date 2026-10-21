"""Readable, offline reports for paired bootstrap samples of whole test pumps."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pumps_and_dumps.backtest.research.ranking_report import _strategy_color, _strategy_label

if TYPE_CHECKING:
    from pumps_and_dumps.backtest.robustness.metric_bootstrap import TestMetricBootstrapResults


METRIC_LABELS = {
    "topk_percent_auc": "Top@k%AUC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
}


def _save_html(figure: go.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    destination = Path(save_path)
    if destination.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("Interactive bootstrap charts must be saved as .html or .htm files.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        destination,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )


def _score_range(values: np.ndarray) -> list[float]:
    lower, upper = float(values.min()), float(values.max())
    padding = max(0.02, (upper - lower) * 0.06)
    return [max(0.0, lower - padding), min(1.0, upper + padding)]


def _classification_note(top_k: int | None) -> str:
    if top_k == 1:
        return "With one true and one selected asset per pump, F1, precision and recall coincide."
    if top_k is not None:
        return f"Classification selects the top {top_k} assets in each pump."
    return "Classification uses the fixed selection rule from the evaluation."


def plot_metric_distributions(
    bootstrap_metrics: pd.DataFrame,
    save_path: str | Path | None = None,
) -> go.Figure:
    """Overlay the empirical distributions of four paired test-pump metrics.

    Each input row is one model evaluated on one bootstrap replicate; every
    replicate must contain every model. Ties contribute their full empirical
    mass at the observed score, without smoothing discrete classification
    metrics. The y-axis is the fraction of replicates at or below the score;
    distributions farther to the right have higher scores. One legend entry
    controls a model in all four panels. Exported HTML needs no network access.
    """
    if bootstrap_metrics.empty:
        raise ValueError("At least one bootstrap replicate is required.")
    models = bootstrap_metrics["model"].drop_duplicates().tolist()
    if (
        bootstrap_metrics.duplicated(["replicate", "model"]).any()
        or not (bootstrap_metrics.groupby("replicate").size() == len(models)).all()
    ):
        raise ValueError("Every bootstrap replicate must contain each model exactly once.")
    values = bootstrap_metrics[list(METRIC_LABELS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Bootstrap metric values must be finite scores between zero and one.")

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(METRIC_LABELS.values()),
        vertical_spacing=0.22,
        horizontal_spacing=0.12,
    )
    for metric_index, (metric, label) in enumerate(METRIC_LABELS.items()):
        row, column = divmod(metric_index, 2)
        bounds = _score_range(bootstrap_metrics[metric].to_numpy())
        for model in models:
            scores = bootstrap_metrics.loc[bootstrap_metrics["model"] == model, metric].to_numpy()
            unique, counts = np.unique(scores, return_counts=True)
            # Include both sides of the first jump, even for constant metrics.
            x = np.r_[bounds[0], unique, bounds[1]]
            y = np.r_[0.0, np.cumsum(counts) / len(scores), 1.0]
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=_strategy_label(model),
                    legendgroup=model,
                    showlegend=metric_index == 0,
                    mode="lines",
                    line={"color": _strategy_color(model), "width": 2.5, "shape": "hv"},
                    hovertemplate=(
                        f"<b>{model}</b><br>{label}: %{{x:.3f}}"
                        "<br>Samples at or below score: %{y:.1%}<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=column + 1,
            )
        figure.update_xaxes(
            title_text="Score (higher is better)", range=bounds, tickformat=".2f", row=row + 1, col=column + 1
        )
        figure.update_yaxes(
            title_text="Samples at or below score" if column == 0 else None,
            range=[0, 1.015],
            tickformat=".0%",
            row=row + 1,
            col=column + 1,
        )

    n_replicates = bootstrap_metrics["replicate"].nunique()
    note = _classification_note(bootstrap_metrics.attrs.get("classification_top_k"))
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=920,
        margin={"l": 85, "r": 30, "t": 150, "b": 175},
        title={
            "text": f"Model metrics across {n_replicates:,} resampled test sets",
            "x": 0.04,
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 23},
            "subtitle": {
                "text": (
                    "The same whole-pump sample is used for every model. Curves farther right are better."
                    f"<br>{note}<br>Click a legend entry to hide a model in all panels; double-click to isolate it."
                ),
                "font": {"size": 12, "color": "#49515c"},
            },
        },
        legend={
            "orientation": "h",
            "x": 0,
            "y": -0.18,
            "yanchor": "top",
            "entrywidth": 0.32,
            "entrywidthmode": "fraction",
            "font": {"size": 11},
            "groupclick": "togglegroup",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        hovermode="closest",
    )
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    _save_html(figure, save_path)
    return figure


def plot_metric_intervals(
    metric_summary: pd.DataFrame,
    save_path: str | Path | None = None,
) -> go.Figure:
    """Compare original-test estimates and pointwise bootstrap intervals.

    A metric dropdown keeps model labels readable in a single full-width
    plot. The model order is fixed by original-test Top@k%AUC across metrics.
    Dots mark original-test estimates and horizontal segments mark percentile
    intervals; a point estimate need not lie inside its bootstrap interval.
    Interval overlap is not a paired hypothesis test. Use the separately
    exported, multiplicity-adjusted pairwise comparisons for such decisions.
    """
    if metric_summary.empty:
        raise ValueError("Metric summaries are required.")
    summary = metric_summary.set_index(["metric", "model"])
    if not summary.index.is_unique:
        raise ValueError("Each model and metric must have exactly one summary row.")
    models = (
        summary.loc["topk_percent_auc"].sort_values("point_estimate", ascending=False, kind="stable").index.tolist()
    )
    labels = [_strategy_label(model) for model in models]
    figure = go.Figure()
    buttons = []
    for metric_index, (metric, label) in enumerate(METRIC_LABELS.items()):
        selected = summary.loc[metric].loc[models]
        interval_x, interval_y = [], []
        for model, row in selected.iterrows():
            interval_x.extend([row["ci_lower"], row["ci_upper"], None])
            interval_y.extend([_strategy_label(model), _strategy_label(model), None])
        figure.add_trace(
            go.Scatter(
                x=interval_x,
                y=interval_y,
                mode="lines",
                line={"color": "#8995a5", "width": 4},
                hoverinfo="skip",
                visible=metric_index == 0,
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=selected["point_estimate"],
                y=labels,
                mode="markers",
                marker={"color": [_strategy_color(model) for model in models], "size": 11},
                customdata=selected[["ci_lower", "ci_upper", "mean_bootstrap", "std_bootstrap"]].to_numpy(),
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{label} on original test set: %{{x:.4f}}"
                    "<br>Percentile interval: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                    "<br>Bootstrap mean: %{customdata[2]:.4f}"
                    "<br>Bootstrap standard deviation: %{customdata[3]:.4f}<extra></extra>"
                ),
                visible=metric_index == 0,
                showlegend=False,
            )
        )
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": [index // 2 == metric_index for index in range(2 * len(METRIC_LABELS))]},
                    {
                        "xaxis.title.text": f"{label} (higher is better)",
                        "xaxis.range": _score_range(selected[["point_estimate", "ci_lower", "ci_upper"]].to_numpy()),
                    },
                ],
            }
        )

    alpha = float(metric_summary["alpha"].iloc[0])
    note = _classification_note(
        int(metric_summary["classification_top_k"].iloc[0]) if "classification_top_k" in metric_summary else None
    )
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=max(620, 45 * len(models) + 265),
        margin={"l": 260, "r": 35, "t": 175, "b": 80},
        title={
            "text": "Test-set estimates and bootstrap uncertainty",
            "x": 0.03,
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 23},
            "subtitle": {
                "text": (
                    f"Dots: original test set. Lines: {(1 - alpha):.0%} pointwise percentile intervals."
                    "<br>Use the paired comparison table to assess differences between models."
                    f"<br>{note}"
                ),
                "font": {"size": 12, "color": "#49515c"},
            },
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.13, "xanchor": "left", "yanchor": "top"}],
        xaxis={
            "title": {"text": "Top@k%AUC (higher is better)"},
            "range": buttons[0]["args"][1]["xaxis.range"],
            "tickformat": ".2f",
            "automargin": True,
        },
        yaxis={"categoryorder": "array", "categoryarray": labels, "autorange": "reversed", "automargin": True},
    )
    _save_html(figure, save_path)
    return figure


def export_metric_bootstrap(results: "TestMetricBootstrapResults", output_dir: str | Path) -> dict[str, Path]:
    """Write all inputs needed to reconstruct each paired metric comparison.

    Sampled event IDs retain their replicate index and repeated draws. Per-event
    sufficient statistics and fitted-model scores make each resampled metric
    auditable without retraining. Metadata records evaluation settings, model
    and metric names, and the pairwise inference method. CSV tables preserve
    floating-point precision; the metadata and chart exports are independent.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "bootstrap_metrics": results.bootstrap_metrics,
        "sampled_events": results.sampled_events.reset_index(),
        "event_metrics": results.event_metrics,
        "metric_summary": results.metric_summary,
        "pairwise": results.pairwise,
        "scored_predictions": pd.concat(
            [frame.assign(model=model) for model, frame in results.scored_datasets.items()], ignore_index=True
        ),
    }
    paths = {}
    for name, frame in tables.items():
        paths[name] = destination / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    metadata = {
        **results.bootstrap_metrics.attrs,
        "models": list(results.scored_datasets),
        "metrics": list(METRIC_LABELS),
        "pairwise_inference": dict(results.pairwise.attrs),
        "tables": {name: {"file": path.name, "rows": len(tables[name])} for name, path in paths.items()},
    }
    paths["metadata"] = destination / "metadata.json"
    paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
