"""Interactive Top@k and Top@k% accuracy charts from computed metrics."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from colorsys import hls_to_rgb
from hashlib import sha256

import pandas as pd
import plotly.graph_objects as go

from pumps_and_dumps.backtest.datasets.sample import DatasetType

_MODEL_COLORS = {
    "Logistic regression": "#1f77b4",
    "Random forest": "#ff7f0e",
    "CatBoost classifier": "#2ca02c",
    "CatBoost classifier + SMOTE": "#d62728",
    "CatBoost classifier + Top@k%AUC early stopping": "#9467bd",
    "CatBoost ranker": "#8c564b",
    "CatBoost + Top-5": "#e377c2",
    "uniform": "#20252b",
}


def _strategy_color(strategy: str) -> str:
    if strategy in _MODEL_COLORS:
        return _MODEL_COLORS[strategy]
    hue = int.from_bytes(sha256(strategy.encode()).digest()[:4], "big") / 2**32
    red, green, blue = hls_to_rgb(hue, 0.43, 0.65)
    return f"rgb({round(red * 255)}, {round(green * 255)}, {round(blue * 255)})"


def _strategy_label(strategy: str) -> str:
    return (
        strategy.replace(" + Tuned", " (tuned)")
        .replace("CatBoost classifier + Top@k%AUC early stopping", "CatBoost · Top@k%AUC")
        .replace(" + ", " · ")
    )


def _with_origin(points: pd.DataFrame) -> pd.DataFrame:
    ordered = points.sort_index().copy()
    if ordered.empty:
        raise ValueError("At least one ranking threshold is required.")
    if 0 not in ordered.index:
        ordered.loc[0] = 0.0
    return ordered.sort_index()


def _accuracy_trace(
    values: pd.Series,
    *,
    percent: bool,
    auc: float | None = None,
    baseline: bool = False,
) -> go.Scatter:
    threshold = "Top %{x:.1%}" if percent else "Top %{x:.0f}"
    auc_detail = f"<br>Top@k%AUC: {auc:.4f}" if auc is not None else ""
    name = str(values.name)
    return go.Scatter(
        x=values.index,
        y=values,
        name=name if baseline else _strategy_label(name),
        mode="lines",
        line={
            "color": _strategy_color("uniform" if baseline else name),
            "width": 2 if baseline else 2.5,
            "dash": "dash" if baseline else "solid",
        },
        hovertemplate=f"<b>{name}</b><br>{threshold}<br>Accuracy: %{{y:.2%}}{auc_detail}<extra></extra>",
    )


def plot_ranking_accuracy(
    points: pd.DataFrame,
    *,
    dataset_type: DatasetType = DatasetType.TEST,
    percent: bool = False,
    random_baseline: Sequence[float] | None = None,
    auc_scores: Mapping[str, float] | None = None,
    save_path: str | Path | None = None,
) -> go.Figure:
    """Plot hit rates indexed by K, or by a fraction of each event's candidates.

    Each column identifies a model. Optional baseline values follow the input
    index order and must use the same selection rule as the supplied metrics.
    AUC scores are displayed in hover details. HTML exports work offline.
    """
    plotted = points.copy()
    baseline_name = "Uniform random"
    if random_baseline is not None:
        plotted[baseline_name] = list(random_baseline)
    plotted = _with_origin(plotted)
    auc_scores = auc_scores or {}
    fig = go.Figure()
    for name in points.columns:
        fig.add_trace(_accuracy_trace(plotted[name], percent=percent, auc=auc_scores.get(name)))
    if random_baseline is not None:
        fig.add_trace(_accuracy_trace(plotted[baseline_name], percent=percent, baseline=True))

    metric = "Top@k%" if percent else "Top@k"
    fig.update_layout(
        template="plotly_white",
        height=680,
        autosize=True,
        margin={"l": 85, "r": 325, "t": 110, "b": 75},
        title={
            "text": f"{metric} accuracy · {dataset_type.name.lower()}",
            "x": 0.04,
            "subtitle": {
                "text": "Higher is better. Hover for accuracy; click a model to hide it, double-click to isolate it.",
                "font": {"size": 13, "color": "#49515c"},
            },
        },
        legend={
            "title": {"text": "Model"},
            "x": 1.02,
            "y": 1,
            "font": {"size": 11},
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        hovermode="closest",
        xaxis={
            "title": "Top fraction of candidate assets" if percent else "Number of top-ranked assets (K)",
            "tickformat": ".0%" if percent else "d",
            "rangemode": "tozero",
        },
        yaxis={"title": "Pumps with the true asset selected", "tickformat": ".0%", "range": [0, 1]},
    )
    if save_path is not None:
        destination = Path(save_path)
        if destination.suffix.lower() not in {".html", ".htm"}:
            raise ValueError("Interactive ranking charts must be saved as .html or .htm files.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            destination,
            include_plotlyjs=True,
            full_html=True,
            config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        )
    return fig
