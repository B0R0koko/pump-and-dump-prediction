import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pumps_and_dumps.backtest.datasets.sample import DatasetType
from pumps_and_dumps.backtest.research.portfolio_frontier_report import annotate_pareto, plot_portfolio_frontiers


def test_pareto_keeps_exact_duplicates_and_requires_strict_improvement() -> None:
    points = pd.DataFrame(
        {
            "strategy": ["A", "B", "A", "A", "A", "A", "B"],
            "portfolio_size": [1, 1, 2, 3, 4, 5, 2],
            "max_drawdown": [0.1, 0.05, 0.1, 0.2, 0.3, 0.1, 0.2],
            "final_return": [0.5, 0.5, 0.4, 0.5, 0.7, 0.5, 0.6],
        },
        index=[7, 7, 8, 8, 1, 2, 3],
    )
    original = points.copy(deep=True)
    result = annotate_pareto(points)
    assert result["is_pareto"].tolist() == [True, True, False, False, True, True, True]
    assert result["is_global_pareto"].tolist() == [False, True, False, False, True, False, True]
    pd.testing.assert_frame_equal(points, original)
    assert result.index.equals(points.index)

    # Identical coordinates stay globally nondominated, including across models.
    duplicates = points.iloc[[0, 5]].copy()
    duplicates.iloc[1, duplicates.columns.get_loc("strategy")] = "B"
    assert annotate_pareto(duplicates)["is_global_pareto"].all()


def test_plot_keeps_all_points_and_connects_only_the_model_frontier(tmp_path) -> None:
    points = pd.DataFrame(
        {
            "strategy": ["Logistic regression"] * 3 + ["uniform"],
            "portfolio_size": [1, 5, 10, 5],
            "max_drawdown": [0.1, 0.2, 0.3, 0.4],
            "final_return": [0.5, -0.1, 1.5, 0.1],
        }
    )
    destination = tmp_path / "plots" / "frontier.html"
    fig = plot_portfolio_frontiers(points, sampled=True, dataset_type=DatasetType.TEST, save_path=destination)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Sampled portfolio medians · test · Equal capital allocation"
    assert "full test paths" in fig.layout.annotations[0].text
    all_points, frontier = fig.data[:2]
    np.testing.assert_allclose(all_points.x, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(all_points.y, [0.5, -0.1, 1.5])
    np.testing.assert_allclose(frontier.x, [0.1, 0.3])
    np.testing.assert_allclose(frontier.y, [0.5, 1.5])
    assert all_points.mode == "markers"
    assert frontier.mode == "lines"
    assert list(all_points.customdata[1]) == ["Logistic regression", 5, "No", "No"]
    assert list(frontier.customdata[1]) == ["Logistic regression", 10, "Yes", "Yes"]
    for trace in fig.data:
        assert "K=%{customdata[1]}" in trace.hovertemplate
        assert "%{y:.2%}" in trace.hovertemplate
        assert "%{x:.2%}" in trace.hovertemplate
    assert [trace.showlegend for trace in fig.data] == [False, True] * 2
    assert [trace.legendgroup for trace in fig.data] == ["Logistic regression"] * 2 + ["uniform"] * 2
    assert fig.layout.legend.groupclick == "togglegroup"
    assert fig.layout.legend.itemdoubleclick == "toggleothers"
    assert fig.layout.xaxis.title.text == "Median maximum drawdown"
    assert fig.layout.yaxis.title.text == "Median final return"
    assert fig.layout.xaxis.tickformat == fig.layout.yaxis.tickformat == ".0%"

    deterministic = plot_portfolio_frontiers(points.iloc[:3])
    assert deterministic.layout.title.text == "Deterministic Top-K portfolios · validation · Equal capital allocation"
    assert deterministic.data[0].marker.color == all_points.marker.color
    assert deterministic.layout.xaxis.title.text == "Maximum drawdown"
    assert deterministic.layout.yaxis.title.text == "Final return"
    html = destination.read_text()
    assert "plotly.js v" in html
    assert "Plotly.newPlot" in html
    assert '"scrollZoom": true' in html
    assert not re.search(r"<script[^>]+\bsrc\s*=", html, flags=re.IGNORECASE)
    probability = plot_portfolio_frontiers(points, sampled=True, allocation="probability")
    assert "Probability-based capital allocation" in probability.layout.title.text
    assert "intended capital weight in cash" in probability.layout.annotations[0].text
