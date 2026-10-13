import re
from math import comb

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.research.probability_comparison import (
    plot_inclusion_probability_comparison,
    plot_inclusion_probability_summary,
)

PLOTTERS = (plot_inclusion_probability_comparison, plot_inclusion_probability_summary)


def _probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "pump_hash": f"pump-{event}",
                "k": k,
                # Uniform draws from eight assets, with different true-label counts.
                "hit_probability": 1 - comb(8 - n_true, k) / comb(8, k),
            }
            for model, true_counts in {
                "Logistic regression": [0, 2, 2, 8],
                "CatBoost + QuerySoftMax": [0, 4, 6, 8],
            }.items()
            for k in (1, 3, 5)
            for event, n_true in enumerate(true_counts)
        ]
    )


def test_separate_figures_preserve_exact_exceedance_mean_median_ties_zeros_and_ones() -> None:
    probabilities = _probabilities()
    original = probabilities.copy(deep=True)
    distribution = plot_inclusion_probability_comparison(probabilities, default_k=1)
    summary = plot_inclusion_probability_summary(probabilities, default_k=1)

    assert len(distribution.data) == 2
    assert len(summary.data) == 4
    for figure in (distribution, summary):
        assert figure.layout.width is None
        assert figure.layout.autosize is True
        assert "xaxis2" not in figure.layout and "yaxis2" not in figure.layout
    for index, model in enumerate(("Logistic regression", "CatBoost + QuerySoftMax")):
        curve = distribution.data[index]
        mean, median = summary.data[2 * index : 2 * index + 2]
        values = probabilities.loc[(probabilities["model"] == model) & (probabilities["k"] == 1), "hit_probability"]
        expected_thresholds = sorted(set([0.0, *values, 1.0]))
        np.testing.assert_array_equal(curve.x, expected_thresholds)
        np.testing.assert_allclose(curve.y, [(values >= threshold).mean() for threshold in expected_thresholds])
        assert curve.line.shape == "vh"
        assert curve.y[0] == 1.0
        assert curve.y[-1] == 0.25
        assert mean.x == (values.mean(),)
        assert median.x == (values.median(),)
        assert mean.marker.symbol == "circle"
        assert median.marker.symbol == "diamond-open"
        assert curve.legendgroup == mean.legendgroup == median.legendgroup == model
        assert mean.showlegend is True and median.showlegend is False
        assert all(detail[0] == model for detail in curve.customdata)
        assert [detail[4] for detail in curve.customdata] == [
            int((values >= threshold).sum()) for threshold in expected_thresholds
        ]
        assert all(detail[5] == 4 for detail in curve.customdata)
        assert mean.customdata[0][0] == median.customdata[0][0] == model

    # Both repeated 25% probabilities count at their threshold, but cease to
    # count immediately above it (the vertical-then-horizontal step).
    assert tuple(distribution.data[0].x) == (0.0, 0.25, 1.0)
    assert tuple(distribution.data[0].y) == (1.0, 0.75, 0.25)
    pd.testing.assert_frame_equal(probabilities, original)


def test_independent_k_controls_preserve_hidden_models_zoom_and_summary_order() -> None:
    probabilities = _probabilities()
    distribution = plot_inclusion_probability_comparison(probabilities)
    summary = plot_inclusion_probability_summary(probabilities)
    summary_initial = summary.to_json()
    model_order = tuple(summary.layout.yaxis.categoryarray)
    summary_range = summary.layout.xaxis.range
    for figure, traces_per_model in ((distribution, 1), (summary, 2)):
        dropdown = figure.layout.updatemenus[0]
        assert dropdown.active == 2  # K=5 is the default.
        assert "K = 5" in figure.layout.title.text
        assert "4 comparison pumps" in figure.layout.title.subtitle.text
        assert len(dropdown.buttons) == 3
        assert figure.layout.legend.groupclick == "togglegroup"
        assert figure.layout.legend.itemdoubleclick == "toggleothers"
        assert figure.layout.legend.uirevision
        for trace in figure.data[:traces_per_model]:
            trace.visible = "legendonly"
        if figure is distribution:
            figure.layout.xaxis.range = [0, 0.1]
        trace_uids = [trace.uid for trace in figure.data]
        for button, k in zip(dropdown.buttons, (1, 3, 5)):
            assert button.method == "update"
            update, layout = button.args
            assert "visible" not in update
            assert set(update) == {"x", "y", "customdata"}
            assert len(update["x"]) == len(figure.data)
            figure.plotly_update(update, layout)
            assert len(figure.data) == len(model_order) * traces_per_model
            assert [trace.uid for trace in figure.data] == trace_uids
            assert all(trace.visible == "legendonly" for trace in figure.data[:traces_per_model])
            assert tuple(summary.layout.yaxis.categoryarray) == model_order
            assert summary.layout.xaxis.range == summary_range
            assert f"K = {k}" in figure.layout.title.text
            for index, model in enumerate(model_order):
                values = probabilities.loc[
                    (probabilities["model"] == model) & (probabilities["k"] == k), "hit_probability"
                ]
                if figure is distribution:
                    assert summary.to_json() == summary_initial
                    assert figure.layout.xaxis.range == (0, 0.1)
                    curve = figure.data[index]
                    np.testing.assert_allclose(curve.y, [(values >= threshold).mean() for threshold in curve.x])
                    assert all(detail[1] == k for detail in curve.customdata)
                    assert [detail[4] for detail in curve.customdata] == [
                        int((values >= threshold).sum()) for threshold in curve.x
                    ]
                else:
                    mean, median = figure.data[index * 2 : index * 2 + 2]
                    assert mean.x == (values.mean(),)
                    assert median.x == (values.median(),)
                    assert mean.customdata[0][1] == median.customdata[0][1] == k


def test_default_falls_back_to_smallest_k_and_only_distribution_has_zoom() -> None:
    distribution, summary = (plotter(_probabilities(), default_k=100) for plotter in PLOTTERS)
    for figure in (distribution, summary):
        assert figure.layout.updatemenus[0].active == 0
        assert "K = 1" in figure.layout.title.text
        assert figure.layout.xaxis.range == (0, 1)
        assert figure.layout.paper_bgcolor == figure.layout.plot_bgcolor == "white"
    assert len(summary.layout.updatemenus) == 1
    for button, upper in zip(distribution.layout.updatemenus[1].buttons, (1, 0.3, 0.1)):
        assert button.method == "relayout"
        assert button.args[0] == {"xaxis.range": [0, upper], "xaxis.autorange": False}


def test_extreme_single_event_distributions_are_exact() -> None:
    frame = pd.DataFrame(
        [
            {"model": model, "pump_hash": "event", "k": 1, "hit_probability": probability}
            for model, probability in (("Zero", 0.0), ("One", 1.0))
        ]
    )
    figure = plot_inclusion_probability_comparison(frame)
    curves = {trace.legendgroup: trace for trace in figure.data}
    assert tuple(curves["Zero"].x) == tuple(curves["One"].x) == (0.0, 1.0)
    assert tuple(curves["Zero"].y) == (1.0, 0.0)
    assert tuple(curves["One"].y) == (1.0, 1.0)


@pytest.mark.parametrize("probability, upper", [(0.0, 0.1), (0.02, 0.1), (0.42, 0.5), (1.0, 1.0)])
def test_summary_range_fits_all_k_and_retains_probability_endpoints(probability, upper) -> None:
    frame = pd.DataFrame(
        [
            {"model": "Model", "pump_hash": "event", "k": k, "hit_probability": value}
            for k, value in ((1, probability / 2), (5, probability))
        ]
    )
    figure = plot_inclusion_probability_summary(frame, default_k=1)
    assert figure.layout.xaxis.range == (0, upper)
    assert figure.layout.xaxis.dtick == pytest.approx(upper / 5)
    update, layout = figure.layout.updatemenus[0].buttons[1].args
    figure.plotly_update(update, layout)
    assert figure.layout.xaxis.range == (0, upper)
    assert figure.data[0].x[0] == probability


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_export_is_standalone_and_contains_all_k_values(tmp_path, plotter) -> None:
    output = tmp_path / "plots" / "comparison.html"
    plotter(_probabilities(), save_path=output)
    html = output.read_text()
    assert "Plotly.newPlot" in html
    assert "plotly.js v" in html
    assert not re.search(r"<script[^>]+src=", html)
    assert all(f"K = {k}" in html for k in (1, 3, 5))
    with pytest.raises(ValueError, match=".html"):
        plotter(_probabilities(), save_path=tmp_path / "wrong.png")


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_requires_valid_probabilities_and_paired_events(plotter) -> None:
    probabilities = _probabilities()
    with pytest.raises(ValueError, match="same pump events"):
        plotter(probabilities.iloc[1:])
    probabilities.loc[0, "hit_probability"] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plotter(probabilities)


@pytest.mark.parametrize("plotter, trace_count", [(PLOTTERS[0], 2), (PLOTTERS[1], 4)])
def test_legacy_with_replacement_data_cannot_be_mislabeled(plotter, trace_count) -> None:
    probabilities = _probabilities().assign(draw_replacement=True)
    with pytest.raises(ValueError, match="draw_replacement=False"):
        plotter(probabilities)
    probabilities["draw_replacement"] = False
    assert len(plotter(probabilities).data) == trace_count
