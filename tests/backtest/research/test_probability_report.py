from math import comb

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.research.probability_report import (
    _exceedance_curve,
    plot_inclusion_probability_distributions,
)


def _probabilities() -> pd.DataFrame:
    rows = []
    for model, true_counts in {
        "Logistic regression": [0, 2, 2, 8],
        "CatBoost + QuerySoftMax": [0, 4, 6, 8],
    }.items():
        for k in (1, 3, 5):
            rows.extend(
                {
                    "model": model,
                    "pump_hash": f"pump-{event}",
                    "k": k,
                    # Uniform draws from eight assets, with different true-label counts.
                    "hit_probability": 1 - comb(8 - n_true, k) / comb(8, k),
                }
                for event, n_true in enumerate(true_counts)
            )
    return pd.DataFrame(rows)


def test_exceedance_curves_are_exact_for_ties_and_zero_one_endpoints() -> None:
    x, y = _exceedance_curve(np.array([0.0, 0.25, 0.25, 1.0]))
    np.testing.assert_array_equal(x, [0.0, 0.25, 1.0])
    np.testing.assert_array_equal(y, [1.0, 0.75, 0.25])
    for values in ([0.0] * 4, [1.0] * 4, [0.2, 0.4, 0.4, 0.9]):
        x, y = _exceedance_curve(np.array(values))
        for threshold in np.linspace(0, 1, 101):
            # For an inclusive empirical survival curve, the next knot's
            # height gives the interval value; no linear interpolation.
            height = y[np.searchsorted(x, threshold, side="left")]
            assert height == pytest.approx(np.mean(np.array(values) >= threshold))


def test_distributions_show_exact_event_fractions_without_histogram_bins(tmp_path) -> None:
    probabilities = _probabilities()
    original = probabilities.copy(deep=True)
    output = tmp_path / "plots" / "probability_distributions.png"
    fig = plot_inclusion_probability_distributions(probabilities, save_path=output)
    try:
        assert output.stat().st_size > 0
        assert len(fig.axes) == 3
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "Logistic regression",
            "CatBoost · QuerySoftMax",
        ]
        for ax, k in zip(fig.axes, (1, 3, 5)):
            assert len(ax.lines) == 2
            assert not ax.patches
            assert ax.get_xlim() == (0.0, 1.0)
            assert ax.get_title(loc="left") == f"K = {k} {'draw' if k == 1 else 'draws'}"
            for line, model in zip(ax.lines, ("Logistic regression", "CatBoost + QuerySoftMax")):
                values = probabilities.loc[
                    (probabilities["model"] == model) & (probabilities["k"] == k), "hit_probability"
                ].to_numpy()
                x, y = line.get_data()
                assert x[0] == 0 and x[-1] == 1
                np.testing.assert_allclose(y, [(values >= threshold).mean() for threshold in x])
                assert line.get_drawstyle() == "steps-pre"
                assert line.get_color() == fig.axes[0].lines[ax.lines.index(line)].get_color()
        assert "threshold" in fig._supxlabel.get_text()
        assert "at least" in fig._supylabel.get_text()
        pd.testing.assert_frame_equal(probabilities, original)
    finally:
        plt.close(fig)


def test_seven_panels_eight_models_and_extreme_tied_distributions() -> None:
    rows = [
        {"model": f"Model {model}", "pump_hash": "one-event", "k": k, "hit_probability": float(model % 2)}
        for model in range(8)
        for k in (1, 2, 3, 5, 10, 20, 30)
    ]
    fig = plot_inclusion_probability_distributions(pd.DataFrame(rows))
    try:
        assert len(fig.axes) == 7
        assert len(fig.legends[0].get_texts()) == 8
        for ax in fig.axes:
            assert len(ax.lines) == 8
            for index, line in enumerate(ax.lines):
                x, y = line.get_data()
                np.testing.assert_array_equal(x, [0, 1])
                np.testing.assert_array_equal(y, [1, index % 2])
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_high_contrast_export_remains_legible_under_dark_notebook_style() -> None:
    frame = _probabilities()
    with plt.style.context("dark_background"):
        fig = plot_inclusion_probability_distributions(frame[frame["model"] == "CatBoost + QuerySoftMax"])
    try:
        np.testing.assert_allclose(fig.get_facecolor(), to_rgba("white"))
        np.testing.assert_allclose(fig.axes[0].get_facecolor(), to_rgba("white"))
        assert fig.axes[0].lines[0].get_color() == "#8c564b"
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_probabilities_are_rejected(bad_value) -> None:
    frame = _probabilities()
    frame.loc[0, "hit_probability"] = bad_value
    with pytest.raises(ValueError):
        plot_inclusion_probability_distributions(frame)


def test_unpaired_events_and_duplicate_observations_are_rejected() -> None:
    frame = _probabilities()
    with pytest.raises(ValueError, match="same pump events"):
        plot_inclusion_probability_distributions(frame.iloc[1:])
    with pytest.raises(ValueError, match="exactly one probability"):
        plot_inclusion_probability_distributions(pd.concat([frame, frame.iloc[[0]]]))


@pytest.mark.parametrize("replacement", [True, None, np.nan, pd.NA])
def test_incompatible_or_missing_replacement_metadata_is_rejected(replacement) -> None:
    frame = _probabilities()
    frame["draw_replacement"] = pd.Series(False, index=frame.index, dtype=object)
    frame.loc[0, "draw_replacement"] = replacement
    with pytest.raises(ValueError, match="draw_replacement=False"):
        plot_inclusion_probability_distributions(frame)


def test_without_replacement_metadata_is_accepted() -> None:
    frame = _probabilities().assign(draw_replacement=False)
    fig = plot_inclusion_probability_distributions(frame)
    try:
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)
