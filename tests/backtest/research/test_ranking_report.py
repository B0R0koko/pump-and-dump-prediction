import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.research.ranking_report import plot_ranking_accuracy


def test_ranking_chart_preserves_supplied_metrics_and_exact_baseline(tmp_path) -> None:
    """Preserve supplied deterministic ranking accuracies when sorting thresholds.

    Input thresholds arrive out of order. The plotted model and baseline must
    retain their paired values, begin at zero, export a self-contained HTML
    chart, and use the same model color for absolute and fractional K.
    """
    points = pd.DataFrame({"CatBoost classifier": [0.8, 0.4]}, index=[0.1, 0.01])
    original = points.copy(deep=True)
    destination = tmp_path / "ranking.html"
    figure = plot_ranking_accuracy(
        points,
        percent=True,
        random_baseline=[0.105, 0.018],
        auc_scores={"CatBoost classifier": 0.87},
        save_path=destination,
    )
    model, baseline = figure.data
    np.testing.assert_allclose(model.x, [0, 0.01, 0.1])
    np.testing.assert_allclose(model.y, [0, 0.4, 0.8])
    np.testing.assert_allclose(baseline.y, [0, 0.018, 0.105])
    assert model.mode == "lines"
    assert baseline.line.dash == "dash"
    assert "Top@k%AUC: 0.8700" in model.hovertemplate
    assert figure.layout.title.text == "Top@k% accuracy · test"
    assert figure.layout.xaxis.tickformat == ".0%"
    assert figure.layout.legend.itemdoubleclick == "toggleothers"
    assert "plotly.js v" in destination.read_text()
    pd.testing.assert_frame_equal(points, original)

    absolute = plot_ranking_accuracy(pd.DataFrame({"CatBoost classifier": [0.6]}, index=[5]))
    assert absolute.layout.title.text == "Top@k accuracy · test"
    assert absolute.layout.xaxis.tickformat == "d"
    assert absolute.data[0].line.color == model.line.color
