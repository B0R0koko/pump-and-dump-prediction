"""Hand-checkable examples for the paired pump-bootstrap reporting layer."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.research.bootstrap_report import (
    METRIC_LABELS,
    export_metric_bootstrap,
    plot_metric_distributions,
    plot_metric_intervals,
)


def _draws() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"replicate": replicate, "model": model, **dict.fromkeys(METRIC_LABELS, value)}
            for model, values in [("CatBoost classifier", [0.4, 0.2, 0.4, 0.8]), ("Random forest", [0.6] * 4)]
            for replicate, value in enumerate(values)
        ]
    )
    frame.attrs.update(dataset_type="TEST", n_bootstrap=4, classification_top_k=1, random_state=42)
    return frame


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "metric": metric,
                "point_estimate": point,
                "mean_bootstrap": mean,
                "std_bootstrap": 0.1,
                "ci_lower": lower,
                "ci_upper": upper,
                "alpha": 0.1,
                "classification_top_k": 1,
            }
            for metric in METRIC_LABELS
            for model, point, mean, lower, upper in [
                ("Random forest", 0.6, 0.6, 0.6, 0.6),
                ("CatBoost classifier", 0.95, 0.45, 0.2, 0.8),
            ]
        ]
    )


def test_ecdf_preserves_tie_mass_and_links_model_visibility_across_panels(tmp_path) -> None:
    """Show exact empirical mass instead of smoothing tied bootstrap scores.

    Four CatBoost draws are 0.2, 0.4, 0.4 and 0.8, so the cumulative share at
    0.4 must be 75%. The second model always scores 0.6 and therefore jumps
    directly from zero to one. All four panels must retain those facts, share
    model colors and legend controls, and export a self-contained offline page
    without changing the supplied table or its bootstrap metadata.
    """
    draws = _draws()
    original = draws.copy(deep=True)
    output = tmp_path / "charts" / "distributions.html"
    figure = plot_metric_distributions(draws, save_path=output)

    for panel in range(4):
        catboost, forest = figure.data[2 * panel : 2 * panel + 2]
        np.testing.assert_allclose(catboost.x[1:-1], [0.2, 0.4, 0.8])
        np.testing.assert_allclose(catboost.y, [0, 0.25, 0.75, 1, 1])
        np.testing.assert_allclose(forest.x[1:-1], [0.6])
        np.testing.assert_allclose(forest.y, [0, 1, 1])
        assert catboost.line.shape == forest.line.shape == "hv"
        assert catboost.line.color == figure.data[0].line.color
        assert forest.line.color == figure.data[1].line.color
        assert catboost.legendgroup == "CatBoost classifier"
        assert forest.legendgroup == "Random forest"
        assert catboost.showlegend == forest.showlegend == (panel == 0)
    assert figure.layout.legend.groupclick == "togglegroup"
    assert "4 resampled test sets" in figure.layout.title.text
    assert "F1, precision and recall coincide" in figure.layout.title.subtitle.text
    assert "plotly.js v" in output.read_text()
    assert '<script src="' not in output.read_text()
    pd.testing.assert_frame_equal(draws, original)
    assert draws.attrs == original.attrs


@pytest.mark.parametrize("problem", ["missing_model", "duplicate_model", "nonfinite"])
def test_distribution_rejects_incomplete_or_invalid_paired_results(problem) -> None:
    """Prevent comparisons that silently discard a model or corrupt draw.

    Removing one model from one replicate, duplicating its row, or replacing
    a metric with NaN must raise instead of plotting differently sized model
    samples or dropping an invalid value. These errors otherwise give a
    plausible but misleading empirical distribution.
    """
    draws = _draws()
    if problem == "missing_model":
        draws = draws.iloc[1:]
    elif problem == "duplicate_model":
        draws = pd.concat([draws, draws.iloc[[0]]], ignore_index=True)
    else:
        draws.loc[0, "precision"] = np.nan
    with pytest.raises(ValueError):
        plot_metric_distributions(draws)


def test_interval_plot_keeps_original_estimate_separate_from_percentile_interval(tmp_path) -> None:
    """Keep an estimate outside its interval visible without stretching the CI.

    The illustrative CatBoost estimate is 0.95, while its supplied percentile
    interval is [0.2, 0.8]. The chart must show exactly that segment and a
    separate 0.95 dot, use the requested 90% confidence level, and keep the
    same model order when switching among all four metrics. Readers must be
    directed to the paired comparison table instead of interval overlap.
    """
    summary = _summary()
    original = summary.copy(deep=True)
    output = tmp_path / "intervals.html"
    figure = plot_metric_intervals(summary, save_path=output)
    assert list(figure.data[0].x) == [0.2, 0.8, None, 0.6, 0.6, None]
    np.testing.assert_allclose(figure.data[1].x, [0.95, 0.6])
    assert list(figure.data[1].y) == ["CatBoost classifier", "Random forest"]
    assert "90% pointwise percentile intervals" in figure.layout.title.subtitle.text
    assert "paired comparison table" in figure.layout.title.subtitle.text
    buttons = figure.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == list(METRIC_LABELS.values())
    for metric_index, button in enumerate(buttons):
        assert list(button.args[0]["visible"]) == [index // 2 == metric_index for index in range(8)]
        assert button.args[1]["xaxis.range"][1] >= 0.95
        np.testing.assert_allclose(figure.data[2 * metric_index + 1].x, [0.95, 0.6])
    assert "plotly.js v" in output.read_text()
    pd.testing.assert_frame_equal(summary, original)


def test_top_two_caption_does_not_claim_identical_classification_metrics() -> None:
    """Describe the chosen rule without claiming the Top-1 metric identity.

    Selecting two candidates per pump can produce precision different from
    recall even when there is exactly one true asset. Both views must explain
    Top-2 selection and omit the conditional equality used for Top-1 charts.
    """
    draws, summary = _draws(), _summary()
    draws.attrs["classification_top_k"] = 2
    summary["classification_top_k"] = 2
    for figure in (plot_metric_distributions(draws), plot_metric_intervals(summary)):
        assert "top 2 assets" in figure.layout.title.subtitle.text
        assert "coincide" not in figure.layout.title.subtitle.text


def test_export_retains_duplicate_pump_draws_scores_and_inference_metadata(tmp_path) -> None:
    """Preserve the information needed to reproduce a sampled-set comparison.

    Replicate zero deliberately samples pump A twice and never samples pump
    B; the CSV must preserve that multiplicity and its replicate identifier.
    The export must also retain fitted-model scores, raw metric draws,
    sufficient statistics, summaries and pairwise results. A read-back of
    metadata verifies the random seed, Top-K rule and correction family are
    available independently of the in-memory DataFrame attributes.
    """
    sampled_events = pd.DataFrame(
        {"draw_0": ["pump-A", "pump-B"], "draw_1": ["pump-A", "pump-A"]},
        index=pd.Index([0, 1], name="replicate"),
    )
    scored = pd.DataFrame({"pump_hash": ["pump-A", "pump-B"], "is_pumped": [True, False], "score": [0.8, 0.3]})
    pairwise = pd.DataFrame({"metric": ["precision"], "p_value_holm": [0.15]})
    pairwise.attrs.update(p_value_adjustment="Holm", multiplicity_family="all model pairs × all four metrics")
    results = SimpleNamespace(
        bootstrap_metrics=_draws(),
        sampled_events=sampled_events,
        event_metrics=pd.DataFrame(
            {"pump_hash": ["pump-A"], "model": ["Random forest"], "tp": [1], "fp": [0], "fn": [0]}
        ),
        metric_summary=_summary(),
        pairwise=pairwise,
        scored_datasets={"CatBoost classifier": scored, "Random forest": scored},
    )
    paths = export_metric_bootstrap(results, tmp_path / "test_bootstrap")
    assert set(paths) == {
        "bootstrap_metrics",
        "sampled_events",
        "event_metrics",
        "metric_summary",
        "pairwise",
        "scored_predictions",
        "metadata",
    }
    pd.testing.assert_frame_equal(pd.read_csv(paths["sampled_events"], index_col="replicate"), sampled_events)
    pd.testing.assert_frame_equal(pd.read_csv(paths["bootstrap_metrics"]), results.bootstrap_metrics)
    predictions = pd.read_csv(paths["scored_predictions"])
    assert predictions.groupby("model")["score"].apply(list).to_dict() == {
        "CatBoost classifier": [0.8, 0.3],
        "Random forest": [0.8, 0.3],
    }
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["classification_top_k"] == 1
    assert metadata["random_state"] == 42
    assert metadata["metrics"] == list(METRIC_LABELS)
    assert metadata["pairwise_inference"] == pairwise.attrs
    assert metadata["tables"]["sampled_events"]["rows"] == 2
