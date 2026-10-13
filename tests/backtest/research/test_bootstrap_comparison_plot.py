import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.research.graphs import plot_model_bootstrap_comparison


@pytest.mark.parametrize("analytical", [False, True])
def test_comparison_plot_uses_paired_decisions_not_interval_overlap(tmp_path, analytical) -> None:
    # The two leading models have disjoint CIs, but their adjusted comparison is
    # inconclusive. The plot must respect the supplied paired-test decision.
    summary = pd.DataFrame(
        {
            "point_estimate": [0.3, 0.8, 0.6],
            "ci_lower": [0.2, 0.75, 0.55],
            "ci_upper": [0.4, 0.85, 0.65],
            "alpha": [0.1, 0.1, 0.1],
        },
        index=["low", "high", "middle"],
    )
    pairwise = pd.DataFrame(
        [
            {
                "model_a": "low",
                "model_b": "high",
                "reject_h0": True,
                "better_model": "high",
            },
            {
                "model_a": "middle",
                "model_b": "low",
                "reject_h0": True,
                "better_model": "middle",
            },
            {
                "model_a": "high",
                "model_b": "middle",
                "reject_h0": False,
                "better_model": None,
            },
        ]
    )
    if analytical:
        summary["metric"] = "analytical_topk_percent_auc"
        summary["dataset_type"] = "TEST"
    output = tmp_path / "validation_comparison.png"
    fig = plot_model_bootstrap_comparison(summary, pairwise, save_path=str(output))
    try:
        assert output.stat().st_size > 0
        forest_ax, matrix_ax = fig.axes
        assert [label.get_text() for label in forest_ax.get_yticklabels()] == [
            "M1: high",
            "M2: middle",
            "M3: low",
        ]
        assert "90% pointwise" in forest_ax.get_title()
        if analytical:
            assert forest_ax.get_xlabel() == "Test Analytical Top@k%AUC (higher is better)"
            assert "exact inclusion probabilities" in fig._suptitle.get_text()
        else:
            assert forest_ax.get_xlabel() == "Validation Top@k%AUC (higher is better)"
        actual = matrix_ax.images[0].get_array()
        expected = np.array([[np.nan, 0, 1], [0, np.nan, 1], [-1, -1, np.nan]])
        np.testing.assert_allclose(actual.filled(np.nan), expected, equal_nan=True)
    finally:
        plt.close(fig)

    # Missing comparisons cannot be presented as evidence of no difference.
    with pytest.raises(ValueError, match="every unordered model pair"):
        plot_model_bootstrap_comparison(summary, pairwise.iloc[:2])
