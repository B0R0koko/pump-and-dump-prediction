import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pumps_and_dumps.backtest.portfolio.frontier import PortfolioFrontierResults
from pumps_and_dumps.backtest.research.allocation_comparison import (
    compare_portfolio_allocations,
    plot_allocation_comparison,
)


def _results(returns, drawdowns):
    paths = pd.DataFrame(
        {
            "strategy": ["model"] * 3,
            "portfolio_size": [5] * 3,
            "simulation": [0, 1, 2],
            "final_return": returns,
            "max_drawdown": drawdowns,
        }
    )
    points = paths.groupby(["strategy", "portfolio_size"], as_index=False)[["final_return", "max_drawdown"]].median()
    return PortfolioFrontierResults(points, points.copy(), paths)


def test_comparison_pairs_ids_and_distinguishes_median_changes_from_medians_of_changes(tmp_path):
    equal = _results([0.0, 0.1, 1.0], [0.1, 0.4, 0.2])
    probability = _results([0.2, 0.05, 0.5], [0.2, 0.1, 0.2])
    probability = replace(probability, sampled_paths=probability.sampled_paths.iloc[::-1])
    result = compare_portfolio_allocations(equal, probability, output_dir=tmp_path)
    row = result.sampled.iloc[0]
    assert row.final_return_delta == pytest.approx(0.1)
    assert row.paired_final_return_delta_median == pytest.approx(-0.05)
    assert row.paired_final_return_proportion_better == pytest.approx(1 / 3)
    assert row.paired_max_drawdown_proportion_better == pytest.approx(1 / 3)
    assert row.paired_max_drawdown_delta_median == 0
    np.testing.assert_allclose(result.paired_paths.final_return_delta, [0.2, -0.05, -0.5])
    assert len(pd.read_parquet(tmp_path / "paired_paths.parquet")) == 3
    assert (
        "not bootstrap confidence intervals" in json.loads((tmp_path / "comparison.json").read_text())["interpretation"]
    )
    figure = plot_allocation_comparison(result, sampled=True, save_path=tmp_path / "comparison.html")
    assert len(figure.data) == 2
    np.testing.assert_allclose(figure.data[0].y, [0.1])
    assert figure.data[0].mode == "lines"


def test_comparison_rejects_unpaired_simulation_ids():
    equal = _results([0.0, 0.1, 1.0], [0.1, 0.4, 0.2])
    probability = replace(equal, sampled_paths=equal.sampled_paths.assign(simulation=[1, 2, 3]))
    with pytest.raises(ValueError, match="identical strategies, portfolio sizes and simulation IDs"):
        compare_portfolio_allocations(equal, probability)
