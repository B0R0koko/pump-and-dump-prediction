import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.research.portfolio_frontier import (
    evaluate_portfolio_frontiers,
    evaluate_validation_portfolio_frontiers,
)


def _experiment(
    name: str, reverse: bool = False, dataset_type: DatasetType = DatasetType.VALIDATION
) -> tuple[Experiment, Dataset, Mock]:
    frame = pd.DataFrame(
        {
            "pump_hash": ["event"] * 3,
            "currency_pair": ["AAA-BTC", "BBB-BTC", "CCC-BTC"],
            "pump_time": [pd.Timestamp("2021-01-01")] * 3,
            "is_pumped": [True, False, False],
            "signal": [1.0, 0.0, 0.0],
        }
    )
    if reverse:
        frame = frame.iloc[::-1].reset_index(drop=True)
    dataset = Dataset(frame, FeatureSet(["signal"], "is_pumped"), dataset_type)

    def selected_split_only(split):
        assert split is dataset_type
        return dataset

    rank = Mock(side_effect=lambda data: data.get_data()["signal"].to_numpy())
    model = SimpleNamespace(ranking_score_kind="probability", rank=rank)
    return Experiment(name, model, SimpleNamespace(get_dataset=selected_split_only)), dataset, rank


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pump_hash": ["event"] * 3,
            "currency_pair": ["CCC-BTC", "AAA-BTC", "BBB-BTC"],
            "pump_time": [pd.Timestamp("2021-01-01")] * 3,
            "net_return": [-0.2, 0.5, 0.0],
            "executed": [True, True, False],
            "is_pumped": False,
        }
    )


@pytest.mark.parametrize("dataset_type", [DatasetType.VALIDATION, DatasetType.TEST])
@pytest.mark.parametrize("allocation", ["equal", "probability"])
def test_frontiers_score_only_selected_split_once_reuse_outcomes_and_export_without_mutation(
    tmp_path, monkeypatch, dataset_type, allocation
):
    first, forward_data, forward_rank = _experiment("forward", dataset_type=dataset_type)
    second, reverse_data, reverse_rank = _experiment("reverse", reverse=True, dataset_type=dataset_type)
    outcomes = _outcomes()
    sources = [forward_data.all_data(), reverse_data.all_data(), outcomes]
    snapshots = [frame.copy(deep=True) for frame in sources]
    collector = Mock(side_effect=AssertionError("Explicit outcomes must be reused"))
    monkeypatch.setattr("pumps_and_dumps.backtest.research.portfolio_frontier.collect_portfolio_outcomes", collector)

    result = evaluate_portfolio_frontiers(
        [first, second],
        dataset_type=dataset_type,
        outcomes=outcomes,
        portfolio_sizes=(size for size in [1, 2, 3]),
        n_simulations=31,
        random_state=17,
        allocation=allocation,
        output_dir=tmp_path,
    )

    forward_rank.assert_called_once()
    reverse_rank.assert_called_once()
    collector.assert_not_called()
    for source, snapshot in zip(sources, snapshots, strict=True):
        assert_frame_equal(source, snapshot)
    top1 = result.deterministic.loc[result.deterministic.portfolio_size.eq(1)]
    np.testing.assert_allclose(top1.final_return, 0.5)
    np.testing.assert_allclose(top1.max_drawdown, 0)
    top3 = result.deterministic.loc[result.deterministic.portfolio_size.eq(3)]
    np.testing.assert_allclose(top3.final_return, 0.5 if allocation == "probability" else 0.1)
    assert len(result.deterministic) == 6
    assert len(result.sampled) == 9
    assert len(result.sampled_paths) == 3 * 3 * 31
    assert {"is_pareto", "is_global_pareto"}.issubset(result.sampled.columns)

    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["allocation"] == allocation
    split = dataset_type.name.lower()
    other_split = "test" if dataset_type is DatasetType.VALIDATION else "validation"
    assert metadata["dataset"] == dataset_type.name
    assert metadata["portfolio_sizes"] == [1, 2, 3]
    assert metadata["n_simulations"] == 31
    assert metadata[f"n_{split}_events"] == 1
    assert f"n_{other_split}_events" not in metadata
    assert metadata["n_candidates"] == 3
    assert metadata["n_unexecuted"] == 1
    assert metadata["score_kinds"] == {"forward": "probability", "reverse": "probability"}
    assert "portfolio_size" not in metadata
    assert len(pd.read_csv(tmp_path / "deterministic.csv")) == 6
    assert len(pd.read_csv(tmp_path / "sampled.csv")) == 9
    assert len(pd.read_parquet(tmp_path / f"{split}_scores.parquet")) == 6
    assert not (tmp_path / f"{other_split}_scores.parquet").exists()
    assert not (tmp_path / f"{other_split}_outcomes.parquet").exists()
    assert_frame_equal(pd.read_parquet(tmp_path / "sampled_paths.parquet"), result.sampled_paths)
    saved_outcomes = pd.read_parquet(tmp_path / f"{split}_outcomes.parquet").set_index("currency_pair")
    assert saved_outcomes.loc["AAA-BTC", "is_pumped"]
    assert not saved_outcomes.loc["BBB-BTC", "executed"]
    assert saved_outcomes.loc["BBB-BTC", "net_return"] == 0


def test_frontiers_reject_train_before_scoring():
    experiment, _, rank = _experiment("model")
    with pytest.raises(ValueError, match="validation or test"):
        evaluate_portfolio_frontiers([experiment], dataset_type=DatasetType.TRAIN, outcomes=_outcomes())
    rank.assert_not_called()


def test_frontiers_reject_extra_or_missing_outcomes_without_candidate_filtering():
    experiment, _, _ = _experiment("model")
    outcomes = _outcomes()
    extra = pd.concat([outcomes, outcomes.iloc[[0]].assign(currency_pair="EXTRA-BTC")], ignore_index=True)
    for invalid in [extra, outcomes.iloc[1:]]:
        with pytest.raises(ValueError):
            evaluate_validation_portfolio_frontiers(
                [experiment],
                outcomes=invalid,
                portfolio_sizes=[1],
                n_simulations=2,
            )
