import pandas as pd
import pytest

from pumps_and_dumps.backtest.datasets.columns import COL_PUMP_HASH, COL_PUMP_TIME
from pumps_and_dumps.backtest.datasets.sample import DatasetType
from pumps_and_dumps.backtest.datasets.splitting import split_model_selection_datasets


def _events(days: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COL_PUMP_HASH: f"pump-{event}",
                COL_PUMP_TIME: pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "candidate": candidate,
            }
            for event, day in enumerate(days)
            for candidate in range(100 if event == 0 else 1 + event % 7)
        ]
    )


@pytest.mark.parametrize(
    "event_count, expected_counts",
    [(20, [12, 2, 4, 2]), (517, [310, 52, 103, 52])],
)
def test_splits_count_events_and_ignore_candidate_count_and_input_order(event_count, expected_counts):
    frame = _events(list(range(event_count)))
    shuffled = frame.sample(frac=1, random_state=17)
    splits = split_model_selection_datasets(shuffled)
    original_splits = split_model_selection_datasets(frame)

    assert [part[COL_PUMP_HASH].nunique() for part in splits.values()] == expected_counts
    for kind, part in splits.items():
        assert set(part[COL_PUMP_HASH]) == set(original_splits[kind][COL_PUMP_HASH])
    parts = list(splits.values())
    for earlier, later in zip(parts, parts[1:]):
        assert earlier[COL_PUMP_TIME].max() < later[COL_PUMP_TIME].min()
    pd.testing.assert_frame_equal(pd.concat(parts).sort_index(), frame)


def test_simultaneous_events_stay_together_and_equidistant_cut_prefers_earlier_time():
    frame = _events(list(range(11)) + [11, 11] + list(range(12, 19)))
    splits = split_model_selection_datasets(frame)

    assert [part[COL_PUMP_HASH].nunique() for part in splits.values()] == [11, 3, 4, 2]
    assert {"pump-11", "pump-12"} <= set(splits[DatasetType.TUNING][COL_PUMP_HASH])


def test_minimal_split_leaves_each_partition_nonempty():
    splits = split_model_selection_datasets(_events([0, 1, 2, 3]))
    assert [part[COL_PUMP_HASH].nunique() for part in splits.values()] == [1, 1, 1, 1]


@pytest.mark.parametrize("days", [[], [0, 0, 1, 2]])
def test_requires_four_distinct_event_times(days):
    frame = _events(days).reindex(columns=[COL_PUMP_HASH, COL_PUMP_TIME, "candidate"])
    with pytest.raises(ValueError, match="four distinct event times"):
        split_model_selection_datasets(frame)


@pytest.mark.parametrize("column", [COL_PUMP_HASH, COL_PUMP_TIME])
def test_missing_event_identity_or_time_is_rejected(column):
    frame = _events([0, 1, 2, 3])
    frame.loc[0, column] = None
    with pytest.raises(ValueError, match="must not be missing"):
        split_model_selection_datasets(frame)


def test_one_event_cannot_have_multiple_timestamps():
    frame = _events([0, 1, 2, 3])
    frame.loc[0, COL_PUMP_TIME] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="exactly one timestamp"):
        split_model_selection_datasets(frame)
