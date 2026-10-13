"""Small input/output examples for the notebook's data leakage boundaries."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.preprocessing import cross_section_standardize
from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType
from pumps_and_dumps.backtest.datasets.splitting import split_model_selection_datasets
from pumps_and_dumps.backtest.pipelines.base_pipeline import impute_training_features


def test_ten_events_have_exact_splits_even_when_future_outcomes_change():
    """Assign whole pump events to chronological splits independently of outcomes.

    Ten consecutive dates A-J each contain two candidate assets; reversing the
    input rows makes the result depend on timestamps rather than input order.
    The 60/10/20/10 split must yield A-F for training, G for tuning, H-I for
    comparison, and J for testing, preserving every row and its original index.
    Flip every true label and drastically change future returns, then expect
    the same event membership and otherwise unchanged source rows.
    This catches outcome-driven splitting, dropped candidates, duplicate rows,
    or a pump's assets being scattered across different dataset partitions.
    """
    frame = pd.DataFrame(
        {
            "pump_hash": np.repeat(list("ABCDEFGHIJ"), 2),
            "pump_time": np.repeat(pd.date_range("2020-01-01", periods=10), 2),
            "currency_pair": ["AAA-BTC", "BBB-BTC"] * 10,
            "is_pumped": [True, False] * 10,
            "target_return@1MIN": [-0.5, 2.0] * 10,
        }
    ).iloc[
        ::-1
    ]  # Input order must not decide which period a row belongs to.
    expected_events = {
        DatasetType.TRAIN: set("ABCDEF"),
        DatasetType.TUNING: {"G"},
        DatasetType.VALIDATION: {"H", "I"},
        DatasetType.TEST: {"J"},
    }
    changed = frame.assign(is_pumped=~frame["is_pumped"])
    changed["target_return@1MIN"] = -1_000_000 * frame["target_return@1MIN"]

    for source in (frame, changed):
        parts = split_model_selection_datasets(source)
        for kind, event_ids in expected_events.items():
            # Compare whole rows, including order/index: no dropped candidates,
            # repeated events, reassigned labels, or filtering failed pumps.
            assert_frame_equal(parts[kind], source.loc[source["pump_hash"].isin(event_ids)])
        assert_frame_equal(pd.concat(parts.values()).sort_index(), source.sort_index())


def test_imputation_uses_training_median_two_despite_extreme_held_out_values():
    """Fill missing values using statistics fitted only on the training period.

    Events A-F form training and contain only two observed signal values, 1
    and 3, so every missing signal in every partition must become median 2.
    A feature absent throughout training must become zero, and the input
    frame must remain unchanged. Held-out observations 999 and 10,000 remain
    intact because imputation only fills missing entries.
    Replace those observations with -1e12 and 1e12 and still expect every
    imputed signal to be 2. This detects fitting preprocessing on held-out
    rows, recomputing statistics per partition, and unintended input mutation.
    """
    frame = pd.DataFrame(
        {
            "pump_hash": list("ABCDEFGHIJ"),
            "pump_time": pd.date_range("2020-01-01", periods=10),
            "signal": [1.0, 3.0, np.nan, np.nan, np.nan, np.nan, 999.0, np.nan, 10_000.0, np.nan],
            "absent_feature": np.nan,
        }
    )
    original = frame.copy(deep=True)
    features = FeatureSet(numeric_features=["signal", "absent_feature"], target="is_pumped")
    expected = frame.assign(
        signal=[1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 999.0, 2.0, 10_000.0, 2.0],
        absent_feature=0.0,
    )
    assert_frame_equal(impute_training_features(frame, features), expected)
    assert_frame_equal(frame, original)

    # Changing tuning/validation observations cannot change any filled value.
    changed = frame.copy(deep=True)
    changed.loc[[6, 8], "signal"] = [-1e12, 1e12]
    expected.loc[[6, 8], "signal"] = [-1e12, 1e12]
    assert_frame_equal(impute_training_features(changed, features), expected)


def test_standardization_uses_only_other_assets_in_the_same_pump():
    """Standardize each pump's candidate assets without mixing other events.

    Interleaved past/future rows contain signals [1, 2, 3] and [100, 200, 300].
    Each group has z-scores [-1, 0, 1]: the past mean is 2 and sample SD is 1;
    the future mean is 200 and sample SD is 100. Constant features become 0.
    Check the complete output, including preserved row order and labels,
    reset row indices, and pump IDs [0, 1, 0, 1, 0, 1].
    Changing only future signals to [-1e12, 0, 1e12] leaves both groups'
    expected standardized values intact. This catches pooling statistics
    across pump events and leaking later observations into earlier features.
    """
    frame = pd.DataFrame(
        {
            "pump_hash": ["past", "future", "past", "future", "past", "future"],
            "signal": [1.0, 100.0, 2.0, 200.0, 3.0, 300.0],
            "constant": [7.0, 99.0, 7.0, 99.0, 7.0, 99.0],
            "is_pumped": [True, False, False, True, False, False],
        },
        index=[8, 3, 9, 2, 7, 1],
    )
    expected = frame.reset_index(drop=True).assign(
        signal=[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], constant=0.0, pump_id=[0, 1, 0, 1, 0, 1]
    )
    assert_frame_equal(cross_section_standardize(frame, ["signal", "constant"]), expected)

    changed = frame.copy(deep=True)
    changed.loc[changed["pump_hash"].eq("future"), "signal"] = [-1e12, 0.0, 1e12]
    assert_frame_equal(cross_section_standardize(changed, ["signal", "constant"]), expected)


def test_model_input_contains_only_declared_features_in_declared_order():
    """Keep target labels and future-return metadata outside the feature matrix.

    Three indexed candidate rows carry two valid predictors alongside a true
    label, the known pumped asset, and a future one-minute return. Declare
    features in [volume, signal] order, opposite to their input column order,
    and expect exactly those two columns with values and indices unchanged.
    Change labels, the pumped asset, and future returns by extreme amounts;
    the model matrix must remain identical. Labels and evaluation returns
    must still be available, separately, through their dedicated accessors.
    This catches accidental inclusion of outcome columns or column reordering
    while ensuring outcome exclusion does not corrupt downstream evaluation.
    """
    frame = pd.DataFrame(
        {
            "signal": [0.1, 0.2, 0.3],
            "volume": [10.0, 20.0, 30.0],
            "is_pumped": [False, True, False],
            "pumped_currency_pair": ["BBB-BTC"] * 3,
            "target_return@1MIN": [-0.1, 9.0, 0.2],
        },
        index=[7, 2, 9],
    )
    features = FeatureSet(numeric_features=["volume", "signal"], target="is_pumped", eval_fields=["target_return@1MIN"])
    expected = pd.DataFrame({"volume": [10.0, 20.0, 30.0], "signal": [0.1, 0.2, 0.3]}, index=[7, 2, 9])
    changed = frame.assign(is_pumped=[True, False, False], pumped_currency_pair="AAA-BTC")
    changed["target_return@1MIN"] = [1e9, -1e9, 0.0]

    for source in (frame, changed):
        dataset = Dataset(source, features, DatasetType.TRAIN)
        assert_frame_equal(dataset.get_data(), expected)
        assert dataset.get_label().tolist() == source["is_pumped"].tolist()
        assert_frame_equal(dataset.get_eval_data(), source[["target_return@1MIN"]])
