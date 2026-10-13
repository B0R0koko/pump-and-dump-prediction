"""Hand-checkable input histories for the feature writer's timestamp audit."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pumps_and_dumps.core.time_utils import NamedTimeDelta as W
from pumps_and_dumps.features.calculation import compute_targets, compute_window_features, prepare_trades
from pumps_and_dumps.features.provenance import timestamp_ns, utc_timestamp


def _trades(times: list[datetime | int], prices: list[float]) -> pl.DataFrame:
    timestamps = pl.Series("trade_time", [timestamp_ns(time) for time in times], dtype=pl.Int64).cast(pl.Datetime("ns"))
    return prepare_trades(
        pl.DataFrame({"trade_time": timestamps, "price": prices, "quantity": [1.0] * len(times)}),
        side=pl.lit(1),
    )


def _by_feature(records: list[dict]) -> dict[str, dict]:
    return {record["feature"]: record for record in records}


def _by_dependency(record: dict) -> dict[str, dict]:
    return {item["name"]: item for item in record["dependencies"]}


def test_audit_reports_actual_window_and_previous_price_without_changing_values() -> None:
    """At noon, only the 11:55 and 11:59 trades belong to the five-minute window.

    Their prices are 110 and 120, but return starts from the 11:54 price of 100:
    the expected return is 2,000 bps and its dependency range is 11:54–11:59.
    Other ordinary features begin at 11:55. Huge trades exactly at noon and
    one second later must appear in neither inputs nor logs. This catches
    audits that describe nominal windows, omit the lag, or include the cutoff.
    Auditing must emit exactly one record per output and preserve every value.
    """
    cutoff = datetime(2021, 6, 1, 12)
    trades = _trades(
        [cutoff - timedelta(minutes=n) for n in (6, 5, 1)] + [cutoff, cutoff + timedelta(seconds=1)],
        [100.0, 110.0, 120.0, 1_000_000.0, 2_000_000.0],
    )
    records = []
    audited = compute_window_features(trades, cutoff, [W.FIVE_MINUTES], audit=records.append)
    plain = compute_window_features(trades, cutoff, [W.FIVE_MINUTES])
    assert_frame_equal(pl.DataFrame(audited), pl.DataFrame(plain), check_exact=True)
    assert audited["asset_return@5MIN"] == pytest.approx(2_000.0)
    assert audited["num_trades@5MIN"] == 2
    assert len(records) == len(audited) == 8
    assert set(_by_feature(records)) == set(audited)
    returned = _by_feature(records)["asset_return@5MIN"]
    assert returned["data_start"] == "2021-06-01T11:54:00.000000000Z"
    assert returned["data_end"] == "2021-06-01T11:59:00.000000000Z"
    assert returned["window_start"] == "2021-06-01T11:55:00.000000000Z"
    assert returned["window_end_exclusive"] == "2021-06-01T12:00:00.000000000Z"
    inputs = _by_dependency(returned)
    assert inputs["previous_price"]["n_rows"] == 1
    assert inputs["window_trades"]["n_rows"] == 2
    assert inputs["window_trades"]["data_start"] == "2021-06-01T11:55:00.000000000Z"
    assert _by_feature(records)["flow_imbalance@5MIN"]["data_start"] == "2021-06-01T11:55:00.000000000Z"
    assert all(record["provenance_complete"] and record["role"] == "predictor" for record in records)
    assert all(record["data_end_ns"] < timestamp_ns(cutoff) for record in records)


@pytest.mark.parametrize("window", [W.FIVE_MINUTES, W.ONE_HOUR])
def test_zscore_audit_includes_actual_normalizer_trades_and_correct_numerator(window: W) -> None:
    """A normalizer uses old trades as well as trades in the requested window.

    Four trades occur at May 3 12:13, June 1 10:20, 11:54 and 11:59, with a
    fifth exactly at the noon cutoff. Both z-scores must report the first four
    actual input times, never rounded hourly bar labels or the fifth trade.
    The five-minute return numerator also depends on the 11:54 previous price;
    its quote numerator only uses 11:59. An hourly numerator uses the two raw
    trades at 11:54 and 11:59 and no previous-price denominator. This catches
    hidden lookback dependencies and incorrect reuse of short-window lineage.
    """
    cutoff = datetime(2021, 6, 1, 12)
    oldest = datetime(2021, 5, 3, 12, 13)
    trades = _trades(
        [oldest, datetime(2021, 6, 1, 10, 20), datetime(2021, 6, 1, 11, 54), datetime(2021, 6, 1, 11, 59), cutoff],
        [100.0, 105.0, 110.0, 120.0, 1_000_000.0],
    )
    records = []
    compute_window_features(trades, cutoff, [window], audit=records.append)
    for kind in ("asset_return_zscore", "quote_abs_zscore"):
        record = _by_feature(records)[f"{kind}@{window.get_slug()}"]
        inputs = _by_dependency(record)
        normalizer = inputs["hourly_normalizer"]
        assert normalizer["data_start"] == "2021-05-03T12:13:00.000000000Z"
        assert normalizer["data_end"] == "2021-06-01T11:59:00.000000000Z"
        assert normalizer["n_rows"] == 4
        assert record["data_start"] == normalizer["data_start"]
        assert record["data_end"] == normalizer["data_end"]
        if window == W.ONE_HOUR:
            assert set(inputs) == {"hourly_window", "hourly_normalizer"}
            assert inputs["hourly_window"]["data_start"] == "2021-06-01T11:54:00.000000000Z"
            assert inputs["hourly_window"]["n_rows"] == 2
        else:
            assert inputs["window_trades"]["data_start"] == "2021-06-01T11:59:00.000000000Z"
            assert inputs["window_trades"]["n_rows"] == 1
            assert ("previous_price" in inputs) == (kind == "asset_return_zscore")


def test_empty_window_has_no_invented_rows_but_retains_normalizer_dependencies() -> None:
    """Two morning trades at 09:10 and 10:20 leave the noon five-minute window empty.

    A trade count of zero must have null actual bounds, zero source rows and
    complete provenance: the absence of inputs is known, not an audit failure.
    Z-scores still read their historical normalizer, so their logged range must
    remain 09:10–10:20 even though the numerator has no rows. This catches logs
    that invent nominal input times or discard normalizer data for empty windows.
    """
    cutoff = datetime(2021, 6, 1, 12)
    trades = _trades([datetime(2021, 6, 1, 9, 10), datetime(2021, 6, 1, 10, 20)], [100.0, 120.0])
    records = []
    values = compute_window_features(trades, cutoff, [W.FIVE_MINUTES], audit=records.append)
    count = _by_feature(records)["num_trades@5MIN"]
    assert values["num_trades@5MIN"] == 0
    assert count["data_start"] is count["data_end"] is None
    assert count["dependencies"][0]["n_rows"] == 0
    assert count["provenance_complete"] is True
    normalized = _by_feature(records)["asset_return_zscore@5MIN"]
    assert normalized["data_start"] == "2021-06-01T09:10:00.000000000Z"
    assert normalized["data_end"] == "2021-06-01T10:20:00.000000000Z"
    assert _by_dependency(normalized)["hourly_normalizer"]["n_rows"] == 2


def test_missing_lag_timestamp_is_unknown_instead_of_claiming_complete_provenance() -> None:
    """A legacy prepared frame can contain a previous price without its timestamp.

    The 11:59 row has price 120 and known previous price 100, but the column
    identifying when that 100 was observed has been removed. Return and its
    five-minute z-score must remain numerically calculable at 2,000 bps while
    reporting an unknown one-row previous-price dependency and incomplete
    provenance. Trade count remains fully traceable. This prevents a false
    'safe' verdict when an out-of-window dependency cannot actually be checked.
    """
    cutoff = datetime(2021, 6, 1, 12)
    trades = _trades([cutoff - timedelta(minutes=6), cutoff - timedelta(minutes=1)], [100.0, 120.0]).drop(
        "trade_time_prev"
    )
    records = []
    values = compute_window_features(trades, cutoff, [W.FIVE_MINUTES], audit=records.append)
    assert values["asset_return@5MIN"] == pytest.approx(2_000.0)
    for kind in ("asset_return", "asset_return_zscore"):
        record = _by_feature(records)[f"{kind}@5MIN"]
        assert record["provenance_complete"] is False
        previous = _by_dependency(record)["previous_price"]
        assert previous["n_rows"] == 1
        assert previous["data_start"] is previous["data_end"] is None
    assert _by_feature(records)["num_trades@5MIN"]["provenance_complete"] is True


def test_outcome_audit_identifies_intentional_future_window_and_previous_price() -> None:
    """A pump at noon has a one-minute target using noon and 12:00:30 prices.

    With the 11:59 previous price of 100 and final in-window price of 120, the
    target is 2,000 bps. A huge trade at 12:01 is excluded by the half-open
    outcome window. The log must label this record 'outcome', give actual bounds
    11:59–12:00:30, and expose the pre-pump denominator separately. This keeps
    intentional label lookahead distinguishable from predictor leakage while
    still checking the target's own window and every input dependency.
    """
    pump = datetime(2021, 6, 1, 12)
    trades = _trades(
        [pump - timedelta(minutes=1), pump, pump + timedelta(seconds=30), pump + timedelta(minutes=1)],
        [100.0, 110.0, 120.0, 1_000_000.0],
    )
    records = []
    targets = compute_targets(trades, pump, [W.ONE_MINUTE], audit=records.append)
    assert targets["target_return@1MIN"] == pytest.approx(2_000.0)
    assert len(records) == 1
    record = records[0]
    assert record["role"] == "outcome"
    assert record["window_start"] == utc_timestamp(pump)
    assert record["window_end_exclusive"] == "2021-06-01T12:01:00.000000000Z"
    assert record["data_start"] == "2021-06-01T11:59:00.000000000Z"
    assert record["data_end"] == "2021-06-01T12:00:30.000000000Z"
    assert _by_dependency(record)["window_trades"]["n_rows"] == 2
    assert record["provenance_complete"] is True
    assert targets == compute_targets(trades, pump, [W.ONE_MINUTE])


def test_nanosecond_before_cutoff_is_preserved_in_human_and_integer_bounds() -> None:
    """Trades one nanosecond apart straddle the noon cutoff exactly.

    The 11:59:59.999999999 trade must be included, while the noon trade must
    be excluded. The audit must retain all nine fractional digits and the exact
    integer epoch nanoseconds, including the same precision in the hourly
    normalizer. Python datetime conversion would truncate these inputs to
    microseconds and could conceal a boundary error; this example catches that.
    """
    cutoff = datetime(2021, 6, 1, 12)
    boundary = timestamp_ns(cutoff)
    trades = _trades([boundary - 2, boundary - 1, boundary], [100.0, 120.0, 1_000_000.0])
    records = []
    values = compute_window_features(trades, cutoff, [W.FIVE_MINUTES], audit=records.append)
    assert values["num_trades@5MIN"] == 2
    for record in records:
        assert record["data_end"] == "2021-06-01T11:59:59.999999999Z"
        assert record["data_end_ns"] == boundary - 1
    assert utc_timestamp(boundary - 1) == "2021-06-01T11:59:59.999999999Z"
