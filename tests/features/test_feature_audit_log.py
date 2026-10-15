"""Small on-disk examples make feature provenance and writer logging inspectable.

The fixture has two assets and two pumps on one day. Each pump has exactly two
trades in its last five-minute predictor window, plus conspicuous trades at the
decision cutoff and after the announcement. Real Hive reads, feature calculation,
Parquet writes, and JSON-lines output are exercised without market data access.
"""

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import pumps_and_dumps.features.writer as writer_module
from pumps_and_dumps.core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.features.audit import FeatureAuditLog
from pumps_and_dumps.features.source import HiveTradeSource
from pumps_and_dumps.features.writer import DECISION_LAG, PumpsFeatureWriter


def _tiny_hive(root: Path) -> tuple[HiveTradeSource, list[PumpEvent]]:
    """Write two assets with readable times, including explicitly forbidden rows."""
    events = [
        PumpEvent(CurrencyPair.from_string("AAA-BTC"), datetime(2021, 6, 1, 12), Exchange.BINANCE_SPOT),
        PumpEvent(CurrencyPair.from_string("BBB-BTC"), datetime(2021, 6, 1, 14), Exchange.BINANCE_SPOT),
    ]
    for symbol in ("AAA-BTC", "BBB-BTC"):
        rows = []
        for event in events:
            cutoff = event.time - DECISION_LAG
            for timestamp, price, quantity in (
                (cutoff - timedelta(hours=3), 8.0, 1.0),
                (cutoff - timedelta(hours=2), 9.0, 2.0),
                (cutoff - timedelta(hours=1), 10.0, 3.0),
                (cutoff - timedelta(minutes=4), 11.0, 4.0),
                (cutoff - timedelta(minutes=1), 12.0, 5.0),
                (cutoff, 1_000.0, 1_000.0),
                (event.time + timedelta(seconds=30), 2_000.0, 2_000.0),
                (event.time + timedelta(minutes=2), 3_000.0, 3_000.0),
            ):
                rows.append(
                    {
                        TRADE_TIME: timestamp,
                        PRICE: price,
                        QUANTITY: quantity,
                        IS_BUYER_MAKER: False,
                    }
                )
        partition = root / "date=2021-06-01" / f"symbol={symbol}"
        partition.mkdir(parents=True)
        pl.DataFrame(rows).sort(TRADE_TIME).write_parquet(partition / "trades.parquet")
    return HiveTradeSource(root), events


def _records(path: Path) -> list[dict]:
    """Parse every physical line so concurrent interleaving cannot go unnoticed."""
    lines = path.read_text().splitlines()
    assert lines
    assert all(line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def _asset_inputs(records: list[dict]) -> list[dict]:
    """Select the single temporal summary written for each computed asset."""
    return [record for record in records if record["record_type"] == "asset_inputs"]


@pytest.mark.parametrize("filename", ["audit", "audit.jsonl", "audit.log"])
def test_audit_uses_log_suffix_and_logging_file_handler(tmp_path, monkeypatch, filename):
    """Custom audit names produce .log files with unchanged JSON provenance.

    Capture actual standard FileHandler emissions while appending two batches.
    Both original records must be present exactly once, with the same run ID,
    and every short-lived handler must be closed. This proves the audit uses
    Python logging without duplicate handlers or changing its parseable format.
    """
    audit = FeatureAuditLog.for_run(tmp_path, tmp_path / filename)
    emitted = []
    original_emit = logging.FileHandler.emit

    def capture_emit(handler, record):
        emitted.append((handler, record))
        original_emit(handler, record)

    monkeypatch.setattr(logging.FileHandler, "emit", capture_emit)
    audit.append([{"record_type": "run_start"}])
    audit.append([{"record_type": "run_end"}])
    assert audit.path == tmp_path / "audit.log"
    records = _records(audit.path)
    assert [record["record_type"] for record in records] == ["run_start", "run_end"]
    assert {record["run_id"] for record in records} == {audit.run_id}
    assert len(emitted) == 2
    assert all(handler.stream is None for handler, _record in emitted)


def test_audit_write_failure_propagates_and_releases_file(tmp_path, monkeypatch):
    """A feature run must fail visibly when an audit record cannot be written.

    Make the real log file's write method raise OSError, keeping its actual file
    descriptor so the process lock is exercised. The exception must escape
    logging and the file must close; a subsequent normal append must succeed,
    proving the failed attempt left no held lock or fabricated record behind.
    """
    audit = FeatureAuditLog.for_run(tmp_path)
    opened = []
    original_open = logging.FileHandler._open

    def failing_open(handler):
        stream = original_open(handler)
        opened.append(stream)

        def failed_write(_message):
            raise OSError("disk full")

        monkeypatch.setattr(stream, "write", failed_write)
        return stream

    with monkeypatch.context() as patched:
        patched.setattr(logging.FileHandler, "_open", failing_open)
        with pytest.raises(OSError, match="disk full"):
            audit.append([{"record_type": "run_start"}])
    assert all(stream.closed for stream in opened)
    audit.append([{"record_type": "run_end"}])
    assert [record["record_type"] for record in _records(audit.path)] == ["run_end"]


def _assert_complete_event_audit(records: list[dict], events: list[PumpEvent]) -> None:
    """Require exactly one truthful, compact temporal check per pump and asset."""
    inputs = _asset_inputs(records)
    assert len(inputs) == len(events) * 2
    assert len({record["run_id"] for record in records}) == 1
    assert not any(record["record_type"] in {"feature", "asset_eligibility"} for record in records)
    for event in events:
        cutoff = event.time - DECISION_LAG
        for asset in ("AAA-BTC", "BBB-BTC"):
            matching = [r for r in inputs if r["pump_hash"] == str(event) and r["asset"] == asset]
            assert len(matching) == 1
            record = matching[0]
            assert record["pump_time"] == f"{event.time.isoformat()}.000000000Z"
            assert record["decision_cutoff"] == f"{cutoff.isoformat()}.000000000Z"
            assert record["feature_max_time"] == f"{(cutoff - timedelta(minutes=1)).isoformat()}.000000000Z"
            assert record["target_min_time"] == f"{cutoff.isoformat()}.000000000Z"
            assert record["provenance_complete"] is True
            assert record["non_overlapping"] is True
            assert record["features_before_cutoff"] is True
            assert record["status"] == "OK"
            assert "dependencies" not in record
            assert "feature" not in record


def test_default_log_summarizes_actual_inputs_once_per_asset(tmp_path: Path) -> None:
    """A two-asset run reports actual feature and target bounds in two records.

    Both assets have predictor trades at 11:41 and 11:44, followed by a trade
    at the 11:45 decision cutoff and a target trade after the noon pump. The
    target return uses that 11:45 price as its denominator, so the audit must
    report feature_max_time=11:44 and target_min_time=11:45, rather than hiding
    the target baseline behind the noon window boundary. Both strict separation
    checks must be true. Saved five-minute trade counts stay unchanged,
    and the complete log stays below 5 KB instead of repeating 86 feature rows
    for each asset. Lifecycle and event records remain available for inspection.
    """
    source, events = _tiny_hive(tmp_path / "hive")
    feature_dir = tmp_path / "features"
    writer = PumpsFeatureWriter(events, data_source=source, feature_dir=feature_dir)

    assert writer.audit_log_path.parent == feature_dir / "logs"
    assert writer.audit_log_path.name.startswith("feature_audit_")
    assert writer.audit_log_path.suffix == ".log"
    assert not writer.audit_log_path.exists()
    writer.run(events[:1])

    records = _records(writer.audit_log_path)
    assert records[0]["record_type"] == "run_start"
    assert records[0]["audit_schema_version"] == 2
    assert records[-1]["record_type"] == "run_end"
    _assert_complete_event_audit(records, events[:1])
    event_records = [r for r in records if r["record_type"] == "event"]
    assert [(r["pump_hash"], r["status"]) for r in event_records] == [(str(events[0]), "written")]
    saved = pl.read_parquet(feature_dir / "pumps" / f"{events[0]}.parquet")
    assert saved["num_trades@5MIN"].to_list() == [2, 2]
    assert writer.audit_log_path.stat().st_size < 5_000
    for record in _asset_inputs(records):
        for key in ("pump_time", "decision_cutoff", "feature_max_time", "target_min_time"):
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z", record[key])


@pytest.mark.parametrize("entrypoint", ["create_cross_section", "compute_features"])
def test_direct_calculation_entrypoints_also_start_an_audit(tmp_path: Path, entrypoint: str) -> None:
    """Calling a calculation directly must not silently bypass the audit file.

    Use the same two-asset Hive fixture through a normally constructed writer,
    but omit run/run_parallel. A cross-section call computes both assets and
    therefore writes two asset summaries; a direct computation of AAA alone
    writes one. The explicit log path starts absent and is created on first use,
    with run metadata and the real pump/asset identifiers on each summary.
    This guards notebook and programmatic callers as well as the command line.
    """
    source, events = _tiny_hive(tmp_path / "hive")
    destination = tmp_path / "custom" / "audit.log"
    writer = PumpsFeatureWriter(
        events, data_source=source, feature_dir=tmp_path / "features", audit_log_path=destination
    )
    assert writer.audit_log_path == destination
    assert not destination.exists()
    if entrypoint == "create_cross_section":
        frame = writer.create_cross_section(events[0], position=0)
        assert frame is not None and frame.height == 2
        expected_assets = {"AAA-BTC", "BBB-BTC"}
    else:
        ticks = source.load(Bounds(datetime(2021, 6, 1), datetime(2021, 6, 2)), events[0].currency_pair)
        trades = writer.preprocess_data_for_currency(ticks)
        result = writer.compute_features(trades, events[0].currency_pair, events[0])
        assert result["num_trades@5MIN"] == 2
        expected_assets = {"AAA-BTC"}
    records = _records(destination)
    assert records[0]["record_type"] == "run_start"
    inputs = _asset_inputs(records)
    assert len(inputs) == len(expected_assets)
    assert {record["asset"] for record in inputs} == expected_assets
    assert {record["pump_hash"] for record in inputs} == {str(events[0])}
    assert all(record["non_overlapping"] is True for record in inputs)


def test_parallel_workers_append_complete_records_to_one_log(tmp_path: Path) -> None:
    """Two real spawned workers must share one parseable, complete audit file.

    The fixture has a noon AAA pump and a 14:00 BBB pump, each with the same two
    eligible assets. Run the actual multiprocessing path with two workers and
    require exactly four asset summaries, one shared run identifier, both written
    event statuses, and both saved Parquets. Parsing every JSON line detects
    interleaved writes; exact per-event/asset coverage detects records
    lost or duplicated while workers concurrently append to the same file.
    """
    source, events = _tiny_hive(tmp_path / "hive")
    feature_dir = tmp_path / "features"
    destination = tmp_path / "shared" / "parallel.log"
    writer = PumpsFeatureWriter(events, data_source=source, feature_dir=feature_dir, audit_log_path=destination)
    writer.run_parallel(cpu_count=2)

    records = _records(destination)
    assert records[0]["record_type"] == "run_start"
    assert records[-1]["record_type"] == "run_end"
    _assert_complete_event_audit(records, events)
    event_records = [record for record in records if record["record_type"] == "event"]
    assert {(r["pump_hash"], r["status"]) for r in event_records} == {(str(event), "written") for event in events}
    assert len(event_records) == 2
    for event in events:
        assert pl.read_parquet(feature_dir / "pumps" / f"{event}.parquet").height == 2


def test_reused_cache_is_marked_not_recomputed_without_inventing_provenance(tmp_path: Path) -> None:
    """A skipped cache must not be falsely certified as freshly audited input.

    First write a real current-schema cache from the two-asset fixture, then
    construct a second writer with a separate audit destination and resume with
    skip_existing=True. The second log must say that the event was reused and
    its provenance was not recomputed; it must contain no asset-input summaries.
    Byte-for-byte preservation of the cache additionally proves the resume path
    really skipped computation instead of hiding an unexpected rewrite.
    """
    source, events = _tiny_hive(tmp_path / "hive")
    feature_dir = tmp_path / "features"
    original = PumpsFeatureWriter(events, data_source=source, feature_dir=feature_dir)
    original.run(events[:1])
    cache = feature_dir / "pumps" / f"{events[0]}.parquet"
    saved_bytes = cache.read_bytes()
    resumed = PumpsFeatureWriter(
        events,
        data_source=source,
        feature_dir=feature_dir,
        audit_log_path=tmp_path / "resumed.log",
    )
    resumed.run(events[:1], skip_existing=True)

    records = _records(resumed.audit_log_path)
    assert not _asset_inputs(records)
    event_records = [record for record in records if record["record_type"] == "event"]
    assert len(event_records) == 1
    assert event_records[0]["pump_hash"] == str(events[0])
    assert event_records[0]["status"] == "reused"
    assert event_records[0]["audit_provenance"] == "not_recomputed"
    assert cache.read_bytes() == saved_bytes


@pytest.mark.parametrize("lag_timestamp", ["future", "missing"])
def test_audit_flags_unsafe_or_untraceable_lag_dependencies(tmp_path: Path, lag_timestamp: str) -> None:
    """The asset summary includes lagged prices, not just visible trade rows.

    Calculate the noon pump's features after damaging only lag provenance.
    In the future case, the five-minute return's denominator claims a 12:30
    timestamp, beyond the 11:45 decision cutoff and the 11:45 target baseline.
    The single asset summary must report LEAKAGE, the true 12:30 maximum, and
    false cutoff/separation checks. When the lag timestamp column is missing,
    the summary must instead report UNKNOWN with incomplete provenance and
    null checks; nominal window boundaries cannot certify unknown inputs.
    Numerical features still run, so this tests audit evidence independently
    of the return calculation rather than causing an unrelated schema error.
    """
    source, events = _tiny_hive(tmp_path / "hive")
    writer = PumpsFeatureWriter(events, data_source=source, feature_dir=tmp_path / "features")
    event = events[0]
    ticks = source.load(Bounds(datetime(2021, 6, 1), datetime(2021, 6, 2)), event.currency_pair)
    trades = writer.preprocess_data_for_currency(ticks)
    if lag_timestamp == "future":
        trades = trades.with_columns(
            pl.when(pl.col(TRADE_TIME) == datetime(2021, 6, 1, 11, 41))
            .then(pl.lit(datetime(2021, 6, 1, 12, 30)))
            .otherwise(pl.col("trade_time_prev"))
            .alias("trade_time_prev")
        )
    else:
        trades = trades.drop("trade_time_prev")

    features = writer.compute_features(trades, event.currency_pair, event)

    inputs = _asset_inputs(_records(writer.audit_log_path))
    assert len(inputs) == 1
    record = inputs[0]
    if lag_timestamp == "future":
        assert record["status"] == "LEAKAGE"
        assert record["feature_max_time"] == "2021-06-01T12:30:00.000000000Z"
        assert record["features_before_cutoff"] is False
        assert record["non_overlapping"] is False
    else:
        assert record["status"] == "UNKNOWN"
        assert record["provenance_complete"] is False
        assert record["features_before_cutoff"] is None
        assert record["non_overlapping"] is None
    assert features["num_trades@5MIN"] == 2


def test_target_baseline_shared_with_predictors_is_reported_as_overlap(tmp_path: Path) -> None:
    """A sparse target's previous-price denominator counts as a shared input.

    Remove the noon pump's 11:45 trade so its first target trade at 12:00:30
    has to use the 11:44 price (12) as the return baseline. That same 11:44 row
    is the last predictor input. Even though every predictor strictly precedes
    the cutoff and the target window starts at noon, equality of actual input
    endpoints must produce OVERLAP and non_overlapping=False. The target's
    numerical return remains (2000 / 12 - 1) * 10000; logging must expose this
    existing calculation rather than silently changing its financial meaning.
    """
    source, events = _tiny_hive(tmp_path / "hive")
    writer = PumpsFeatureWriter(events, data_source=source, feature_dir=tmp_path / "features")
    event = events[0]
    ticks = source.load(Bounds(datetime(2021, 6, 1), datetime(2021, 6, 2)), event.currency_pair)
    ticks = ticks.filter(pl.col(TRADE_TIME) != event.time - DECISION_LAG)
    trades = writer.preprocess_data_for_currency(ticks)

    features = writer.compute_features(trades, event.currency_pair, event)

    inputs = _asset_inputs(_records(writer.audit_log_path))
    assert len(inputs) == 1
    record = inputs[0]
    assert record["feature_max_time"] == "2021-06-01T11:44:00.000000000Z"
    assert record["target_min_time"] == record["feature_max_time"]
    assert record["features_before_cutoff"] is True
    assert record["non_overlapping"] is False
    assert record["provenance_complete"] is True
    assert record["status"] == "OVERLAP"
    assert features["num_trades@5MIN"] == 2
    assert features["target_return@1MIN"] == pytest.approx((2_000 / 12 - 1) * 10_000)


def test_cli_passes_custom_audit_path_to_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The user-facing --audit-log option must reach the writer unchanged.

    Invoke the real argument parser with a nested .log path, two workers, and
    skip_existing. Replace pump loading and the writer with small recording
    doubles so the test cannot launch a production rebuild. The constructor
    must receive the requested Path and event list, and run_parallel must keep
    its independently requested worker/resume options. This catches a CLI flag
    that parses successfully but never configures the actual audit destination.
    """
    destination = tmp_path / "custom" / "audit.log"
    captured = {}
    events = [PumpEvent(CurrencyPair.from_string("AAA-BTC"), datetime(2021, 6, 1, 12), Exchange.BINANCE_SPOT)]

    class RecordingWriter:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs

        def run_parallel(self, **kwargs):
            captured["run"] = kwargs

    monkeypatch.setattr(writer_module, "PumpsFeatureWriter", RecordingWriter)
    monkeypatch.setattr(writer_module, "load_pumps", lambda **kwargs: events)
    monkeypatch.setattr(writer_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys, "argv", ["pumps-compute-features", "--audit-log", str(destination), "--cpu-count", "2", "--skip-existing"]
    )

    writer_module.main()

    assert captured["constructor"]["pump_events"] == events
    assert captured["constructor"]["audit_log_path"] == destination
    assert captured["run"] == {"cpu_count": 2, "skip_existing": True}
