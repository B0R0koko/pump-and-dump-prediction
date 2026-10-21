"""Exercise configurable feature generation with readable, offline pump data."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from pumps_and_dumps.core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
from pumps_and_dumps.features import writer as writer_module
from pumps_and_dumps.features.utils import load_pumps, parse_pumps, parse_utc_time


def _event(time: str = "2021-06-01 12:00:00", **overrides: str) -> dict[str, str]:
    return {"symbol": "AAA-BTC", "exchange": "BINANCE_SPOT", "time": time, **overrides}


def test_hocon_generates_selected_pump_with_complete_history_and_audit(tmp_path: Path) -> None:
    """A real HOCON-driven worker must select outputs without truncating history.

    The manifest contains AAA announcements at 10:00, 12:00, and 14:00 UTC.
    Selecting [12:00, 14:00) must write only the noon pump, while its previous
    pump count remains one because the 10:00 announcement was already known at
    the 11:45 decision cutoff. Two trades at 11:41 and 11:44 belong to the last
    five-minute window; a conspicuous trade at 11:45 must be excluded. This
    exercises actual HOCON parsing, relative paths, a spawned worker, Parquet
    output and a single asset input-range log, catching date-filter leakage and
    history loss without producing a separate record for every feature.
    """
    partition = tmp_path / "hive" / "date=2021-06-01" / "symbol=AAA-BTC"
    partition.mkdir(parents=True)
    pl.DataFrame(
        {
            TRADE_TIME: [
                datetime(2021, 6, 1, hour, minute)
                for hour, minute in [(8, 0), (9, 0), (10, 0), (11, 41), (11, 44), (11, 45), (12, 1), (12, 2)]
            ],
            PRICE: [8.0, 9.0, 10.0, 11.0, 12.0, 1000.0, 2000.0, 3000.0],
            QUANTITY: [1.0] * 8,
            IS_BUYER_MAKER: [False] * 8,
        }
    ).write_parquet(partition / "trades.parquet")
    config = tmp_path / "features.conf"
    config.write_text("""
        # Paths are relative to this config, not the command's working directory.
        hive-dir = "hive"
        feature-dir = "features"
        audit-log = "logs/audit.log"
        processes = 1
        start = "2021-06-01T12:00:00Z"
        end = "2021-06-01T14:00:00Z"
        pumps = [
          {symbol: "AAA-BTC", exchange: "BINANCE_SPOT", time: "2021-06-01 10:00:00"},
          {symbol: "AAA-BTC", exchange: "binance_spot", time: "2021-06-01T15:00:00+03:00"},
          {symbol: "AAA-BTC", exchange: "BINANCE_SPOT", time: "2021-06-01T14:00:00Z"}
        ]
        """)

    writer_module.main(["--config", str(config)])

    paths = list((tmp_path / "features" / "pumps").glob("*.parquet"))
    assert len(paths) == 1
    assert "12-00-00" in paths[0].name
    result = pl.read_parquet(paths[0])
    assert result["currency_pair"].to_list() == ["AAA-BTC"]
    assert result["num_prev_pump"].to_list() == [1]
    assert result["num_trades@5MIN"].to_list() == [2]
    records = [json.loads(line) for line in (tmp_path / "logs" / "audit.log").read_text().splitlines()]
    assert records[0]["pump_count"] == 1
    assert records[-1]["written"] == 1
    inputs = [record for record in records if record["record_type"] == "asset_inputs"]
    assert len(inputs) == 1
    assert inputs[0]["asset"] == "AAA-BTC"
    assert inputs[0]["feature_max_time"] == "2021-06-01T11:44:00.000000000Z"
    assert inputs[0]["target_min_time"] == "2021-06-01T11:45:00.000000000Z"
    assert inputs[0]["non_overlapping"] is True
    assert inputs[0]["features_before_cutoff"] is True
    assert inputs[0]["status"] == "OK"
    assert not any(record["record_type"] in {"feature", "asset_eligibility"} for record in records)


def test_cli_overrides_inline_manifest_workers_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit flags must override both ordinary and structured file settings.

    A HOCON file chooses AAA, three workers, resume=true and a relative output
    path. The command instead supplies a JSON BBB manifest, two workers,
    --no-skip-existing and another output path. A recording writer must receive
    only BBB and every explicit override, while the unmodified Hive setting
    remains relative to the config. This catches accidental mixing of manifests
    and booleans that can be enabled from a file but never disabled at runtime.
    """
    (tmp_path / "hive").mkdir()
    config = tmp_path / "features.conf"
    config.write_text(
        'hive-dir="hive"\nfeature-dir="original"\nprocesses=3\nskip-existing=true\n'
        + 'pumps=[{symbol:"AAA-BTC",exchange:"BINANCE_SPOT",time:"2021-06-01 12:00:00"}]'
    )
    pumps_file = tmp_path / "replacement.json"
    pumps_file.write_text(json.dumps([_event(symbol="BBB-BTC")]))
    captured = {}

    class RecordingWriter:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs

        def run_parallel(self, **kwargs):
            captured["run"] = kwargs

    monkeypatch.setattr(writer_module, "PumpsFeatureWriter", RecordingWriter)
    writer_module.main(
        [
            "--config",
            str(config),
            "--pumps-file",
            str(pumps_file),
            "--processes",
            "2",
            "--no-skip-existing",
            "--feature-dir",
            str(tmp_path / "override"),
        ]
    )

    assert [event.currency_pair.name for event in captured["constructor"]["pump_events"]] == ["BBB-BTC"]
    assert captured["constructor"]["data_source"].root == tmp_path / "hive"
    assert captured["constructor"]["feature_dir"] == tmp_path / "override"
    assert captured["run"] == {"cpu_count": 2, "skip_existing": False}


def test_json_config_loads_relative_external_pump_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON configuration must support an external legacy-format pump list.

    Put one AAA event in manifests/pumps.json and reference it relatively from
    config.json with underscore keys and the existing cpu-count alias. The
    recording writer must receive the exact UTC-naive noon event and one worker.
    This protects existing JSON manifests while HOCON inline lists are added,
    and catches resolution against the shell directory instead of the config.
    """
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "pumps.json").write_text(json.dumps([_event()]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"pumps_file": "manifests/pumps.json", "cpu_count": 1}))
    captured = {}

    class RecordingWriter:
        def __init__(self, **kwargs):
            captured["events"] = kwargs["pump_events"]

        def run_parallel(self, **kwargs):
            captured["run"] = kwargs

    monkeypatch.setattr(writer_module, "PumpsFeatureWriter", RecordingWriter)
    writer_module.main(["--config", str(config)])

    assert captured["events"][0].currency_pair.name == "AAA-BTC"
    assert captured["events"][0].time == datetime(2021, 6, 1, 12)
    assert captured["run"] == {"cpu_count": 1, "skip_existing": False}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pumps": None}, "non-empty list"),
        ({"pumps": []}, "non-empty list"),
        ({"pumps": [_event(), _event()]}, "duplicates"),
        ({"pumps": [{"symbol": "AAA-BTC"}]}, "exactly symbol, exchange, and time"),
        ({"pumps": [_event(time="yesterday")]}, "invalid ISO date/time"),
        ({"pumps": [_event(time="2021-06-01T12:00:00.500000Z")]}, "must use whole seconds"),
        ({"pumps": [_event(time="2021-06-01T12:00:00.000000001Z")]}, "must use whole seconds"),
        ({"pumps": [_event(exchange="UNKNOWN")]}, "exchange must be BINANCE_SPOT"),
        ({"pumps": [_event(symbol="../AAA-BTC")]}, "must be an instrument"),
        ({"pumps": [_event(symbol="AAA-USDT")]}, "BTC-quoted"),
        ({"pumps-file": "pumps.json"}, "only one of pumps and pumps-file"),
        ({"processes": 0}, "must be positive"),
        ({"start": "2021-06-02", "end": "2021-06-01"}, "later than"),
        ({"start": "2021-06-02"}, "no pump announcements"),
        ({"hive-dir": "missing-hive"}, "existing directory"),
        ({"feature-dir": "not-a-directory"}, "must be a directory"),
        ({"audit-log": "hive"}, "must be a file path"),
    ],
)
def test_invalid_config_fails_before_writer_or_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changes: dict,
    message: str,
) -> None:
    """Bad manifests and execution settings must fail before feature work starts.

    Begin with one valid AAA event, an empty local Hive directory and fresh
    output/audit paths. Each case changes one fact: missing/duplicate/malformed
    event data (including an explicit null list), unsupported instrument or exchange, conflicting manifest
    sources, invalid interval/workers, an empty selection, or an unusable path.
    The CLI must exit with argparse status 2 and a specific readable reason.
    Fractional pump times are rejected because stored event identifiers retain
    only whole seconds; accepting them would shift the downstream event anchor.
    A constructor that raises if reached, plus absent output/log paths, proves
    errors are caught before workers start or partially generated files appear.
    """
    (tmp_path / "hive").mkdir()
    (tmp_path / "not-a-directory").write_text("a file")
    config = tmp_path / "invalid.json"
    config.write_text(
        json.dumps(
            {"pumps": [_event()], "hive-dir": "hive", "feature-dir": "features", "audit-log": "audit.log", **changes}
        )
    )

    def unexpected_writer(**kwargs):
        raise AssertionError("invalid input reached the writer")

    monkeypatch.setattr(writer_module, "PumpsFeatureWriter", unexpected_writer)
    with pytest.raises(SystemExit) as error:
        writer_module.main(["--config", str(config)])

    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "audit.log").exists()


@pytest.mark.parametrize("timestamp", ["2021-06-01 12:00:00", "2021-06-01T12:00:00Z", "2021-06-01T15:00:00+03:00"])
def test_pump_timestamps_normalize_to_utc(timestamp: str) -> None:
    """Equivalent legacy, UTC and offset timestamps must identify the same pump.

    Noon with no timezone, noon with Z, and 15:00 with +03:00 all represent
    12:00 UTC in this project. Both interval parsing and pump parsing must return
    the same naive datetime. This prevents applying machine-local timezone
    offsets or comparing aware configuration values with naive Hive trades.
    """
    expected = datetime(2021, 6, 1, 12)
    assert parse_utc_time(timestamp) == expected
    assert parse_pumps([_event(time=timestamp)])[0].time == expected


def test_json_pump_loader_rejects_empty_manifest(tmp_path: Path) -> None:
    """An explicitly empty external manifest must not silently run zero workers.

    Write the complete JSON file [] and load it through the public manifest
    loader. Expect the same non-empty-list validation used for inline HOCON
    pumps, catching separate file parsing paths that bypass schema validation.
    """
    path = tmp_path / "pumps.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="non-empty list"):
        load_pumps(path)
