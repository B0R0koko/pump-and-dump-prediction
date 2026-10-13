from datetime import datetime

import polars as pl
import pyarrow.parquet as pq
import pytest

import pumps_and_dumps.backtest.datasets.build_dataset as dataset_module
import pumps_and_dumps.features.writer as writer_module
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.features.schema import (
    FEATURE_SCHEMA_METADATA_KEY,
    FEATURE_SCHEMA_VERSION,
    regressor_columns,
)
from pumps_and_dumps.features.storage import (
    is_current_feature_file,
    validate_feature_file,
    write_feature_file,
)


def _frame():
    return pl.DataFrame({**{col: [0.0] for col in regressor_columns()}, "currency_pair": ["AAA-BTC"]})


def _event():
    return PumpEvent(CurrencyPair.from_string("AAA-BTC"), datetime(2021, 6, 1), Exchange.BINANCE_SPOT)


def test_loader_rejects_unversioned_and_incomplete_schema_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "FEATURE_DIR", tmp_path)
    path = tmp_path / "pumps" / f"{_event()}.parquet"
    path.parent.mkdir()
    pl.DataFrame({"currency_pair": ["AAA-BTC"], "num_prev_pump": [1]}).write_parquet(path)
    assert not is_current_feature_file(path)
    with pytest.raises(ValueError, match=r"Regenerate.*pumps_and_dumps\.features\.writer"):
        dataset_module._read_cross_section(_event())
    # Missing provenance is rejected even when columns appear compatible.
    _frame().write_parquet(path)
    with pytest.raises(ValueError, match="Outdated feature schema"):
        validate_feature_file(path)
    write_feature_file(_frame(), path)
    assert is_current_feature_file(path)
    event, result = dataset_module._read_cross_section(_event())
    assert event == _event()
    assert result["pump_hash"].iloc[0] == str(event)
    assert pq.read_schema(path).metadata[FEATURE_SCHEMA_METADATA_KEY] == FEATURE_SCHEMA_VERSION.encode()


@pytest.mark.parametrize("version", [b"2", b"3"])
def test_loader_reuses_original_columns_from_expanded_feature_caches(tmp_path, monkeypatch, version):
    monkeypatch.setattr(dataset_module, "FEATURE_DIR", tmp_path)
    path = tmp_path / "pumps" / f"{_event()}.parquet"
    path.parent.mkdir()
    frame = _frame().with_columns(
        pl.lit(123.0).alias("days_since_last_pump"),
        pl.lit(0.25).alias("target_return@1MIN"),
    )
    table = frame.to_arrow().replace_schema_metadata({FEATURE_SCHEMA_METADATA_KEY: version})
    pq.write_table(table, path)

    assert not is_current_feature_file(path)
    event, result = dataset_module._read_cross_section(_event())

    assert event == _event()
    assert set(regressor_columns()).issubset(result.columns)
    assert "days_since_last_pump" not in result.columns
    assert result["target_return@1MIN"].iloc[0] == 0.25


def test_failed_write_preserves_previous_complete_cache(tmp_path, monkeypatch):
    path = tmp_path / "features.parquet"
    write_feature_file(_frame(), path)
    previous = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", fail_write)
    with pytest.raises(OSError, match="disk full"):
        write_feature_file(_frame(), path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_all_null_regressor_stays_numeric_after_storage(tmp_path):
    path = tmp_path / "features.parquet"
    frame = _frame().with_columns(pl.lit(None).alias("asset_return@5MIN"))
    write_feature_file(frame, path)
    result = pq.read_table(path).to_pandas()
    assert result["asset_return@5MIN"].dtype.kind == "f"
    assert result["asset_return@5MIN"].isna().all()


def test_resume_regenerates_old_schema_and_skips_only_current_files(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, "FEATURE_DIR", tmp_path)
    path = tmp_path / "pumps" / f"{_event()}.parquet"
    path.parent.mkdir()
    old_table = _frame().to_arrow().replace_schema_metadata({FEATURE_SCHEMA_METADATA_KEY: b"2"})
    pq.write_table(old_table, path)
    assert not is_current_feature_file(path)
    writer = object.__new__(writer_module.PumpsFeatureWriter)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _frame()

    monkeypatch.setattr(writer, "create_cross_section", create)
    writer._write_cross_section(_event(), skip_existing=True)
    writer._write_cross_section(_event(), skip_existing=True)
    assert len(calls) == 1
    assert is_current_feature_file(path)


def test_regeneration_removes_cached_events_that_are_no_longer_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, "FEATURE_DIR", tmp_path)
    path = tmp_path / "pumps" / f"{_event()}.parquet"
    write_feature_file(_frame(), path)
    writer = object.__new__(writer_module.PumpsFeatureWriter)
    monkeypatch.setattr(writer, "create_cross_section", lambda **kwargs: None)
    writer._write_cross_section(_event())
    assert not path.exists()
