"""Hand-checkable configuration and instrument filtering contracts for Hive jobs."""

import json
import zipfile
from datetime import date, datetime, timezone

import polars as pl
import pytest

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.preprocessing.pipelines import binance_spot_trades_to_hive as module


def _write_archive(raw_dir, pair, day):
    """Write two valid trades with easily checked prices and UTC timestamps."""
    path = raw_dir / pair / f"trades@{day}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    rows = f"1,10,2,20,{timestamp},False,True\n2,20,3,60,{timestamp + 1000},True,True\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trades.csv", rows)


def test_hocon_conversion_filters_real_archives_by_pair_and_exclusive_dates(tmp_path, monkeypatch):
    """Convert only the selected instrument and day from four tiny ZIPs.

    ETH-BTC and BNB-BTC each have two trades on June 16 and June 17, with
    prices 10 and 20. A HOCON job selects ETH-BTC for [June 16, June 17), and
    references input, output, and instrument-file paths relative to itself.
    Real ZIP reading, normalization, and Parquet writing must produce only
    the two June-16 ETH rows. A sequential pool substitute also verifies that
    the explicit worker count overrides the file, without spawning processes.
    """
    raw = tmp_path / "archives"
    for pair in ("ETH-BTC", "BNB-BTC"):
        for day in (date(2021, 6, 16), date(2021, 6, 17)):
            _write_archive(raw, pair, day)
    (tmp_path / "pairs.json").write_text(json.dumps({"currency_pairs": ["ETH-BTC"]}))
    config = tmp_path / "hive.conf"
    config.write_text("""
        start = "2021-06-16"
        end = "2021-06-17"
        raw-data-dir = "archives"
        output-dir = "hive"
        pairs-file = "pairs.json"
        processes = 1
    """)
    seen = {}

    def sequential_run(pipe, processes):
        seen["processes"] = processes
        seen["tasks"] = list(pipe.iterate_over_tasks())
        for day, pair in seen["tasks"]:
            pipe.unzip_and_save_to_hive(pair, day)

    monkeypatch.setattr(module.BinanceSpotTrades2Hive, "run_multiprocessing", sequential_run)
    module.run_main(["--config", str(config), "--processes", "2"])

    assert seen == {"processes": 2, "tasks": [(date(2021, 6, 16), CurrencyPair.from_string("ETH-BTC"))]}
    frame = pl.read_parquet(tmp_path / "hive", hive_partitioning=True).sort("price")
    assert frame["symbol"].to_list() == ["ETH-BTC", "ETH-BTC"]
    assert frame["date"].to_list() == [date(2021, 6, 16)] * 2
    assert frame["price"].to_list() == [10.0, 20.0]
    assert not (tmp_path / "hive" / "date=2021-06-17").exists()


@pytest.mark.parametrize("pair_names", [None, ["ETH-BTC"]])
def test_optional_pair_filter_preserves_default_all_instrument_behavior(tmp_path, pair_names):
    """Omitting an instrument selection retains the converter's previous scope.

    Two local instruments each have one in-range archive. With no filter,
    task discovery must yield both; with only ETH-BTC requested it must yield
    exactly ETH-BTC. This catches default changes that would silently exclude
    downloaded instruments while adding the new optional selection argument.
    """
    day = date(2021, 6, 16)
    raw = tmp_path / "raw"
    for name in ("ETH-BTC", "BNB-BTC"):
        _write_archive(raw, name, day)
    pairs = None if pair_names is None else [CurrencyPair.from_string(name) for name in pair_names]
    pipe = module.BinanceSpotTrades2Hive(Bounds.for_day(day), raw, tmp_path / "hive", currency_pairs=pairs)
    tasks = {(day, pair.name) for day, pair in pipe.iterate_over_tasks()}
    expected_names = ["ETH-BTC", "BNB-BTC"] if pair_names is None else pair_names
    assert tasks == {(day, name) for name in expected_names}


@pytest.mark.parametrize(
    "config_values",
    [
        {"start": "2021-06-16", "end": "2021-06-16"},
        {"start": "2021-06-17", "end": "2021-06-16"},
        {"processes": 0},
        {"processes": -1},
        {"instruments": ["ETHBTC"]},
    ],
)
def test_invalid_conversion_job_fails_before_pipeline_creation(tmp_path, monkeypatch, config_values):
    """Refuse invalid configuration before scanning archives or writing data.

    Dummy JSON jobs cover empty/inverted intervals, zero/negative worker
    counts, and a malformed instrument name. Each must exit through argparse
    with code 2 before the pipeline constructor runs. This makes errors in
    mounted config files visible immediately and prevents partial conversion.
    """
    config = tmp_path / "hive.json"
    config.write_text(json.dumps(config_values))
    monkeypatch.setattr(module, "BinanceSpotTrades2Hive", lambda **_: pytest.fail("Unexpected conversion"))
    with pytest.raises(SystemExit) as result:
        module.run_main(["--config", str(config)])
    assert result.value.code == 2
