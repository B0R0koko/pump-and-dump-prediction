"""Stage catalog, execution order, and prerequisites for partial installations."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import os
from typing import Callable


def available_cpu_count() -> int:
    """Support the lightweight host launcher on Python 3.10 and newer."""
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0)) or 1
        except OSError:
            pass
    return os.cpu_count() or 1


@dataclass(frozen=True)
class StageSettings:
    root: Path
    start_date: date
    end_date: date
    hive_workers: int = field(default_factory=available_cpu_count)
    feature_workers: int = field(default_factory=available_cpu_count)
    skip_existing_features: bool = False


@dataclass(frozen=True)
class StageInput:
    producer: str
    directory: str
    pattern: str
    missing_message: str


@dataclass(frozen=True)
class StageOption:
    key: str
    title: str
    description: str
    module: str
    options: Callable[[StageSettings], list[str]]
    manifest: str | None = None
    prerequisite: StageInput | None = None


def _archive_dates(settings: StageSettings) -> list[str]:
    return [
        "--start-date",
        str(settings.start_date),
        "--end-date",
        str(settings.end_date),
    ]


def _trade_options(settings: StageSettings) -> list[str]:
    pairs = settings.root / "resources" / "btc_currencies.json"
    return [*_archive_dates(settings), "--pairs-file", str(pairs), "--skip-existing"]


def _candle_options(settings: StageSettings) -> list[str]:
    return [
        *_archive_dates(settings),
        "--pairs",
        "BTC-USDT",
        "--interval",
        "1m",
        "--skip-existing",
    ]


def _parquet_options(settings: StageSettings) -> list[str]:
    return [*_archive_dates(settings), "--processes", str(settings.hive_workers)]


def _feature_options(settings: StageSettings) -> list[str]:
    options = ["--cpu-count", str(settings.feature_workers)]
    if settings.skip_existing_features:
        options.append("--skip-existing")
    return options


STAGES = (
    StageOption(
        "trades",
        "Trade archives",
        "Verify BTC-pair ZIPs; download missing or damaged files.",
        "pumps_and_dumps.market_data.parsers.binance.spot_trades",
        _trade_options,
        manifest="btc_currencies.json",
    ),
    StageOption(
        "candles",
        "BTC/USDT candles",
        "Verify 1-minute candle ZIPs; download missing or damaged files.",
        "pumps_and_dumps.market_data.parsers.binance.spot_klines",
        _candle_options,
    ),
    StageOption(
        "parquet",
        "Hive Parquet",
        "Convert trade ZIPs into date/symbol partitions.",
        "pumps_and_dumps.preprocessing.pipelines.binance_spot_trades_to_hive",
        _parquet_options,
        prerequisite=StageInput(
            "trades",
            "raw/binance/spot/trades",
            "*/*.zip",
            "No trade ZIPs found here. Include Trade archives or choose a directory containing raw trades.",
        ),
    ),
    StageOption(
        "features",
        "Event features",
        "Compute features from the existing Hive trades.",
        "pumps_and_dumps.features.writer",
        _feature_options,
        manifest="pumps.json",
        prerequisite=StageInput(
            "parquet",
            "transformed/binance/spot/trades",
            "date=*/symbol=*/*.parquet",
            "No Hive Parquet files found here. Include Hive Parquet or choose an existing dataset.",
        ),
    ),
)
STAGE_KEYS = tuple(stage.key for stage in STAGES)


def ordered_stages(selected) -> list[str]:
    """Run each selected stage once, in pipeline order."""
    return [key for key in STAGE_KEYS if key in selected]


def validate_inputs(selected, data_dir: Path, *, skip_stages=()) -> str | None:
    """Require existing inputs only when the stage producing them is omitted."""
    if not selected:
        return "Select at least one stage to continue."
    if data_dir.exists() and not data_dir.is_dir():
        return f"The data location is a file: {data_dir}"
    for stage in STAGES:
        required = stage.prerequisite
        if stage.key not in selected or stage.key in skip_stages or required is None:
            continue
        if required.producer in selected:
            continue
        directory = data_dir / required.directory
        if not any(path.is_file() for path in directory.glob(required.pattern)):
            return required.missing_message
    return None
