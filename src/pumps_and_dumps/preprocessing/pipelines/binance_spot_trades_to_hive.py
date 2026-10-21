"""Coordinate archive reading, trade normalization, and Hive storage."""

import argparse
from collections.abc import Iterator
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from pumps_and_dumps.core.cli_config import parse_config_args
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.paths import BINANCE_SPOT_RAW_TRADES, BINANCE_SPOT_HIVE_TRADES
from pumps_and_dumps.core.progress import (
    ProgressReporter,
    progress_enabled,
    worker_progress,
)
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.core.utils import available_cpu_count
from pumps_and_dumps.market_data.pair_selection import parse_pair_names, read_pairs_file
from pumps_and_dumps.preprocessing.archive_source import (
    ArchiveSource,
    BinanceTradeArchives,
    filter_by_bounds,
)
from pumps_and_dumps.preprocessing.hive_store import HiveTradeStore, TradeBatchStore
from pumps_and_dumps.preprocessing.trades import (
    TradeTimeUnit,
    detect_trade_time_unit as _detect_trade_time_unit,
    normalize_trade_batch,
)


def _initialize_conversion_worker() -> None:
    """The process pool supplies parallelism; avoid another CPU pool per worker."""
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


class BinanceSpotTrades2Hive:
    def __init__(
        self,
        bounds: Bounds,
        raw_data_dir: Path,
        output_dir: Path,
        currency_pairs: list[CurrencyPair] | None = None,
        archive_source: ArchiveSource | None = None,
        batch_store: TradeBatchStore | None = None,
    ):
        self.bounds = bounds
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.currency_pairs = None if currency_pairs is None else frozenset(pair.name for pair in currency_pairs)
        self._archive_source = BinanceTradeArchives(raw_data_dir) if archive_source is None else archive_source
        self._batch_store = HiveTradeStore(output_dir) if batch_store is None else batch_store

    @staticmethod
    def preprocess_batched_data(df_batch: pd.DataFrame, currency_pair: CurrencyPair, day: date) -> pd.DataFrame:
        return normalize_trade_batch(df_batch, currency_pair, day)

    def save_batched_data_to_hive(self, df_batch: pd.DataFrame) -> None:
        self._batch_store.write(df_batch)

    def _partition_dir(self, currency_pair: CurrencyPair, day: date) -> Path:
        return HiveTradeStore(self.output_dir).partition_dir(currency_pair, day)

    def _reset_partition(self, currency_pair: CurrencyPair, day: date) -> None:
        self._batch_store.reset(currency_pair, day)

    def unzip_and_save_to_hive(self, currency_pair: CurrencyPair, day: date) -> None:
        with worker_progress(unit="rows", detail=f"{currency_pair.name} {day}") as progress:
            self._batch_store.replace_partition(
                currency_pair,
                day,
                self._normalized_batches(currency_pair, day, progress),
            )

    def _normalized_batches(
        self, currency_pair: CurrencyPair, day: date, progress: ProgressReporter
    ) -> Iterator[pd.DataFrame]:
        for batch in self._archive_source.batches(currency_pair, day):
            normalized = self.preprocess_batched_data(df_batch=batch, currency_pair=currency_pair, day=day)
            yield normalized
            progress.update(advance=len(normalized))

    def _convert_task(self, task: tuple[date, CurrencyPair]) -> str:
        day, currency_pair = task
        self.unzip_and_save_to_hive(currency_pair=currency_pair, day=day)
        return f"{currency_pair.name} {day}"

    def iterate_over_tasks(self) -> Iterator[tuple[date, CurrencyPair]]:
        for day, pair in self._archive_source.tasks(self.bounds):
            if self.currency_pairs is None or pair.name in self.currency_pairs:
                yield day, pair

    def run_multiprocessing(self, processes: int | None = None) -> None:
        processes = available_cpu_count() if processes is None else processes
        if processes < 1:
            raise ValueError("processes must be positive")
        with ProgressReporter(unit="archives", detail="Scanning downloaded archives") as progress:
            tasks = []
            for task in self.iterate_over_tasks():
                tasks.append(task)
                progress.update(detail=f"Scanning downloaded archives: {len(tasks):,} found")
            progress.update(total=len(tasks), detail="Converting archives", force=True)
            with Pool(processes=processes, initializer=_initialize_conversion_worker) as pool:
                results = pool.imap_unordered(self._convert_task, tasks)
                for detail in tqdm(
                    results,
                    total=len(tasks),
                    desc="Saving zipped csv files to HiveDataset",
                    disable=progress_enabled(),
                ):
                    progress.update(advance=1, detail=detail)


def run_main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Convert daily trade ZIPs into date/symbol Hive Parquet partitions.")
    parser.add_argument("--start-date", "--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument(
        "--end-date",
        "--end",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
        help="Exclusive end date",
    )
    parser.add_argument("--raw-data-dir", type=Path, default=BINANCE_SPOT_RAW_TRADES)
    parser.add_argument("--output-dir", type=Path, default=BINANCE_SPOT_HIVE_TRADES)
    pairs_source = parser.add_mutually_exclusive_group()
    pairs_source.add_argument(
        "--pairs",
        "--instruments",
        nargs="+",
        help="Convert only these BASE-QUOTE pairs (default: all downloaded pairs).",
    )
    pairs_source.add_argument("--pairs-file", type=Path, help="JSON array or object containing a currency_pairs list.")
    parser.add_argument(
        "--processes",
        type=int,
        default=available_cpu_count(),
        help="Worker processes (default: all available CPUs).",
    )
    args = parse_config_args(parser, argv, path_keys=("raw_data_dir", "output_dir", "pairs_file"))
    if args.processes < 1:
        parser.error("--processes must be positive")
    if args.start_date >= args.end_date:
        parser.error("--start must be before the exclusive --end")
    try:
        pairs = (
            read_pairs_file(args.pairs_file)
            if args.pairs_file
            else parse_pair_names(args.pairs) if args.pairs is not None else None
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"Cannot read conversion instruments: {error}")
    bounds: Bounds = Bounds.for_days(args.start_date, args.end_date)
    raw_dir = args.raw_data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        parser.error(f"Raw trade directory does not exist: {raw_dir}")
    print(f"Converting trades from {args.start_date} to {args.end_date} (exclusive)")
    print(f"Input: {raw_dir}\nOutput: {output_dir}", flush=True)
    pipe = BinanceSpotTrades2Hive(
        bounds=bounds,
        raw_data_dir=raw_dir,
        output_dir=output_dir,
        currency_pairs=pairs,
    )
    pipe.run_multiprocessing(processes=args.processes)


if __name__ == "__main__":
    run_main()
