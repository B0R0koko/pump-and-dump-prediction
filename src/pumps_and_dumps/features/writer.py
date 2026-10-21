"""Coordinate event feature generation using independent data and calculation components."""

import argparse
import logging
import os
from bisect import bisect_left
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, List

import polars as pl
from tqdm import tqdm

from pumps_and_dumps.core.cli_config import parse_config_args
from pumps_and_dumps.core.columns import IS_BUYER_MAKER, TRADE_TIME
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.feature_type import FeatureType
from pumps_and_dumps.core.paths import FEATURE_DIR, get_root_dir
from pumps_and_dumps.core.progress import (
    PROGRESS_FD_ENV,
    ProgressReporter,
    progress_enabled,
    worker_progress,
)
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.core.time_utils import Bounds, NamedTimeDelta
from pumps_and_dumps.core.utils import available_cpu_count, configure_logging
from pumps_and_dumps.features.audit import AssetInputAudit, FeatureAuditLog
from pumps_and_dumps.features.calculation import (
    compute_window_features,
    compute_targets,
    has_recent_activity,
    prepare_trades,
)
from pumps_and_dumps.features.provenance import (
    dependency,
    feature_provenance,
    timestamp_ns,
    utc_timestamp,
)
from pumps_and_dumps.features.schema import FEATURE_SCHEMA_VERSION, REGRESSOR_OFFSETS
from pumps_and_dumps.features.source import HiveTradeSource, TradeSource
from pumps_and_dumps.features.storage import is_current_feature_file, write_feature_file
from pumps_and_dumps.features.utils import load_pumps, parse_pumps, parse_utc_time

# Offsets to compute decay returns
DECAY_OFFSETS: List[NamedTimeDelta] = [
    NamedTimeDelta.ONE_MINUTE,
    NamedTimeDelta.TWO_MINUTES,
    NamedTimeDelta.THREE_MINUTES,
    NamedTimeDelta.FOUR_MINUTES,
    NamedTimeDelta.FIVE_MINUTES,
]

# Decision lag between the last trade a feature is allowed to see and the pump
# announcement time. The portfolio submits its entry at ``T - DECISION_LAG``
# and fills on the first subsequent pre-announcement trade (see
# ``TOPKPortfolio.buy_before``); features must reflect only information
# available when that order is submitted. The interval
# ``[T - DECISION_LAG, T]`` must never contribute to any feature. Changing this
# constant requires regenerating all feature parquets from raw trades.
DECISION_LAG: timedelta = timedelta(minutes=15)

# Normalizer statistics are estimated from a fixed, past-only history whose
# right edge is the decision time. Candidate assets must also have traded
# recently enough that the backtest can form a contemporaneous entry price.
NORMALIZER_LOOKBACK: timedelta = timedelta(days=30)
ELIGIBILITY_LOOKBACK: timedelta = timedelta(days=1)

_WORKER_WRITER: "PumpsFeatureWriter | None" = None
_WORKER_PROGRESS_CONNECTION: Connection | None = None


@contextmanager
def _worker_thread_limit() -> Iterator[None]:
    """Use one Polars thread per process unless an explicit limit was supplied."""
    previous = os.environ.get("POLARS_MAX_THREADS")
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("POLARS_MAX_THREADS", None)


@contextmanager
def _progress_connection() -> Iterator[Connection | None]:
    """Keep a transferable duplicate of the installer's pipe alive for the pool."""
    progress_fd = os.environ.get(PROGRESS_FD_ENV)
    if progress_fd is None:
        yield None
        return
    with Connection(os.dup(int(progress_fd)), readable=False, writable=True) as connection:
        yield connection


def _initialize_feature_worker(writer: "PumpsFeatureWriter", lock: Any, progress_connection: Connection | None) -> None:
    """Initialize each spawned worker once, including its own progress descriptor."""
    global _WORKER_WRITER, _WORKER_PROGRESS_CONNECTION
    _WORKER_WRITER = writer
    _WORKER_PROGRESS_CONNECTION = progress_connection
    tqdm.set_lock(lock)
    if progress_connection is None:
        os.environ.pop(PROGRESS_FD_ENV, None)
        return
    os.environ[PROGRESS_FD_ENV] = str(progress_connection.fileno())


def _write_worker_task(task: tuple[int, PumpEvent], skip_existing: bool) -> str:
    if _WORKER_WRITER is None:
        raise RuntimeError("Feature worker has not been initialized")
    return _WORKER_WRITER._write_task(task, skip_existing=skip_existing)


def get_currency_pairs(bounds: Bounds) -> list[CurrencyPair]:
    """Discover the BTC-pair universe from the default Hive dataset."""
    return HiveTradeSource(Exchange.BINANCE_SPOT.get_hive_location()).currency_pairs(bounds)


class PumpsFeatureWriter:
    def __init__(
        self,
        pump_events: list[PumpEvent],
        data_source: TradeSource | None = None,
        feature_dir: Path | None = None,
        audit_log_path: Path | None = None,
    ):
        self._pump_events = pump_events
        self._pump_times_by_currency: dict[str, list[datetime]] = {}
        for event in sorted(pump_events, key=lambda event: event.time):
            self._pump_times_by_currency.setdefault(event.currency_pair.name, []).append(event.time)
        self._data_source = (
            data_source if data_source is not None else HiveTradeSource(Exchange.BINANCE_SPOT.get_hive_location())
        )
        self._feature_dir = FEATURE_DIR if feature_dir is None else feature_dir
        self._audit_log = FeatureAuditLog.for_run(self._feature_dir, audit_log_path)
        self._audit_started = False

    @property
    def audit_log_path(self) -> Path:
        """Destination printed at startup; created on the first computation/run."""
        return self._audit_log.path

    def _start_audit(self, event_count: int | None = None) -> None:
        # Compatibility with lightweight test doubles that bypass __init__.
        audit = getattr(self, "_audit_log", None)
        if audit is None or self._audit_started:
            return
        audit.append(
            [
                {
                    "record_type": "run_start",
                    "audit_schema_version": 2,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "pump_count": len(self._pump_events) if event_count is None else event_count,
                    "decision_lag_seconds": DECISION_LAG.total_seconds(),
                    "normalizer_lookback_seconds": NORMALIZER_LOOKBACK.total_seconds(),
                    "timezone": "UTC",
                    "interval_convention": "actual input bounds inclusive; non_overlapping requires feature_max_time < target_min_time",
                }
            ]
        )
        self._audit_started = True
        logging.getLogger(__name__).info("Feature timestamp audit log: %s", audit.path)

    def _append_audit(self, records: list[dict[str, Any]]) -> None:
        audit = getattr(self, "_audit_log", None)
        if audit is not None:
            self._start_audit()
            audit.append(records)

    @staticmethod
    def _event_context(pump_event: PumpEvent) -> dict[str, Any]:
        return {"pump_hash": pump_event.as_pump_hash(), "pump_time": utc_timestamp(pump_event.time)}

    def load_data_for_currency_pair(self, bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame:
        """Load only requested partitions, with an exclusive trade-time upper bound."""
        return self._data_source.load(bounds, currency_pair)

    @staticmethod
    def side_expr() -> pl.Expr:
        return 1 - 2 * pl.col(IS_BUYER_MAKER)

    def preprocess_data_for_currency(self, df: pl.DataFrame) -> pl.DataFrame:
        return prepare_trades(df, side=self.side_expr())

    @staticmethod
    def has_pre_decision_activity(df_ticks: pl.DataFrame, rb: datetime) -> bool:
        return has_recent_activity(df_ticks, cutoff=rb, lookback=ELIGIBILITY_LOOKBACK)

    def _num_prev_pumps(self, currency_pair: CurrencyPair, pump_event: PumpEvent) -> int:
        """Only announcements strictly before the decision cutoff are known."""
        times = self._pump_times_by_currency.get(currency_pair.name, [])
        return bisect_left(times, pump_event.time - DECISION_LAG)

    def compute_features(
        self,
        df: pl.DataFrame,
        currency_pair: CurrencyPair,
        pump_event: PumpEvent,
    ) -> dict[str, Any]:
        """Calculate separate predictor/target inputs and log their bounds once per asset."""
        cutoff = pump_event.time - DECISION_LAG
        self._start_audit()
        audit = AssetInputAudit() if getattr(self, "_audit_log", None) is not None else None
        callback = audit.observe if audit is not None else None
        # The shared load contains both history and outcome rows. Only the
        # past-only slice reaches predictor calculations and their normalizers.
        predictor_trades = df.with_columns(pl.col(TRADE_TIME).cast(pl.Datetime("ns"))).filter(
            pl.col(TRADE_TIME) < cutoff
        )
        features = compute_window_features(predictor_trades, cutoff, REGRESSOR_OFFSETS, audit=callback)
        count = self._num_prev_pumps(currency_pair, pump_event)
        features[FeatureType.NUM_PREV_PUMP.lower()] = count
        if callback is not None:
            known_pumps = self._pump_times_by_currency.get(currency_pair.name, [])[:count]
            start = timestamp_ns(known_pumps[0]) if count else None
            end = timestamp_ns(known_pumps[-1]) if count else None
            callback(
                feature_provenance(
                    FeatureType.NUM_PREV_PUMP.lower(),
                    "predictor",
                    None,
                    cutoff,
                    [dependency("previous_pump_announcements", start, end, count)],
                )
            )
        features.update(compute_targets(df, pump_event.time, DECAY_OFFSETS, audit=callback))
        if audit is not None:
            self._append_audit(
                [
                    {
                        "record_type": "asset_inputs",
                        **self._event_context(pump_event),
                        "asset": currency_pair.name,
                        "decision_cutoff": utc_timestamp(cutoff),
                        **audit.summary(cutoff),
                    }
                ]
            )
        return features

    def _features_for_pair(
        self, currency_pair: CurrencyPair, pump_event: PumpEvent, bounds: Bounds
    ) -> dict[str, Any] | None:
        ticks = self.load_data_for_currency_pair(bounds=bounds, currency_pair=currency_pair)
        cutoff = pump_event.time - DECISION_LAG
        if not self.has_pre_decision_activity(df_ticks=ticks, rb=cutoff):
            return None
        trades = self.preprocess_data_for_currency(df=ticks)
        features = self.compute_features(
            df=trades,
            currency_pair=currency_pair,
            pump_event=pump_event,
        )
        return features | {"currency_pair": currency_pair.name}

    def create_cross_section(self, pump_event: PumpEvent, position: int) -> pl.DataFrame | None:
        self._start_audit()
        cutoff = pump_event.time - DECISION_LAG
        load_bounds = Bounds(
            start_inclusive=cutoff - NORMALIZER_LOOKBACK,
            end_exclusive=pump_event.time + max(window.get_td() for window in DECAY_OFFSETS),
        )
        # Discovery is past-only; exact timestamp eligibility is checked after
        # loading, since a boundary day's partition can include later trades.
        discovery_bounds = Bounds(start_inclusive=cutoff - ELIGIBILITY_LOOKBACK, end_exclusive=cutoff)
        with worker_progress(unit="pairs", detail=f"{pump_event}: discovering candidates") as progress:
            pairs = self._data_source.currency_pairs(discovery_bounds)
            progress.update(total=len(pairs), force=True)
            rows = self._collect_cross_section(pairs, pump_event, load_bounds, position, progress)
            if not any(row["currency_pair"] == pump_event.currency_pair.name for row in rows):
                progress.update(detail=f"{pump_event}: no pre-decision target data", force=True)
                return None
            return pl.DataFrame(rows, infer_schema_length=None)

    def _collect_cross_section(
        self,
        pairs: list[CurrencyPair],
        pump_event: PumpEvent,
        bounds: Bounds,
        position: int,
        progress: ProgressReporter,
    ) -> list[dict[str, Any]]:
        rows = []
        with tqdm(
            total=len(pairs),
            desc="Iterating over currency_pairs",
            position=2 + position,
            leave=False,
            disable=progress_enabled(),
        ) as pbar:
            for pair in pairs:
                progress.update(detail=f"{pump_event}: {pair.name}", force=True)
                features = self._features_for_pair(pair, pump_event, bounds)
                if features is not None:
                    rows.append(features)
                pbar.update(1)
                progress.update(advance=1)
        return rows

    def _feature_path(self, pump_event: PumpEvent) -> Path:
        root = getattr(self, "_feature_dir", FEATURE_DIR)
        return root / "pumps" / f"{pump_event}.parquet"

    def _write_cross_section(self, pump_event: PumpEvent, position: int = 0, skip_existing: bool = False) -> str:
        path = self._feature_path(pump_event)
        if skip_existing and is_current_feature_file(path):
            self._append_audit(
                [
                    {
                        "record_type": "event",
                        **self._event_context(pump_event),
                        "status": "reused",
                        "audit_provenance": "not_recomputed",
                        "feature_file": str(path),
                    }
                ]
            )
            return "reused"
        features = self.create_cross_section(pump_event=pump_event, position=position)
        if features is None:
            # Remove obsolete cached events that no longer satisfy causal eligibility.
            path.unlink(missing_ok=True)
            self._append_audit(
                [
                    {
                        "record_type": "event",
                        **self._event_context(pump_event),
                        "status": "unavailable",
                        "reason": "target_has_no_pre_cutoff_activity",
                        "feature_file": str(path),
                    }
                ]
            )
            return "unavailable"
        write_feature_file(features, path)
        self._append_audit(
            [
                {
                    "record_type": "event",
                    **self._event_context(pump_event),
                    "status": "written",
                    "n_assets": features.height,
                    "feature_file": str(path),
                    "audit_provenance": "computed",
                }
            ]
        )
        return "written"

    def run(self, pump_events: list[PumpEvent], skip_existing: bool = False) -> None:
        self._start_audit(event_count=len(pump_events))
        counts = dict(written=0, reused=0, unavailable=0)
        try:
            for event in tqdm(pump_events):
                counts[self._write_cross_section(pump_event=event, skip_existing=skip_existing)] += 1
        except Exception as error:
            self._append_audit([{"record_type": "run_error", "error": str(error), **counts}])
            raise
        self._append_audit([{"record_type": "run_end", **counts}])

    def _write_task(self, task: tuple[int, PumpEvent], skip_existing: bool) -> str:
        position, pump_event = task
        return self._write_cross_section(pump_event, position=position, skip_existing=skip_existing)

    def run_parallel(
        self,
        cpu_count: int | None = None,
        skip_existing: bool = False,
        pump_events: list[PumpEvent] | None = None,
    ) -> None:
        """Generate selected events while retaining the complete announcement history.

        ``pump_events`` limits output tasks only. The constructor's full history
        still supplies previous-pump features, including announcements before a
        requested CLI start date.
        """
        cpu_count = available_cpu_count() if cpu_count is None else cpu_count
        if cpu_count < 1:
            raise ValueError("cpu_count must be positive")
        selected_events = self._pump_events if pump_events is None else pump_events
        self._start_audit(event_count=len(selected_events))
        # Fresh interpreters avoid inheriting Polars threads' locked state
        # through a process fork.
        context = get_context("spawn")
        tqdm.set_lock(context.RLock())
        counts = dict(written=0, reused=0, unavailable=0)
        try:
            with (
                _worker_thread_limit(),
                _progress_connection() as connection,
                context.Pool(
                    processes=cpu_count,
                    initializer=_initialize_feature_worker,
                    initargs=(self, tqdm.get_lock(), connection),
                ) as pool,
            ):
                with ProgressReporter(
                    total=len(selected_events), unit="events", detail="Computing features"
                ) as progress:
                    results = pool.imap_unordered(
                        partial(_write_worker_task, skip_existing=skip_existing),
                        ((index % cpu_count, event) for index, event in enumerate(selected_events)),
                    )
                    for result in tqdm(
                        results,
                        total=len(selected_events),
                        desc="Overall progress",
                        disable=progress_enabled(),
                    ):
                        counts[result] += 1
                        progress.update(advance=1, **counts)
        except Exception as error:
            self._append_audit([{"record_type": "run_error", "error": str(error), **counts}])
            raise
        self._append_audit([{"record_type": "run_end", **counts}])

    @property
    def pump_times_by_currency(self) -> dict[str, list[datetime]]:
        return self._pump_times_by_currency


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate features from local Hive trades and a complete pump manifest. Dates/times use UTC."
    )
    parser.add_argument(
        "--cpu-count",
        "--processes",
        type=int,
        default=available_cpu_count(),
        help="Worker processes (default: all available CPUs).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip only files with the current feature schema",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        help=(
            "Append one JSON predictor/target input-range check per pump/asset to this .log file "
            "(default: features/logs/feature_audit_<run>.log; other suffixes become .log)."
        ),
    )
    parser.add_argument("--pumps-file", type=Path, help="JSON pump list (default: bundled resources/pumps.json).")
    parser.add_argument(
        "--hive-dir", type=Path, help="Input Hive trades directory (default: configured data directory)."
    )
    parser.add_argument(
        "--feature-dir", type=Path, help="Output feature directory (default: configured features directory)."
    )
    parser.add_argument(
        "--start",
        "--start-date",
        type=parse_utc_time,
        help="Include pump announcements at or after this UTC date/time.",
    )
    parser.add_argument(
        "--end",
        "--end-date",
        type=parse_utc_time,
        help="Include pump announcements strictly before this UTC date/time.",
    )
    return parser


def _validate_output_path(parser: argparse.ArgumentParser, name: str, path: Path | None) -> None:
    if path is None:
        return
    if name == "audit_log" and path.is_dir():
        parser.error(f"--audit-log must be a file path: {path}")
    directory = path if name == "feature_dir" else path.parent
    for parent in (directory, *directory.parents):
        if parent.exists() and not parent.is_dir():
            parser.error(f"--{name.replace('_', '-')} parent must be a directory: {parent}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.cpu_count < 1:
        parser.error("--cpu-count/--processes must be positive")
    if args.start is not None and args.end is not None and args.end <= args.start:
        parser.error("--end must be later than --start")
    if args.hive_dir is not None and not args.hive_dir.is_dir():
        parser.error(f"--hive-dir must be an existing directory: {args.hive_dir}")
    _validate_output_path(parser, "feature_dir", args.feature_dir)
    _validate_output_path(parser, "audit_log", args.audit_log)
    # An explicit CLI file replaces an inline config list; conflicting sources
    # inside one config are almost certainly a typo and remain an error.
    if {"pumps", "pumps_file"} <= args._config_keys:
        parser.error("config must specify only one of pumps and pumps-file")


def _load_pump_events(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[PumpEvent]:
    use_inline_pumps = "pumps" in args._config_keys and "pumps_file" not in args._cli_keys
    try:
        pump_events = (
            parse_pumps(args.pumps)
            if use_inline_pumps
            else load_pumps(path=args.pumps_file or get_root_dir() / "resources/pumps.json")
        )
    except (OSError, ValueError) as error:
        parser.error(f"invalid pump manifest: {error}")
    if any(event.currency_pair.term != "BTC" for event in pump_events):
        parser.error("feature generation currently supports BTC-quoted pump instruments only")
    return pump_events


def main(argv: list[str] | None = None):
    configure_logging()
    parser = _argument_parser()
    args = parse_config_args(
        parser,
        argv,
        path_keys=("pumps_file", "hive_dir", "feature_dir", "audit_log"),
        object_keys=("pumps",),
    )
    _validate_args(parser, args)
    pump_events = _load_pump_events(parser, args)
    selected_events = [
        event
        for event in pump_events
        if (args.start is None or event.time >= args.start) and (args.end is None or event.time < args.end)
    ]
    if not selected_events:
        parser.error("no pump announcements fall within the requested interval")
    writer = PumpsFeatureWriter(
        pump_events=pump_events,
        data_source=HiveTradeSource(args.hive_dir) if args.hive_dir is not None else None,
        feature_dir=args.feature_dir,
        audit_log_path=args.audit_log,
    )
    run_kwargs: dict[str, Any] = {"cpu_count": args.cpu_count, "skip_existing": args.skip_existing}
    if args.start is not None or args.end is not None:
        run_kwargs["pump_events"] = selected_events
    writer.run_parallel(**run_kwargs)


if __name__ == "__main__":
    main()
