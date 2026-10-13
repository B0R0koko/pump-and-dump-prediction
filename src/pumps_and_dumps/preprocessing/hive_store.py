"""Partition storage for normalized trade batches."""

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Protocol

import pandas as pd

from pumps_and_dumps.core.currency_pair import CurrencyPair


class TradeBatchStore(Protocol):
    def replace_partition(
        self,
        currency_pair: CurrencyPair,
        day: date,
        batches: Iterable[pd.DataFrame],
    ) -> None: ...

    def reset(self, currency_pair: CurrencyPair, day: date) -> None: ...

    def write(self, frame: pd.DataFrame) -> None: ...


@dataclass(frozen=True)
class HiveTradeStore:
    root: Path

    def partition_dir(self, currency_pair: CurrencyPair, day: date) -> Path:
        return self.root / f"date={day}" / f"symbol={currency_pair.name}"

    def replace_partition(
        self,
        currency_pair: CurrencyPair,
        day: date,
        batches: Iterable[pd.DataFrame],
    ) -> None:
        """Keep the existing partition until every replacement batch is written.

        Staging and backups sit outside the Hive root so dataset scans cannot see
        them. The parent and root must share a filesystem. Publication assumes no
        concurrent readers or converters for the same partition: directory renames
        leave a brief gap. Handled failures restore the previous partition; a hard
        kill during publication can require restoring the retained backup manually.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.stat().st_dev != self.root.parent.stat().st_dev:
            raise ValueError("Hive root and its parent must share a filesystem")
        with TemporaryDirectory(dir=self.root.parent, prefix=".hive-staging-") as name:
            staging = HiveTradeStore(Path(name))
            for frame in batches:
                staging.write(frame)
            staged_partition = staging.partition_dir(currency_pair, day)
            if not staged_partition.is_dir():
                raise ValueError(f"No trade rows to write for {currency_pair.name} {day}")
            _publish_partition(
                staged_partition,
                self.partition_dir(currency_pair, day),
                backup_root=self.root.parent,
            )

    def reset(self, currency_pair: CurrencyPair, day: date) -> None:
        """Legacy direct reset; archive conversion uses replace_partition instead."""
        path = self.partition_dir(currency_pair, day)
        if path.exists():
            shutil.rmtree(path)

    def write(self, frame: pd.DataFrame) -> None:
        frame.to_parquet(
            self.root,
            engine="pyarrow",
            compression="gzip",
            partition_cols=["date", "symbol"],
            existing_data_behavior="overwrite_or_ignore",
        )


def _publish_partition(staged: Path, destination: Path, *, backup_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        staged.rename(destination)
        return

    # Keep backups outside TemporaryDirectory: even failed restoration must never
    # clean up the only remaining copy of the original data.
    backup_dir = Path(mkdtemp(dir=backup_root, prefix=".hive-backup-"))
    backup = backup_dir / destination.parent.name / destination.name
    backup.parent.mkdir()
    try:
        destination.rename(backup)
    except BaseException:
        if backup.exists():
            _restore_partition(backup, destination)
        shutil.rmtree(backup_dir)
        raise

    try:
        staged.rename(destination)
    except BaseException:
        _restore_partition(backup, destination)
        shutil.rmtree(backup_dir)
        raise
    shutil.rmtree(backup_dir)


def _restore_partition(backup: Path, destination: Path) -> None:
    try:
        backup.rename(destination)
    except OSError as error:
        raise OSError(
            f"Could not restore Hive partition {destination}. " f"The original data is preserved at {backup}."
        ) from error
