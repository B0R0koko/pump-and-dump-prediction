"""Discover and stream local Binance archives without knowing the output format."""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from pumps_and_dumps.core.columns import (
    BINANCE_TRADE_COLS,
    IS_BUYER_MAKER,
    PRICE,
    QUANTITY,
    TRADE_TIME,
)
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.time_utils import Bounds

_ARCHIVE_NAME_RE = re.compile(r"@(\d{4}-\d{2}-\d{2})\.zip$")
_USE_COLS = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]
_log = logging.getLogger(__name__)


def filter_by_bounds(bounds: Bounds, file_names: list[str]) -> list[date]:
    """Ignore unrelated files; return archive dates inside the requested bounds."""
    dates = []
    for name in file_names:
        match = _ARCHIVE_NAME_RE.search(name)
        if match is None:
            _log.warning("Skipping raw file with unexpected name: %s", name)
            continue
        day = date.fromisoformat(match.group(1))
        if bounds.contain_days(day=day):
            dates.append(day)
    return dates


class ArchiveSource(Protocol):
    def tasks(self, bounds: Bounds) -> Iterator[tuple[date, CurrencyPair]]: ...

    def batches(
        self, currency_pair: CurrencyPair, day: date
    ) -> Iterator[pd.DataFrame]: ...


@dataclass(frozen=True)
class BinanceTradeArchives:
    root: Path
    batch_size: int = 1_000_000

    def tasks(self, bounds: Bounds) -> Iterator[tuple[date, CurrencyPair]]:
        for directory in self.root.iterdir():
            pair = _directory_pair(directory)
            if pair is None:
                continue
            dates = filter_by_bounds(
                bounds, [path.name for path in directory.iterdir()]
            )
            for day in dates:
                yield day, pair

    def batches(self, currency_pair: CurrencyPair, day: date) -> Iterator[pd.DataFrame]:
        path = self.root / currency_pair.name / f"trades@{day}.zip"
        with pd.read_csv(
            path,
            chunksize=self.batch_size,
            header=None,
            names=BINANCE_TRADE_COLS,
            usecols=_USE_COLS,
        ) as reader:
            yield from reader


def _directory_pair(directory: Path) -> CurrencyPair | None:
    if not directory.is_dir():
        _log.warning("Skipping non-directory entry in raw data dir: %s", directory.name)
        return None
    try:
        return CurrencyPair.from_string(symbol=directory.name)
    except ValueError:
        _log.warning(
            "Skipping directory that does not look like a CurrencyPair: %s",
            directory.name,
        )
        return None
