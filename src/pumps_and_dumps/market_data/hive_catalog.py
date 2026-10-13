"""Read available currency pairs from local Hive date/symbol partitions."""

import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.time_utils import Bounds

_log = logging.getLogger(__name__)


def get_cross_section_currencies(hive_dir: Path, bounds: Bounds) -> List[CurrencyPair]:
    """Collect the union of currency pairs present under `date=YYYY-MM-DD` partitions matching `bounds`.

    Entries in the date-partition directory that do not follow the `symbol=<PAIR>` naming convention
    (pyarrow metadata files, `.DS_Store`, checksums, etc.) are skipped with a warning rather than
    raising, so a stray file cannot brick the feature pipeline.
    """
    matched_dirs: List[str] = []

    for directory in os.listdir(hive_dir):
        match: Optional[re.Match[str]] = re.search(
            string=directory, pattern=r"(\d{4}-\d{2}-\d{2})"
        )
        date_matched: Optional[str] = match.group(1) if match else None
        if date_matched is None:
            continue

        dir_date: date = datetime.strptime(date_matched, "%Y-%m-%d").date()

        if bounds.contain_days(day=dir_date):
            matched_dirs.append(directory)

    all_currency_pairs: set[CurrencyPair] = set()

    for directory in matched_dirs:
        for symbol_dir in os.listdir(hive_dir.joinpath(directory)):
            parts: List[str] = symbol_dir.split("=", 1)
            if len(parts) != 2 or parts[0] != "symbol":
                _log.warning(
                    "Skipping non-symbol entry under %s: %s", directory, symbol_dir
                )
                continue
            try:
                all_currency_pairs.add(CurrencyPair.from_string(symbol=parts[1]))
            except ValueError:
                _log.warning(
                    "Skipping entry under %s with unparseable pair: %s",
                    directory,
                    symbol_dir,
                )

    # A set is convenient while collecting the union, but exposing its hash-order
    # makes feature parquet row order (and score-tie handling) process-dependent.
    return sorted(all_currency_pairs, key=lambda currency_pair: currency_pair.name)
