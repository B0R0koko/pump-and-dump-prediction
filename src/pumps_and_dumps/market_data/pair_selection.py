"""Validate user-supplied instrument selections before starting a data job."""

import json
import re
from pathlib import Path

from pumps_and_dumps.core.currency_pair import CurrencyPair

_PAIR_NAME = re.compile(r"[A-Z0-9]+-[A-Z0-9]+\Z")


def parse_pair_names(names: object) -> list[CurrencyPair]:
    """Return sorted unique BASE-QUOTE pairs, rejecting empty or malformed lists."""
    if not isinstance(names, list) or not names:
        raise ValueError("Instrument selection must be a nonempty list of BASE-QUOTE strings")
    normalized = []
    for name in names:
        if not isinstance(name, str) or not _PAIR_NAME.fullmatch(name.strip().upper()):
            raise ValueError(f"Invalid instrument {name!r}; expected BASE-QUOTE, for example ETH-BTC")
        normalized.append(name.strip().upper())
    return [CurrencyPair.from_string(name) for name in sorted(set(normalized))]


def read_pairs_file(path: Path) -> list[CurrencyPair]:
    """Read a JSON array or an object containing a ``currency_pairs`` array."""
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "currency_pairs" not in payload:
            raise ValueError("Pairs file must contain a currency_pairs list")
        payload = payload["currency_pairs"]
    return parse_pair_names(payload)
