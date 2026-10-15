import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import polars as pl

from pumps_and_dumps.core.columns import TRADE_TIME, PRICE
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.pump_event import PumpEvent


def aggregate_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades by TRADE_TIME"""
    df_trades: pl.DataFrame = df_ticks.group_by(TRADE_TIME, maintain_order=True).agg(
        price_first=pl.col(PRICE).first(),  # if someone placed a trade with price impact, then price_first
        price_last=pl.col(PRICE).last(),  # and price_last will differ
        # Amount spent in quote asset for the trade
        quote_abs=pl.col("quote_abs").sum(),
        quote_sign=pl.col("quote_sign").sum(),
        quantity_sign=pl.col("quantity_sign").sum(),
        # Amount of base asset transacted
        quantity_abs=pl.col("quantity").sum(),
        num_ticks=pl.col("price").count(),  # number of ticks for each trade
    )
    # A zero-net-side timestamp is neutral, not a buy. Counting it as long
    # biases the share-of-long-trades feature upward in mixed-fill groups.
    df_trades = df_trades.with_columns(is_long=pl.col("quantity_sign") > 0)
    return df_trades


def parse_utc_time(value: str) -> datetime:
    """Parse an ISO date/time as naive UTC, matching the trade timestamp convention.

    Legacy ``YYYY-MM-DD HH:MM:SS`` pump manifests remain valid. An absent
    timezone means UTC; an explicit offset is converted to UTC before dropping
    its timezone, so comparisons with the Hive dataset remain unambiguous.
    """
    if not isinstance(value, str):
        raise ValueError("timestamps must be quoted ISO date/time strings")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid ISO date/time: {value!r}") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def parse_pumps(events: object) -> List[PumpEvent]:
    """Validate manifest records and preserve their order for feature generation.

    Each record has exactly ``symbol``, ``exchange``, and ``time``. Reject empty
    manifests and duplicate output identities before writing any features; two
    announcements mapping to one filename would otherwise overwrite each other.
    """
    if not isinstance(events, list) or not events:
        raise ValueError("pumps must be a non-empty list of event objects")
    pump_events: List[PumpEvent] = []
    identities: set[str] = set()
    for index, event in enumerate(events):
        prefix = f"pumps[{index}]"
        if not isinstance(event, Mapping) or set(event) != {"symbol", "exchange", "time"}:
            raise ValueError(f"{prefix} must contain exactly symbol, exchange, and time")
        symbol = event["symbol"]
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9]+-[A-Za-z0-9]+", symbol):
            raise ValueError(f"{prefix}.symbol must be an instrument such as ETH-BTC")
        try:
            exchange = Exchange.parse_from_lower(event["exchange"])
        except (AttributeError, KeyError) as error:
            raise ValueError(f"{prefix}.exchange must be BINANCE_SPOT") from error
        try:
            pump_time = parse_utc_time(event["time"])
        except ValueError as error:
            raise ValueError(f"{prefix}.time: {error}") from error
        if pump_time.microsecond or re.search(r"[.,]\d+", event["time"]):
            raise ValueError(
                f"{prefix}.time must use whole seconds; pump identifiers do not preserve fractional seconds"
            )
        pump = PumpEvent(CurrencyPair.from_string(symbol.upper()), pump_time, exchange)
        if pump.as_pump_hash() in identities:
            raise ValueError(f"{prefix} duplicates an event's output filename: {pump}")
        identities.add(pump.as_pump_hash())
        pump_events.append(pump)
    return pump_events


def load_pumps(path: Path) -> List[PumpEvent]:
    """Load a JSON event list using the same validation as inline HOCON pumps."""
    with open(path) as file:
        return parse_pumps(json.load(file))
