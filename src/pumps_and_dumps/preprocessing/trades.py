"""Binance trade batch normalization, independent of archive and Parquet storage."""

from datetime import date
from typing import Literal

import pandas as pd

from pumps_and_dumps.core.columns import TRADE_TIME
from pumps_and_dumps.core.currency_pair import CurrencyPair

TradeTimeUnit = Literal["ms", "us"]
_MS_MIN, _MS_MAX = 10**12, 10**13
_US_MIN, _US_MAX = 10**15, 10**16


def detect_trade_time_unit(sample: int, day: date) -> TradeTimeUnit:
    """Validate timestamp magnitude instead of trusting an archive's filename date."""
    if _MS_MIN <= sample < _MS_MAX:
        return "ms"
    if _US_MIN <= sample < _US_MAX:
        return "us"
    raise ValueError(
        f"Unrecognised Binance trade_time magnitude {sample} for day {day}; "
        f"expected milliseconds ({_MS_MIN}..{_MS_MAX}) or microseconds ({_US_MIN}..{_US_MAX})."
    )


def normalize_trade_batch(
    df_batch: pd.DataFrame, currency_pair: CurrencyPair, day: date
) -> pd.DataFrame:
    """Decode ms/us timestamps and attach Hive partition keys without doing I/O."""
    unit: TradeTimeUnit = "ms"
    if not df_batch.empty:
        unit = detect_trade_time_unit(sample=int(df_batch[TRADE_TIME].iat[0]), day=day)
    df_batch[TRADE_TIME] = pd.to_datetime(df_batch[TRADE_TIME], unit=unit)
    df_batch["date"] = day
    df_batch["symbol"] = currency_pair.name
    return df_batch
