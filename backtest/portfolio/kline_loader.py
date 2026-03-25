from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from core.columns import (
    BINANCE_KLINES_COLS,
    CLOSE_PRICE,
    OPEN_PRICE,
    OPEN_TIME,
    QUOTE_ASSET_VOLUME,
    TAKER_BUY_QUOTE_ASSET_VOLUME,
)
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_SPOT_RAW_KLINES
from core.time_utils import Bounds


class MinuteKlineLoader:
    """
    Load Binance 1-minute candles from daily zip files.

    The loader caches symbol/day slices because price-impact models are repeatedly
    fit on overlapping lookback windows for the same assets.
    """

    def __init__(self, raw_klines_dir: Path = BINANCE_SPOT_RAW_KLINES):
        self.raw_klines_dir: Path = raw_klines_dir
        self._cache: Dict[Tuple[str, date], Optional[pd.DataFrame]] = {}

    def _kline_zip_path(self, symbol: str, day: date) -> Path:
        return self.raw_klines_dir / symbol / f"klines@1m@{str(day)}.zip"

    def _load_symbol_day_klines(self, symbol: str, day: date) -> Optional[pd.DataFrame]:
        cache_key: Tuple[str, date] = (symbol, day)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path: Path = self._kline_zip_path(symbol=symbol, day=day)
        if not path.exists():
            self._cache[cache_key] = None
            return None

        df: pd.DataFrame = pd.read_csv(
            path,
            compression="zip",
            header=None,
            names=BINANCE_KLINES_COLS,
            usecols=[
                OPEN_TIME,
                OPEN_PRICE,
                CLOSE_PRICE,
                QUOTE_ASSET_VOLUME,
                TAKER_BUY_QUOTE_ASSET_VOLUME,
            ],
        )
        if df.empty:
            self._cache[cache_key] = None
            return None

        df[OPEN_TIME] = pd.to_datetime(df[OPEN_TIME], unit="ms")
        for col in [OPEN_PRICE, CLOSE_PRICE, QUOTE_ASSET_VOLUME, TAKER_BUY_QUOTE_ASSET_VOLUME]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[OPEN_TIME, OPEN_PRICE, CLOSE_PRICE, QUOTE_ASSET_VOLUME]).sort_values(OPEN_TIME)
        df = df[~df[OPEN_TIME].duplicated(keep="last")].reset_index(drop=True)
        self._cache[cache_key] = df
        return df

    @staticmethod
    def _full_minute_index(bounds: Bounds) -> pd.DatetimeIndex:
        start_ts = pd.Timestamp(bounds.start_inclusive)
        end_ts = pd.Timestamp(bounds.end_exclusive)

        start_minute = start_ts.floor("min") if start_ts == start_ts.floor("min") else start_ts.ceil("min")
        end_minute = (end_ts - pd.Timedelta(microseconds=1)).floor("min")

        if end_minute < start_minute:
            return pd.DatetimeIndex([], name=OPEN_TIME)
        return pd.date_range(start=start_minute, end=end_minute, freq="1min", name=OPEN_TIME)

    def load_klines(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for day in bounds.date_range():
            df_day = self._load_symbol_day_klines(symbol=currency_pair.name, day=day)
            if df_day is not None and not df_day.empty:
                frames.append(df_day)

        if not frames:
            return pd.DataFrame(
                columns=[
                    OPEN_TIME,
                    OPEN_PRICE,
                    CLOSE_PRICE,
                    QUOTE_ASSET_VOLUME,
                    TAKER_BUY_QUOTE_ASSET_VOLUME,
                ]
            )

        minute_index = self._full_minute_index(bounds=bounds)
        if minute_index.empty:
            return pd.DataFrame(
                columns=[
                    OPEN_TIME,
                    OPEN_PRICE,
                    CLOSE_PRICE,
                    QUOTE_ASSET_VOLUME,
                    TAKER_BUY_QUOTE_ASSET_VOLUME,
                ]
            )

        df = pd.concat(frames, ignore_index=True)
        df = df.set_index(OPEN_TIME).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.reindex(minute_index)
        return df.reset_index()
