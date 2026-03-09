from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.utils.IndicativePriceProvider import IndicativePriceProvider
from core.columns import BINANCE_KLINES_COLS


def _write_1m_kline_zip(
    root_dir: Path,
    symbol: str,
    day: str,
    rows: list[list[float | int | str]],
) -> None:
    path = root_dir / symbol / f"klines@1m@{day}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=BINANCE_KLINES_COLS).to_csv(
        path,
        compression="zip",
        index=False,
        header=False,
    )


def test_quote_to_usdt_uses_direct_symbol_vwap(tmp_path: Path) -> None:
    _write_1m_kline_zip(
        root_dir=tmp_path,
        symbol="BTC-USDT",
        day="2025-01-01",
        rows=[
            [
                int(datetime(2025, 1, 1, 12, 0).timestamp() * 1000),
                10000.0,
                10010.0,
                9990.0,
                10005.0,
                2.0,
                0,
                20_000.0,
                1,
                1.0,
                10_000.0,
                0,
            ],
            [
                int(datetime(2025, 1, 1, 12, 1).timestamp() * 1000),
                10500.0,
                10510.0,
                10490.0,
                10505.0,
                4.0,
                0,
                42_000.0,
                1,
                2.0,
                21_000.0,
                0,
            ],
        ],
    )

    converter = IndicativePriceProvider(raw_klines_dir=tmp_path)
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("BTC", datetime(2025, 1, 1, 12, 1, 30)),
        10_500.0,
    )
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("BTC", datetime(2025, 1, 1, 12, 2, 30)),
        10_500.0,
    )
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("USDT", datetime(2025, 1, 1, 12, 2, 30)),
        1.0,
    )


def test_quote_to_usdt_supports_inverse_symbol(tmp_path: Path) -> None:
    _write_1m_kline_zip(
        root_dir=tmp_path,
        symbol="USDT-BTC",
        day="2025-01-01",
        rows=[
            [
                int(datetime(2025, 1, 1, 12, 0).timestamp() * 1000),
                0.00005,
                0.000051,
                0.000049,
                0.00005,
                20_000.0,
                0,
                1.0,
                1,
                10_000.0,
                0.5,
                0,
            ],
        ],
    )

    converter = IndicativePriceProvider(raw_klines_dir=tmp_path)
    assert np.isclose(
        converter.get_quote_to_usdt_indicative_price("BTC", datetime(2025, 1, 1, 12, 0, 30)),
        20_000.0,
    )
