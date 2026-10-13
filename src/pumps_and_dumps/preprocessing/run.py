import logging
from datetime import date

import polars as pl

from pumps_and_dumps.core.columns import DATE
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.core.utils import configure_logging


def run_main():
    """Check if the data uploaded to HIVE is correct"""
    configure_logging()
    bounds: Bounds = Bounds.for_days(date(2025, 5, 1), date(2025, 5, 15))
    logging.info("Collecting data for %s", str(bounds))
    res = (
        pl.scan_parquet(
            Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True
        )
        .select(pl.col(DATE).min())
        .collect()
        .item()
    )

    print(res)


if __name__ == "__main__":
    run_main()
