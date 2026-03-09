import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

from core.columns import DATE, IS_BUYER_MAKER, MID_PRICE, PRICE, QUANTITY, SAMPLED_TIME, SYMBOL, TRADE_TIME
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_SPOT_HIVE_TRADES, INDICATIVE_DATA_DIR
from core.time_utils import Bounds, format_date, start_of_the_day
from core.utils import configure_logging


class BinanceSpotIndicativeMidPriceWriter:

    def __init__(
            self,
            bounds: Bounds,
            input_dir: Path,
            output_dir: Path,
            currency_pair: CurrencyPair,
    ):
        self.bounds: Bounds = bounds
        self.input_dir: Path = input_dir
        self.output_dir: Path = output_dir
        self.currency_pair: CurrencyPair = currency_pair
        self._hive: pl.LazyFrame = pl.scan_parquet(input_dir, hive_partitioning=True)

    @property
    def output_file_name(self) -> str:
        return f"{self.currency_pair.base}.{self.currency_pair.term}.parquet"

    def load_trades_for_day(self, day: date) -> pl.DataFrame:
        return (
            self._hive.filter(
                (pl.col(SYMBOL) == self.currency_pair.name) & (pl.col(DATE) == day)
            )
            .select([TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER])
            .collect()
            .sort(TRADE_TIME)
        )

    @staticmethod
    def aggregate_ticks_into_trade_mids(df_ticks: pl.DataFrame) -> pl.DataFrame:
        if df_ticks.is_empty():
            return pl.DataFrame(
                schema={
                    TRADE_TIME: pl.Datetime("ns"),
                    MID_PRICE: pl.Float64,
                }
            )

        return (
            df_ticks.group_by(TRADE_TIME, maintain_order=True)
            .agg(
                price_first=pl.col(PRICE).first(),
                price_last=pl.col(PRICE).last(),
            )
            .with_columns(
                ((pl.col("price_first") + pl.col("price_last")) / 2.0).alias(MID_PRICE)
            )
            .select([TRADE_TIME, MID_PRICE])
        )

    @staticmethod
    def sample_trade_mids_to_minutes(df_trade_mids: pl.DataFrame, day: date) -> pl.DataFrame:
        if df_trade_mids.is_empty():
            return pl.DataFrame(
                schema={
                    SAMPLED_TIME: pl.Datetime("ns"),
                    MID_PRICE: pl.Float64,
                }
            )

        df_sampled = (
            df_trade_mids.group_by_dynamic(
                index_column=TRADE_TIME,
                every=timedelta(minutes=1),
                period=timedelta(minutes=1),
                label="left",
            )
            .agg(pl.col(MID_PRICE).last().alias(MID_PRICE))
            .rename({TRADE_TIME: SAMPLED_TIME})
            .sort(SAMPLED_TIME)
        )

        day_start: datetime = start_of_the_day(day)
        day_end: datetime = day_start + timedelta(days=1) - timedelta(minutes=1)
        time_dtype = df_sampled.schema[SAMPLED_TIME]
        minute_grid = pl.DataFrame(
            {
                SAMPLED_TIME: pl.datetime_range(
                    start=day_start,
                    end=day_end,
                    interval="1m",
                    eager=True,
                ).cast(time_dtype)
            }
        )

        return (
            minute_grid.join_asof(df_sampled, on=SAMPLED_TIME, strategy="backward")
            .drop_nulls(subset=[MID_PRICE])
            .select([SAMPLED_TIME, MID_PRICE])
        )

    def build_indicative_mid_price_for_day(self, day: date) -> pl.DataFrame:
        df_ticks = self.load_trades_for_day(day=day)
        df_trade_mids = self.aggregate_ticks_into_trade_mids(df_ticks=df_ticks)
        return self.sample_trade_mids_to_minutes(df_trade_mids=df_trade_mids, day=day)

    def output_path_for_day(self, day: date) -> Path:
        return self.output_dir / format_date(day) / self.output_file_name

    def write_day(self, day: date) -> None:
        df_sampled = self.build_indicative_mid_price_for_day(day=day)
        if df_sampled.is_empty():
            return

        output_path = self.output_path_for_day(day=day)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_sampled.write_parquet(output_path)

    def run(self) -> None:
        days = list(self.bounds.date_range())
        for day in tqdm(
                days,
                desc=f"Writing indicative mid prices for {self.currency_pair.name}",
        ):
            self.write_day(day=day)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write 1-minute indicative mid prices from Binance spot trades into "
            "daily parquet files."
        )
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--symbol", default="BTC-USDT", help="Currency pair in BASE-TERM format.")
    parser.add_argument(
        "--input-dir",
        default=str(BINANCE_SPOT_HIVE_TRADES),
        help="Input Hive parquet directory with Binance spot trades.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(INDICATIVE_DATA_DIR),
        help="Output directory for indicative daily parquet files.",
    )
    return parser.parse_args()


def run_main() -> None:
    configure_logging()
    bounds = Bounds.for_days(
        start_inclusive=date(2024, 1, 1),
        end_exclusive=date(2025, 1, 1),
    )
    pipe = BinanceSpotIndicativeMidPriceWriter(
        bounds=bounds,
        input_dir=Path(BINANCE_SPOT_HIVE_TRADES),
        output_dir=Path(INDICATIVE_DATA_DIR),
        currency_pair=CurrencyPair.from_string("BTC-USDT"),
    )
    pipe.run()


if __name__ == "__main__":
    run_main()
