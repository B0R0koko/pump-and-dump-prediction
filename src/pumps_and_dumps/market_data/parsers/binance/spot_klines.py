import argparse
from datetime import date
from collections.abc import Callable
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.paths import BINANCE_SPOT_RAW_KLINES
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.market_data.parsers.binance.base import BinanceBaseParser
from pumps_and_dumps.market_data.parsers.settings import SETTINGS

_SUPPORTED_INTERVALS: set[str] = {
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


class BinanceSpotKlinesParser(BinanceBaseParser):
    name = "binance_spot_klines_parser"

    def __init__(
        self,
        bounds: Bounds,
        currency_pairs: List[CurrencyPair],
        output_dir: Path,
        interval: str,
        *,
        skip_existing: bool = False,
        archive_verifier: Callable[..., None] | None = None,
        archive_installer: Callable[..., None] | None = None,
    ):
        super().__init__(
            bounds=bounds,
            currency_pairs=currency_pairs,
            output_dir=output_dir.expanduser().resolve(),
            skip_existing=skip_existing,
            archive_verifier=archive_verifier,
            archive_installer=archive_installer,
        )
        if interval not in _SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported Binance kline interval: {interval}")
        self.interval: str = interval

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/spot/daily/klines/{currency_pair.binance_name}/{self.interval}/"

    def output_zip_path(self, currency_pair: CurrencyPair, day: date) -> Path:
        return (
            self.output_dir
            / currency_pair.name
            / f"klines@{self.interval}@{str(day)}.zip"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Binance spot daily klines from data.binance.vision."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2018, 1, 1),
        help="Inclusive start date.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
        help="Exclusive end date.",
    )
    parser.add_argument(
        "--interval", default="1m", choices=sorted(_SUPPORTED_INTERVALS)
    )
    parser.add_argument(
        "--pairs", nargs="+", default=["BTC-USDT"], help="Pairs in BASE-QUOTE format."
    )
    parser.add_argument(
        "--quote-asset",
        help="Filter the explicitly supplied pairs by quote asset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BINANCE_SPOT_RAW_KLINES,
        help="Directory where downloaded zip files will be stored.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Verify existing ZIP sizes and full CRCs; download only missing or damaged archives",
    )
    return parser.parse_args()


def run_main() -> None:
    args = _parse_args()
    bounds: Bounds = Bounds.for_days(args.start_date, args.end_date)
    pairs = [CurrencyPair.from_string(name) for name in args.pairs]
    if args.quote_asset:
        pairs = [pair for pair in pairs if pair.term == args.quote_asset]
    if not pairs:
        raise ValueError("No kline pairs match the requested quote asset")
    output_dir = args.output_dir.expanduser().resolve()
    print(
        f"Collecting {len(pairs)} kline pairs from {args.start_date} to {args.end_date} (exclusive)"
    )
    print(f"Output: {output_dir}", flush=True)
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    crawler = process.create_crawler(BinanceSpotKlinesParser)
    process.crawl(
        crawler,
        bounds=bounds,
        currency_pairs=pairs,
        output_dir=output_dir,
        interval=args.interval,
        skip_existing=args.skip_existing,
    )
    process.start()
    raise SystemExit(crawler.spider.exit_code if crawler.spider is not None else 1)


if __name__ == "__main__":
    run_main()
