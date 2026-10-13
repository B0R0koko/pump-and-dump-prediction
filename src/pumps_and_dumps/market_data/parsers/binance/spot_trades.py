import argparse
import json
from datetime import date
from collections.abc import Callable
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.market_data.symbols import collect_all_spot_currency_pairs
from pumps_and_dumps.core.paths import BINANCE_SPOT_RAW_TRADES
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.market_data.parsers.binance.base import BinanceBaseParser
from pumps_and_dumps.market_data.parsers.settings import SETTINGS


class BinanceSpotTradesParser(BinanceBaseParser):
    name = "binance_spot_trades_parser"

    def __init__(
        self,
        bounds: Bounds,
        currency_pairs: List[CurrencyPair],
        output_dir: Path,
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

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/spot/daily/trades/{currency_pair.binance_name}/"

    def output_zip_path(self, currency_pair: CurrencyPair, day: date) -> Path:
        return self.output_dir / currency_pair.name / f"trades@{str(day)}.zip"


def run_main():
    parser = argparse.ArgumentParser(
        description="Download Binance spot daily trade archives."
    )
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=date(2018, 1, 1)
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
        help="Exclusive end date",
    )
    parser.add_argument("--output-dir", type=Path, default=BINANCE_SPOT_RAW_TRADES)
    pairs_source = parser.add_mutually_exclusive_group()
    pairs_source.add_argument(
        "--manifest",
        type=Path,
        help="Use historical pairs from an existing archive inventory.",
    )
    pairs_source.add_argument(
        "--pairs-file", type=Path, help="JSON object containing a currency_pairs list."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Verify existing ZIP sizes and full CRCs; download only missing or damaged archives",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    bounds: Bounds = Bounds.for_days(args.start_date, args.end_date)
    if args.pairs_file:
        names = sorted(
            set(json.loads(args.pairs_file.expanduser().read_text())["currency_pairs"])
        )
        btc_currencies = [CurrencyPair.from_string(name) for name in names]
    elif args.manifest:
        manifest = json.loads(args.manifest.expanduser().read_text())
        names = sorted(
            {
                item["currency_pair"]
                for item in manifest["archives"]
                if item["kind"] == "trades"
            }
        )
        btc_currencies = [CurrencyPair.from_string(name) for name in names]
    else:
        btc_currencies = [
            pair for pair in collect_all_spot_currency_pairs() if pair.term == "BTC"
        ]
    if not btc_currencies or any(pair.term != "BTC" for pair in btc_currencies):
        parser.error(
            "The trade pair list must contain at least one pair and only BTC-quoted pairs"
        )

    print(
        f"Collecting {len(btc_currencies)} currencies from {args.start_date} to {args.end_date} (exclusive)"
    )
    print(f"Output: {args.output_dir}", flush=True)

    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    crawler = process.create_crawler(BinanceSpotTradesParser)
    process.crawl(
        crawler,
        bounds=bounds,
        currency_pairs=btc_currencies,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
    )
    process.start()
    raise SystemExit(crawler.spider.exit_code if crawler.spider is not None else 1)


if __name__ == "__main__":
    run_main()
