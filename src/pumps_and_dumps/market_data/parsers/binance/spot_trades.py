import argparse
import json
from datetime import date
from collections.abc import Callable
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from pumps_and_dumps.core.cli_config import parse_config_args
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.market_data.pair_selection import parse_pair_names, read_pairs_file
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


def run_main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Download Binance spot daily trade archives.")
    parser.add_argument("--start-date", "--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument(
        "--end-date",
        "--end",
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
    pairs_source.add_argument("--pairs-file", type=Path, help="JSON array or object containing a currency_pairs list.")
    pairs_source.add_argument("--pairs", "--instruments", nargs="+", help="BTC-quoted pairs in BASE-QUOTE format.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify existing ZIP sizes and full CRCs; download only missing or damaged archives",
    )
    args = parse_config_args(parser, argv, path_keys=("output_dir", "manifest", "pairs_file"))
    if args.start_date >= args.end_date:
        parser.error("--start must be before the exclusive --end")
    args.output_dir = args.output_dir.expanduser().resolve()
    bounds: Bounds = Bounds.for_days(args.start_date, args.end_date)
    try:
        if args.pairs is not None:
            btc_currencies = parse_pair_names(args.pairs)
        elif args.pairs_file:
            btc_currencies = read_pairs_file(args.pairs_file)
        elif args.manifest:
            manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(manifest.get("archives"), list):
                raise ValueError("Archive manifest must contain an archives list")
            names = [item["currency_pair"] for item in manifest["archives"] if item["kind"] == "trades"]
            btc_currencies = parse_pair_names(names)
        else:
            btc_currencies = [pair for pair in collect_all_spot_currency_pairs() if pair.term == "BTC"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f"Cannot read trade instruments: {error}")
    if not btc_currencies or any(pair.term != "BTC" for pair in btc_currencies):
        parser.error("The trade pair list must contain at least one pair and only BTC-quoted pairs")

    print(f"Collecting {len(btc_currencies)} currencies from {args.start_date} to {args.end_date} (exclusive)")
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
