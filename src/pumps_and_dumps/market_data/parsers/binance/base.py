import asyncio
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

import scrapy
from scrapy import signals
from scrapy.http import Request, Response
from tqdm import tqdm

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.progress import ProgressReporter, progress_enabled
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.market_data.parsers.binance.integrity import (
    ArchiveIntegrityError,
    install_archive,
    verify_archive,
)
from pumps_and_dumps.market_data.parsers.binance.listing import parse_listing

BINANCE_S3: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_DATAVISION: str = "https://data.binance.vision"

_log = logging.getLogger(__name__)


def filter_hrefs_by_bounds(
    hrefs: List[str], bounds: Bounds
) -> Tuple[List[str], List[date]]:
    """Takes bounds as input and returns a list of hrefs that matches passed in Bounds.

    Hrefs that do not contain a YYYY-MM-DD date are skipped with a warning rather than raising, so
    stray index entries do not abort the crawl.
    """

    filtered_hrefs: List[str] = []
    href_dates: List[date] = []

    pattern_str: str = r"\d{4}-\d{2}-\d{2}"

    for href in hrefs:
        # Find date in href string and parse it to datetime
        match: Optional[re.Match[str]] = re.search(pattern=pattern_str, string=href)
        if match is None:
            _log.warning("Skipping href with no parseable date: %s", href)
            continue
        href_date: date = datetime.strptime(match.group(0), "%Y-%m-%d").date()
        if bounds.contain_days(day=href_date):
            filtered_hrefs.append(href)
            href_dates.append(href_date)

    return filtered_hrefs, href_dates


def get_zip_file_url(href: str) -> str:
    """Returns a formatted url string which leads to a zip file with trades data"""
    return f"{BINANCE_S3}/{href}"


class BinanceBaseParser(ABC, scrapy.Spider):
    def __init__(
        self,
        bounds: Bounds,
        currency_pairs: List[CurrencyPair],
        output_dir: Path,
        *args,
        skip_existing: bool = False,
        archive_verifier: Callable[..., None] | None = None,
        archive_installer: Callable[..., None] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bounds: Bounds = bounds
        self.currency_pairs: List[CurrencyPair] = currency_pairs
        self.output_dir: Path = output_dir
        self.skip_existing: bool = skip_existing
        self._verify_archive = (
            verify_archive if archive_verifier is None else archive_verifier
        )
        self._install_archive = (
            install_archive if archive_installer is None else archive_installer
        )
        self.exit_code: int = 0
        self._scheduled_hrefs: Set[str] = set()
        self._pbar: Optional[tqdm] = None
        self._shutdown_timer_started: bool = False
        # (currency_pair, day) tuples whose zip download was rejected as empty or malformed. Logged
        # at spider close so operators can reconcile against the requested range.
        self.failed_downloads: List[Tuple[CurrencyPair, date]] = []
        self._listed_pairs: Set[str] = set()
        self._progress = ProgressReporter(
            unit="files", detail="Discovering available archives"
        )
        self._downloaded = self._verified = self._failed = self._received_bytes = 0
        self._damaged = self._missing = 0
        # CRC reads run outside Scrapy's event loop, with bounded disk concurrency.
        self._archive_slots = asyncio.Semaphore(4)

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider._on_spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(spider._on_spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(spider._on_error, signal=signals.spider_error)
        crawler.signals.connect(
            spider._on_bytes_received, signal=signals.bytes_received
        )
        return spider

    def _on_spider_opened(self, **_kwargs) -> None:
        self._pbar = tqdm(
            total=0,
            desc=self.name,
            unit="file",
            dynamic_ncols=True,
            disable=progress_enabled(),
        )

    def _report_progress(self, detail: str, *, force: bool = False) -> None:
        listed = len(self._listed_pairs)
        self._progress.update(
            completed=self._downloaded + self._verified + self._failed,
            total=len(self._scheduled_hrefs),
            total_known=listed == len(self.currency_pairs),
            downloaded=self._downloaded,
            verified=self._verified,
            damaged=self._damaged,
            missing=self._missing,
            failed=self._failed,
            received_bytes=self._received_bytes,
            detail=f"Listings {listed}/{len(self.currency_pairs)} | {detail}",
            force=force,
        )

    def _on_bytes_received(self, data: bytes, request: Request, **_kwargs) -> None:
        if "day" not in request.meta:
            return
        self._received_bytes += len(data)
        self._report_progress(
            f"Receiving {request.meta['currency_pair'].name} {request.meta['day']}"
        )

    def _on_error(self, failure, **_kwargs) -> None:
        self.exit_code = 1
        request = getattr(failure, "request", None)
        if request is not None and "day" in request.meta:
            self._failed += 1
        self._report_progress(failure.getErrorMessage(), force=True)
        _log.error("%s failed: %s", self.name, failure.getErrorMessage())

    def _on_spider_closed(self, reason: str = "finished", **_kwargs) -> None:
        if reason != "finished" or not self._collection_complete():
            self.exit_code = 1
        summary = self._completion_summary(reason)
        self._report_progress(summary, force=True)
        self._progress.update(
            status="done" if self.exit_code == 0 else "failed", force=True
        )
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        print(summary, flush=True)
        self._log_failed_downloads()
        self._schedule_shutdown()

    def _collection_complete(self) -> bool:
        if self.failed_downloads:
            return False
        if len(self._listed_pairs) != len(self.currency_pairs):
            return False
        return self._downloaded + self._verified == len(self._scheduled_hrefs)

    def _completion_summary(self, reason: str) -> str:
        if self.exit_code:
            return (
                f"Archive check incomplete ({reason}): {len(self._listed_pairs)}/{len(self.currency_pairs)} listings, "
                f"{self._verified:,} verified, {self._downloaded:,} downloaded, {self._failed:,} failed files."
            )
        if not self._scheduled_hrefs:
            return "No Binance archives available for the requested pairs and dates."
        if self._downloaded == 0:
            return f"All {self._verified:,} available archives verified; no downloads needed."
        return f"Archives ready: {self._verified:,} verified, {self._downloaded:,} downloaded."

    def _log_failed_downloads(self) -> None:
        if not self.failed_downloads:
            return
        failed_repr = ", ".join(
            f"({cp.name}, {day})" for cp, day in self.failed_downloads
        )
        _log.warning(
            "%s finished with %d failed downloads: %s",
            self.name,
            len(self.failed_downloads),
            failed_repr,
        )

    def _schedule_shutdown(self) -> None:
        if self._shutdown_timer_started:
            return
        self._shutdown_timer_started = True
        # Safety net: scrapy's reactor sometimes won't shut down after the
        # spider finishes because the HTTP connection pool is still holding
        # keep-alive connections. Force-exit after a short grace period so
        # the CLI actually returns.
        timer = threading.Timer(3.0, lambda: os._exit(self.exit_code))
        timer.daemon = True
        timer.start()

    @abstractmethod
    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        """Return prefix"""

    def output_zip_path(self, currency_pair: CurrencyPair, day: date) -> Path:
        """Where the downloaded zip for a given currency pair / day is stored."""
        return self.output_dir / currency_pair.name / f"trades@{str(day)}.zip"

    def _get_currency_url(
        self, currency_pair: CurrencyPair, marker: Optional[str] = None
    ) -> str:
        params: Dict[str, str] = {
            "delimiter": "/",
            "prefix": self.get_prefix(currency_pair=currency_pair),
        }

        if marker is not None:
            params["marker"] = marker

        datavision_url: str = f"{BINANCE_S3}?{urlencode(params)}"
        return datavision_url

    async def start(self) -> AsyncIterator[Request]:
        """Yield the initial archive-listing requests for Scrapy's async startup."""
        for currency_pair in self.currency_pairs:
            yield self._listing_request(currency_pair)

    def _listing_request(
        self, currency_pair: CurrencyPair, marker: str | None = None
    ) -> Request:
        meta: dict[str, object] = {"currency_pair": currency_pair}
        if marker is not None:
            meta["listing_marker"] = marker
        return Request(
            url=self._get_currency_url(currency_pair, marker),
            callback=self._parse_currency_pair,
            errback=self._on_error,
            meta=meta,
        )

    async def _parse_currency_pair(self, response: Response) -> AsyncIterator[Request]:
        """Check every listed archive, scheduling only missing or damaged ZIPs."""

        currency_pair = response.meta["currency_pair"]
        sizes, marker = parse_listing(
            response.body, response.meta.get("listing_marker")
        )
        archives = self._reserve_archives(sizes)
        if marker is not None:
            yield self._listing_request(currency_pair, marker)
        for href, day in archives:
            if await self._reuse_archive(currency_pair, day, sizes[href]):
                continue
            self._report_progress(f"Queued {currency_pair.name} {day}")
            yield Request(
                url=get_zip_file_url(href),
                callback=self._parse_zip_file,
                errback=self._on_error,
                meta={
                    "currency_pair": currency_pair,
                    "day": day,
                    "archive_size": sizes[href],
                },
            )
        if marker is None:
            self._listed_pairs.add(currency_pair.name)
        self._report_progress(f"Listed {currency_pair.name}", force=True)

    def _reserve_archives(self, sizes: dict[str, int]) -> list[tuple[str, date]]:
        hrefs = [href for href in sizes if href.endswith(".zip")]
        filtered_hrefs, href_dates = filter_hrefs_by_bounds(
            hrefs=hrefs, bounds=self.bounds
        )
        archives = [
            (href, day)
            for href, day in zip(filtered_hrefs, href_dates)
            if href not in self._scheduled_hrefs
        ]
        # Count the whole page before yielding: a later page can finish while this
        # page is still validating large files in a background thread.
        self._scheduled_hrefs.update(href for href, _day in archives)
        if self._pbar is not None:
            self._pbar.total += len(archives)
            self._pbar.refresh()
        return archives

    async def _reuse_archive(
        self, currency_pair: CurrencyPair, day: date, size: int
    ) -> bool:
        path = self.output_zip_path(currency_pair, day)
        if not path.exists():
            self._missing += 1
            return False
        if not self.skip_existing or not path.is_file():
            return False
        detail = f"Verifying {currency_pair.name} {day}"
        self._report_progress(detail)
        try:
            await self._archive_io(
                self._verify_archive, path, expected_size=size, detail=detail
            )
        except ArchiveIntegrityError as exc:
            self._damaged += 1
            _log.warning("Repairing damaged archive %s: %s", path, exc)
            return False
        self._verified += 1
        self._finish_file(f"Verified {currency_pair.name} {day}")
        return True

    async def _archive_io(self, operation, *args, **kwargs) -> None:
        """Bound disk concurrency without blocking Scrapy's event loop."""
        async with self._archive_slots:
            await asyncio.to_thread(operation, *args, **kwargs)

    def _finish_file(self, detail: str, *, force: bool = False) -> None:
        self._report_progress(detail, force=force)
        if self._pbar is not None:
            self._pbar.update(1)

    async def _parse_zip_file(self, response: Response) -> None:
        day = response.meta["day"]
        currency_pair = response.meta["currency_pair"]

        path: Path = self.output_zip_path(currency_pair=currency_pair, day=day)
        try:
            await self._archive_io(
                self._install_archive,
                path,
                response.body,
                expected_size=response.meta.get("archive_size"),
                detail=f"Validating download {currency_pair.name} {day}",
            )
        except ArchiveIntegrityError as exc:
            _log.warning(
                "Rejecting invalid ZIP response for %s @ %s: %s",
                currency_pair.name,
                day,
                exc,
            )
            self.failed_downloads.append((currency_pair, day))
            self.exit_code = 1
            self._failed += 1
            self._finish_file(f"Rejected {currency_pair.name} {day}", force=True)
            return

        self._downloaded += 1
        self._finish_file(f"Saved {currency_pair.name} {day}")
