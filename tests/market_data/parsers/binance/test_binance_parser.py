"""Regression tests for `BinanceParser` download-integrity and href-filter guards.

These tests do not require the network; scrapy `Response` objects are constructed directly with
synthetic bodies, and the spider's file I/O is redirected under `tmp_path`.
"""

import asyncio
import importlib
import json
import logging
import sys
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import List
from urllib.parse import parse_qs, urlparse

import pytest
from scrapy.http import Request, TextResponse
from twisted.python.failure import Failure

from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.time_utils import Bounds
from pumps_and_dumps.market_data.parsers.binance.base import (
    BinanceBaseParser,
    filter_hrefs_by_bounds,
)
from pumps_and_dumps.market_data.parsers.binance.spot_klines import (
    BinanceSpotKlinesParser,
)
from pumps_and_dumps.market_data.parsers.binance.spot_trades import (
    BinanceSpotTradesParser,
)
from pumps_and_dumps.market_data.parsers.binance.integrity import (
    ArchiveIntegrityError,
    install_archive,
    verify_archive,
)


class _StubBinanceParser(BinanceBaseParser):
    """Concrete subclass so the abstract `BinanceBaseParser` can be instantiated in tests."""

    name = "stub_binance_parser"

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/spot/daily/trades/{currency_pair.binance_name}/"


def _make_parser(tmp_path: Path) -> _StubBinanceParser:
    return _StubBinanceParser(
        bounds=Bounds.for_day(date(2021, 6, 16)),
        currency_pairs=[CurrencyPair.from_string("BTC-USDT")],
        output_dir=tmp_path,
    )


def _make_zip_response(
    url: str, currency_pair: CurrencyPair, day: date, body: bytes
) -> TextResponse:
    request: Request = Request(
        url=url, meta={"currency_pair": currency_pair, "day": day}
    )
    return TextResponse(url=url, request=request, body=body, headers={})


def _make_real_zip_bytes() -> bytes:
    buffer: BytesIO = BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zf:
        zf.writestr("file.csv", "a,b,c\n1,2,3\n")
    return buffer.getvalue()


def _corrupt_zip_bytes() -> bytes:
    # Keep the directory, length and header valid; only a full CRC check detects this.
    return _make_real_zip_bytes().replace(b"1,2,3", b"9,2,3")


def _listing_response(pair, entries, *, truncated=False, marker=None, request=None):
    request = request or Request(
        "https://example.com/list", meta={"currency_pair": pair}
    )
    contents = "".join(
        f"<Contents><Key>{key}</Key><Size>{size}</Size></Contents>"
        for key, size in entries
    )
    next_marker = f"<NextMarker>{marker}</NextMarker>" if marker else ""
    body = (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{next_marker}{contents}</ListBucketResult>"
    )
    return TextResponse(request.url, request=request, body=body.encode())


async def _collect_requests(generator):
    return [request async for request in generator]


@pytest.mark.parametrize(
    "parser_class, options, prefix",
    [
        (BinanceSpotTradesParser, {}, "data/spot/daily/trades/{symbol}/"),
        (
            BinanceSpotKlinesParser,
            {"interval": "1m"},
            "data/spot/daily/klines/{symbol}/1m/",
        ),
    ],
)
@pytest.mark.parametrize("pair_names", [[], ["ETH-BTC", "THETA-BTC"]])
def test_async_start_schedules_currency_pairs(
    tmp_path: Path, parser_class, options, prefix, pair_names
) -> None:
    pairs = [CurrencyPair.from_string(name) for name in pair_names]
    parser = parser_class(
        bounds=Bounds.for_day(date(2021, 6, 16)),
        currency_pairs=pairs,
        output_dir=tmp_path,
        **options,
    )

    async def collect_requests():
        return [request async for request in parser.start()]

    requests = asyncio.run(collect_requests())
    assert len(requests) == len(pairs)
    for request, pair in zip(requests, pairs):
        assert parse_qs(urlparse(request.url).query) == {
            "delimiter": ["/"],
            "prefix": [prefix.format(symbol=pair.binance_name)],
        }
        assert request.callback == parser._parse_currency_pair
        assert request.meta == {"currency_pair": pair}


def test_filter_hrefs_by_bounds_skips_href_with_no_date(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bounds: Bounds = Bounds.for_day(date(2021, 6, 16))
    hrefs: List[str] = [
        "data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2021-06-16.zip",
        "data/spot/daily/trades/BTCUSDT/index.html",  # no date
        "data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2021-06-15.zip",
    ]

    with caplog.at_level(logging.WARNING):
        filtered, dates = filter_hrefs_by_bounds(hrefs=hrefs, bounds=bounds)

    assert filtered == ["data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2021-06-16.zip"]
    assert dates == [date(2021, 6, 16)]
    assert any("no parseable date" in rec.message for rec in caplog.records)


def test_parse_zip_file_writes_when_body_is_real_zip(tmp_path: Path) -> None:
    parser: _StubBinanceParser = _make_parser(tmp_path)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    day: date = date(2021, 6, 16)
    body: bytes = _make_real_zip_bytes()
    response: TextResponse = _make_zip_response(
        "http://example/BTCUSDT.zip", currency_pair, day, body
    )

    asyncio.run(parser._parse_zip_file(response))

    written: Path = parser.output_zip_path(currency_pair=currency_pair, day=day)
    assert written.exists()
    assert written.read_bytes() == body
    assert parser.failed_downloads == []


def test_parse_zip_file_rejects_empty_body(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parser: _StubBinanceParser = _make_parser(tmp_path)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    day: date = date(2021, 6, 16)
    response: TextResponse = _make_zip_response(
        "http://example/BTCUSDT.zip", currency_pair, day, body=b""
    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(parser._parse_zip_file(response))

    written: Path = parser.output_zip_path(currency_pair=currency_pair, day=day)
    assert not written.exists()
    assert parser.failed_downloads == [(currency_pair, day)]
    assert any(
        "Rejecting invalid ZIP response" in rec.message for rec in caplog.records
    )


def test_parse_zip_file_rejects_non_zip_body(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parser: _StubBinanceParser = _make_parser(tmp_path)
    currency_pair: CurrencyPair = CurrencyPair.from_string("BTC-USDT")
    day: date = date(2021, 6, 16)
    # Simulates the AWS S3 XML error page that occasionally comes back instead of a zip.
    body: bytes = b'<?xml version="1.0"?><Error><Code>NoSuchKey</Code></Error>'
    response: TextResponse = _make_zip_response(
        "http://example/BTCUSDT.zip", currency_pair, day, body
    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(parser._parse_zip_file(response))

    written: Path = parser.output_zip_path(currency_pair=currency_pair, day=day)
    assert not written.exists()
    assert parser.failed_downloads == [(currency_pair, day)]


@pytest.mark.parametrize(
    "parser_class,options",
    [
        (BinanceSpotTradesParser, {}),
        (BinanceSpotKlinesParser, {"interval": "1m"}),
    ],
)
@pytest.mark.parametrize(
    "skip_existing,archive,expected_downloads",
    [
        (True, "valid", 0),
        (True, "truncated", 1),
        (True, "crc_error", 1),
        (False, "valid", 1),
    ],
)
def test_resume_downloads_reuses_only_fully_verified_zips(
    tmp_path, parser_class, options, skip_existing, archive, expected_downloads
):
    pair = CurrencyPair.from_string("ETH-BTC")
    day = date(2024, 12, 31)
    parser = parser_class(
        bounds=Bounds.for_day(day),
        currency_pairs=[pair],
        output_dir=tmp_path,
        skip_existing=skip_existing,
        **options,
    )
    path = parser.output_zip_path(pair, day)
    path.parent.mkdir(parents=True)
    body = {
        "valid": _make_real_zip_bytes(),
        "truncated": b"PK-incomplete-zip",
        "crc_error": _corrupt_zip_bytes(),
    }[archive]
    path.write_bytes(body)
    before = path.stat().st_mtime_ns
    response = _listing_response(
        pair, [("ETHBTC-2024-12-31.zip", len(_make_real_zip_bytes()))]
    )
    requests = asyncio.run(_collect_requests(parser._parse_currency_pair(response)))
    assert len(requests) == expected_downloads
    assert path.read_bytes() == body
    assert path.stat().st_mtime_ns == before
    assert parser._progress.fields["verified"] == (1 if expected_downloads == 0 else 0)
    assert parser._progress.fields["damaged"] == (
        1 if skip_existing and archive != "valid" else 0
    )
    assert all(request.errback == parser._on_error for request in requests)


def test_corrupt_download_does_not_replace_existing_file(tmp_path):
    parser = _make_parser(tmp_path)
    pair, day = parser.currency_pairs[0], date(2021, 6, 16)
    path = parser.output_zip_path(pair, day)
    path.parent.mkdir()
    original = _make_real_zip_bytes()
    path.write_bytes(original)
    damaged = _corrupt_zip_bytes()
    assert zipfile.is_zipfile(BytesIO(damaged))
    response = _make_zip_response("https://example/archive.zip", pair, day, damaged)
    asyncio.run(parser._parse_zip_file(response))
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    assert parser.exit_code == 1
    assert parser._progress.fields["downloaded"] == 0
    assert parser._progress.fields["failed"] == 1


def test_interrupted_atomic_install_preserves_destination_and_removes_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "data.zip"
    path.write_bytes(b"old archive")

    def interrupted_replace(*_args):
        raise OSError("disk failure")

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="disk failure"):
        install_archive(path, _make_real_zip_bytes())
    assert path.read_bytes() == b"old archive"
    assert list(tmp_path.iterdir()) == [path]


def test_verification_rejects_empty_zip_and_size_mismatch(tmp_path):
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(ArchiveIntegrityError, match="no CSV"):
        verify_archive(empty)
    with pytest.raises(ArchiveIntegrityError, match="Binance lists"):
        verify_archive(BytesIO(_make_real_zip_bytes()), expected_size=1)


def test_verification_checks_every_member(tmp_path):
    path = tmp_path / "multiple.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("first.csv", "valid data")
        archive.writestr("second.csv", "more data")
    path.write_bytes(path.read_bytes().replace(b"more data", b"lost data"))
    assert zipfile.is_zipfile(path)
    with pytest.raises(ArchiveIntegrityError, match="CRC"):
        verify_archive(path)


@pytest.mark.parametrize("explicit_marker", [True, False])
def test_listing_pagination_uses_truncation_and_includes_missing_archives(
    tmp_path, explicit_marker
):
    pair = CurrencyPair.from_string("ETH-BTC")
    parser = BinanceSpotTradesParser(
        bounds=Bounds.for_days(date(2024, 12, 30), date(2025, 1, 1)),
        currency_pairs=[pair],
        output_dir=tmp_path,
        skip_existing=True,
    )
    first_key = "ETHBTC-trades-2024-12-30.zip"
    last_key = first_key + ".CHECKSUM"
    response = _listing_response(
        pair,
        [(first_key, 100), (last_key, 64)],
        truncated=True,
        marker=last_key if explicit_marker else None,
    )
    first = asyncio.run(_collect_requests(parser._parse_currency_pair(response)))
    assert len(first) == 2
    next_page, download = first
    assert parse_qs(urlparse(next_page.url).query)["marker"] == [last_key]
    assert download.meta["day"] == date(2024, 12, 30)
    assert parser._progress.fields["total_known"] is False
    response = _listing_response(
        pair,
        [
            (first_key, 100),  # overlapping page must not double-count
            ("ETHBTC-trades-2024-12-31.zip", 100),
            ("ETHBTC-trades-2025-01-01.zip", 100),  # exclusive end
        ],
        request=next_page,
    )
    second = asyncio.run(_collect_requests(parser._parse_currency_pair(response)))
    assert len(second) == 1
    assert second[0].meta["day"] == date(2024, 12, 31)
    assert parser._progress.fields["total_known"] is True
    assert parser._progress.fields["total"] == parser._progress.fields["missing"] == 2


@pytest.mark.parametrize(
    "body",
    [
        b"<html/>",
        b"<ListBucketResult/>",
        b"<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>",
    ],
)
def test_invalid_or_nonadvancing_listing_cannot_report_complete(tmp_path, body):
    parser = _make_parser(tmp_path)
    pair = parser.currency_pairs[0]
    request = Request("https://example/list", meta={"currency_pair": pair})
    response = TextResponse(request.url, request=request, body=body)
    with pytest.raises(ValueError):
        asyncio.run(_collect_requests(parser._parse_currency_pair(response)))
    parser._shutdown_timer_started = True
    parser._on_spider_closed()
    assert parser.exit_code == 1
    assert parser._progress.fields["status"] == "failed"


def test_complete_collection_reports_verification_without_rewriting(tmp_path, capsys):
    parser = _make_parser(tmp_path)
    parser.skip_existing = True
    pair, day = parser.currency_pairs[0], date(2021, 6, 16)
    path = parser.output_zip_path(pair, day)
    path.parent.mkdir()
    body = _make_real_zip_bytes()
    path.write_bytes(body)
    response = _listing_response(pair, [("BTCUSDT-trades-2021-06-16.zip", len(body))])
    assert asyncio.run(_collect_requests(parser._parse_currency_pair(response))) == []
    parser._shutdown_timer_started = True
    parser._on_spider_closed()
    assert parser.exit_code == 0
    assert (
        "All 1 available archives verified; no downloads needed"
        in capsys.readouterr().out
    )
    assert parser._progress.fields["downloaded"] == 0
    assert parser._progress.fields["completed"] == parser._progress.fields["total"] == 1


def test_request_failure_sets_unsuccessful_exit_status(tmp_path):
    parser = _make_parser(tmp_path)
    parser._on_error(Failure(ConnectionError("download interrupted")))
    assert parser.exit_code == 1


def test_download_progress_reports_bytes_before_file_finishes(tmp_path):
    parser = _make_parser(tmp_path)
    pair = CurrencyPair.from_string("BTC-USDT")
    day = date(2021, 6, 16)
    request = Request(
        "https://example/archive.zip", meta={"currency_pair": pair, "day": day}
    )
    parser._on_bytes_received(b"part of a large archive", request)
    assert parser._progress.fields["received_bytes"] == len(b"part of a large archive")
    assert parser._progress.fields["downloaded"] == 0
    assert parser._progress.fields["total_known"] is False
    failure = Failure(ConnectionError("transfer failed"))
    failure.request = request
    parser._on_error(failure)
    assert parser._progress.fields["failed"] == 1
    assert parser.exit_code == 1


@pytest.mark.parametrize(
    "initial_code,reason,expected",
    [(0, "finished", 0), (1, "finished", 1), (0, "shutdown", 1)],
)
def test_shutdown_fallback_preserves_error_status(
    tmp_path, monkeypatch, initial_code, reason, expected
):
    module = importlib.import_module("pumps_and_dumps.market_data.parsers.binance.base")
    exit_codes = []
    monkeypatch.setattr(module.os, "_exit", exit_codes.append)
    monkeypatch.setattr(
        module.threading,
        "Timer",
        lambda delay, callback: SimpleNamespace(start=callback),
    )
    parser = _make_parser(tmp_path)
    parser._listed_pairs.update(pair.name for pair in parser.currency_pairs)
    parser.exit_code = initial_code
    parser._on_spider_closed(reason=reason)
    assert exit_codes == [expected]


@pytest.mark.parametrize("module_name", ["spot_trades", "spot_klines"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_download_cli_honors_dates_pairs_resume_and_failure(
    tmp_path, monkeypatch, module_name, exit_code
):
    module = importlib.import_module(
        f"pumps_and_dumps.market_data.parsers.binance.{module_name}"
    )
    calls = {}

    class FakeProcess:
        def __init__(self, settings):
            pass

        def create_crawler(self, parser_class):
            return SimpleNamespace(spider=SimpleNamespace(exit_code=exit_code))

        def crawl(self, crawler, **kwargs):
            calls.update(kwargs)

        def start(self):
            pass

    args = [
        "download",
        "--start-date",
        "2024-12-31",
        "--end-date",
        "2025-01-01",
        "--output-dir",
        str(tmp_path),
        "--skip-existing",
    ]
    if module_name == "spot_trades":
        pairs_file = tmp_path / "pairs.json"
        pairs_file.write_text(
            json.dumps({"currency_pairs": ["ETH-BTC", "ETH-BTC", "BNB-BTC"]})
        )
        args += ["--pairs-file", str(pairs_file)]
        expected_pairs = ["BNB-BTC", "ETH-BTC"]
        monkeypatch.setattr(
            module,
            "collect_all_spot_currency_pairs",
            lambda: pytest.fail("Unexpected exchange lookup"),
        )
    else:
        args += ["--pairs", "BTC-USDT", "--interval", "5m"]
        expected_pairs = ["BTC-USDT"]
    monkeypatch.setattr(module, "CrawlerProcess", FakeProcess)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as result:
        module.run_main()
    assert result.value.code == exit_code
    assert list(calls["bounds"].date_range()) == [date(2024, 12, 31)]
    assert [pair.name for pair in calls["currency_pairs"]] == expected_pairs
    assert calls["output_dir"] == tmp_path.resolve()
    assert calls["skip_existing"] is True
    if module_name == "spot_klines":
        assert calls["interval"] == "5m"
