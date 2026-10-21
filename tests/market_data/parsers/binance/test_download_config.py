"""Offline CLI contracts for configured Binance archive download jobs."""

import importlib
import json
from datetime import date
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["spot_trades", "spot_klines"])
def downloader(request, monkeypatch):
    """Replace networking with a crawler that records the exact requested job."""
    module = importlib.import_module(f"pumps_and_dumps.market_data.parsers.binance.{request.param}")
    calls = {}

    class FakeProcess:
        def __init__(self, settings):
            calls["started"] = False

        def create_crawler(self, parser_class):
            return SimpleNamespace(spider=SimpleNamespace(exit_code=0))

        def crawl(self, crawler, **kwargs):
            calls.update(kwargs)

        def start(self):
            calls["started"] = True

    monkeypatch.setattr(module, "CrawlerProcess", FakeProcess)
    if request.param == "spot_trades":
        monkeypatch.setattr(
            module,
            "collect_all_spot_currency_pairs",
            lambda: pytest.fail("An explicit instrument selection must not query the exchange"),
        )
    return module, calls


def test_hocon_download_configuration_and_cli_overrides(tmp_path, downloader):
    """Translate a portable HOCON job into the exact crawler inputs.

    The dummy file requests two BTC-quoted instruments, a two-day interval,
    relative output storage, and archive reuse. CLI flags shorten the interval
    to one day and disable reuse. The recorded crawler must receive precisely
    those effective settings, protecting against ignored overrides, paths
    resolved against the working directory, and inclusive end-date mistakes.
    Klines additionally checks that its configured interval can be overridden.
    """
    module, calls = downloader
    config = tmp_path / "download.conf"
    contents = """
        start = "2021-06-15"
        end = "2021-06-17"
        instruments = ["ETH-BTC", "BNB-BTC", "ETH-BTC"]
        output-dir = "archives"
        skip-existing = true
    """
    argv = ["--config", str(config), "--start", "2021-06-16", "--no-skip-existing"]
    if module.__name__.endswith("spot_klines"):
        contents += '\ninterval = "1m"\n'
        argv += ["--interval", "5m"]
    config.write_text(contents)

    with pytest.raises(SystemExit) as result:
        module.run_main(argv)

    assert result.value.code == 0
    assert calls["started"] is True
    assert list(calls["bounds"].date_range()) == [date(2021, 6, 16)]
    assert [pair.name for pair in calls["currency_pairs"]] == ["BNB-BTC", "ETH-BTC"]
    assert calls["output_dir"] == tmp_path / "archives"
    assert calls["skip_existing"] is False
    if module.__name__.endswith("spot_klines"):
        assert calls["interval"] == "5m"


@pytest.mark.parametrize("as_object", [False, True])
def test_json_config_resolves_pair_file_beside_config(tmp_path, downloader, as_object):
    """Read an instrument manifest relative to a JSON job configuration.

    A one-pair manifest is written as either a bare array or the project's
    existing currency_pairs object. The job refers only to its filename.
    Both download commands must select ETH-BTC and preserve the exclusive
    end date; this guards portable bind-mounted configs and legacy manifests.
    """
    module, calls = downloader
    names = ["ETH-BTC"]
    (tmp_path / "pairs.json").write_text(json.dumps({"currency_pairs": names} if as_object else names))
    config = tmp_path / "download.json"
    config.write_text(json.dumps({"start": "2021-06-16", "end": "2021-06-17", "pairs_file": "pairs.json"}))

    with pytest.raises(SystemExit) as result:
        module.run_main(["--config", str(config)])

    assert result.value.code == 0
    assert [pair.name for pair in calls["currency_pairs"]] == names
    assert list(calls["bounds"].date_range()) == [date(2021, 6, 16)]


def test_explicit_instruments_replace_configured_manifest(tmp_path, downloader):
    """An explicit instrument list overrides an alternative config source.

    The HOCON file deliberately references a nonexistent pairs file. Passing
    --instruments ETH-BTC must replace that selection without opening the
    missing file or reporting a mutually-exclusive-options error. This checks
    that CLI precedence applies across the entire instrument-source group.
    """
    module, calls = downloader
    config = tmp_path / "download.conf"
    config.write_text('pairs-file = "missing.json"\n')

    with pytest.raises(SystemExit) as result:
        module.run_main(["--config", str(config), "--instruments", "ETH-BTC"])

    assert result.value.code == 0
    assert [pair.name for pair in calls["currency_pairs"]] == ["ETH-BTC"]


@pytest.mark.parametrize("payload", [{}, {"currency_pairs": "ETH-BTC"}, [], ["ETHBTC"], [42]])
def test_invalid_pair_files_fail_before_creating_crawler(tmp_path, downloader, payload):
    """Reject structurally invalid instrument files before any download starts.

    Dummy inputs cover a missing currency_pairs key, a string instead of a
    list, an empty list, a symbol without BASE-QUOTE separators, and a numeric
    list member. Each must produce the ordinary argparse exit code 2 and
    leave the fake crawler untouched, avoiding tracebacks or accidental broad
    exchange-wide downloads after malformed configuration.
    """
    module, calls = downloader
    pairs_file = tmp_path / "invalid.json"
    pairs_file.write_text(json.dumps(payload))

    with pytest.raises(SystemExit) as result:
        module.run_main(["--pairs-file", str(pairs_file)])

    assert result.value.code == 2
    assert calls == {}


@pytest.mark.parametrize("end", ["2021-06-15", "2021-06-16"])
def test_invalid_date_interval_fails_before_selecting_instruments(downloader, end):
    """Empty and inverted date intervals must stop before network work.

    The start is fixed at June 16; the exclusive end is either June 15 or
    June 16. Neither interval contains any day. Both commands must raise an
    argparse error before creating a crawler, and the trades fixture also
    fails if it attempts the default live exchange lookup before validation.
    """
    module, calls = downloader
    with pytest.raises(SystemExit) as result:
        module.run_main(["--start", "2021-06-16", "--end", end])
    assert result.value.code == 2
    assert calls == {}


def test_trade_configuration_preserves_btc_quote_restriction(tmp_path, monkeypatch):
    """The new instrument flag must retain the trade parser's BTC universe.

    BTC-USDT is syntactically valid but lies outside this project's BTC-quoted
    trade dataset. Supplying it through JSON must yield exit code 2 before a
    crawler is constructed. This prevents a new configuration route from
    bypassing the same restriction applied to historical pair manifests.
    """
    from pumps_and_dumps.market_data.parsers.binance import spot_trades

    config = tmp_path / "download.json"
    config.write_text(json.dumps({"instruments": ["BTC-USDT"]}))
    monkeypatch.setattr(spot_trades, "CrawlerProcess", lambda **_: pytest.fail("Unexpected crawler"))
    with pytest.raises(SystemExit) as result:
        spot_trades.run_main(["--config", str(config)])
    assert result.value.code == 2
