"""Hand-checkable examples for file configuration and command-line precedence."""

import argparse
from datetime import date
from pathlib import Path

import pytest

from pumps_and_dumps.core.cli_config import parse_config_args


def stage_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", "--start-date", type=date.fromisoformat)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval", choices=["1m", "1h"], default="1m")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--pairs", "--instruments", nargs="+")
    sources.add_argument("--pairs-file", type=Path)
    return parser


def parse(argv):
    return parse_config_args(stage_parser(), argv, path_keys=("output_dir", "pairs_file"), object_keys=("pumps",))


def test_hocon_values_have_cli_types_and_relative_paths(tmp_path):
    """An annotated file produces dates, integers, booleans, lists and absolute paths.

    The relative output directory must resolve beside the config, independent of
    pytest's working directory. Two inline pump records remain structured data
    for the feature CLI to validate, rather than becoming shell arguments.
    """
    config = tmp_path / "job.conf"
    config.write_text(
        '# one small job\nstart = "2021-01-01"\nprocesses = 2\n'
        'output-dir = "out data"\ninstruments = ["ETH-BTC", "ADA-BTC"]\n'
        'skip-existing = true\npumps = [{symbol: "ETH-BTC"}, {symbol: "ADA-BTC"}]\n'
    )
    args = parse(["--config", str(config)])
    assert args.start == date(2021, 1, 1)
    assert args.processes == 2
    assert args.skip_existing is True
    assert args.output_dir == tmp_path / "out data"
    assert args.pairs == ["ETH-BTC", "ADA-BTC"]
    assert [pump["symbol"] for pump in args.pumps] == ["ETH-BTC", "ADA-BTC"]


def test_explicit_flags_override_file_values_and_mutually_exclusive_source(tmp_path):
    """CLI flags replace configured dates, booleans, workers and instrument source.

    The file requests two inline pairs and caching. Explicit `--pairs-file` must
    clear those inline pairs, and `--no-skip-existing` must turn caching off.
    The CLI's relative path remains relative to the process working directory.
    """
    config = tmp_path / "job.json"
    config.write_text('{"start":"2021-01-01","processes":2,"pairs":["ETH-BTC"],"skip_existing":true}')
    args = parse(
        [
            "--config",
            str(config),
            "--start-date",
            "2022-01-01",
            "--processes=3",
            "--no-skip-existing",
            "--pairs-file",
            "selected.json",
        ]
    )
    assert args.start == date(2022, 1, 1)
    assert args.processes == 3
    assert args.skip_existing is False
    assert args.pairs is None
    assert args.pairs_file == Path("selected.json")


@pytest.mark.parametrize(
    "contents,reason",
    [
        ("unknown-option = 1", "unknown config option"),
        ("processes = true", "expected a scalar"),
        ("processes = 1.5", "invalid config option"),
        ('skip-existing = "false"', "expected true or false"),
        ('pairs = "ETH-BTC"', "expected a list"),
        ("pairs = []", "nonempty list"),
        ('interval = "banana"', "expected one of"),
        ('start = "bad date"', "invalid config option"),
        ('start = "2021-01-01"\nstart-date = "2021-01-02"', "multiple aliases"),
        ('pairs = ["ETH-BTC"]\npairs-file = "pairs.json"', "conflicting config"),
        ("processes = {", "cannot read config"),
        ('[{"processes": 1}]', "config must be an object"),
    ],
)
def test_bad_config_stops_with_actionable_error(tmp_path, capsys, contents, reason):
    """Malformed or ambiguous configuration fails with argparse status 2 before work.

    Each tiny fixture isolates one mistake: misspelled option, wrong type, empty
    instrument list, invalid choice/date, duplicate alias, conflicting sources,
    malformed HOCON, or a list at the document root. Matching the message checks
    that the user receives the cause rather than a later processing traceback.
    """
    config = tmp_path / "bad.conf"
    config.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        parse(["--config", str(config)])
    assert exc.value.code == 2
    assert reason in capsys.readouterr().err


def test_hocon_local_include_and_substitution(tmp_path):
    """HOCON includes and substitutions provide actual HOCON behavior beyond JSON.

    An adjacent file supplies two workers and the main file uses that value for
    the same option via substitution. The resulting integer proves includes are
    resolved relative to the configuration file, even when invoked elsewhere.
    """
    (tmp_path / "defaults.conf").write_text("processes = 2\n")
    config = tmp_path / "job.conf"
    config.write_text('include "defaults.conf"\nprocesses = ${processes}\n')
    assert parse(["--config", str(config)]).processes == 2


def test_help_does_not_require_config_file(tmp_path, capsys):
    """Help works even when a mounted configuration path is missing.

    Invoking `--help` with a nonexistent file should print the documented config
    option and exit zero. It must not attempt to read the file or begin work.
    """
    with pytest.raises(SystemExit) as exc:
        parse(["--config", str(tmp_path / "missing.conf"), "--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out
