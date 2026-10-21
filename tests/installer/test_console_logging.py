"""Check streamed console output and persistent logs with tiny real child processes.

The children use only Python's standard library, run in temporary directories
and never invoke Docker or access the network. Their literal output and exit
codes make console/file behavior easy to verify by inspection.
"""

import os
import re
import sys

import pytest

from installer import cli
from installer.state import Stage
from pumps_and_dumps.core.paths import DATA_DIR_ENV


def test_console_streams_stdout_stderr_and_final_partial_line(tmp_path, capsys):
    """Both child streams reach the terminal and a timestamped .log file.

    The child emits one stdout line, one stderr line, a 3,000-character line and
    a final line with no newline. Every message must appear untruncated in both
    destinations, including the final buffered fragment. INFO records prove
    the file is written through the shared standard-logging handler rather
    than by redirecting raw bytes into an arbitrarily named output file.
    """
    long_line = "x" * 3000
    source = (
        "import sys; print('from stdout', flush=True); "
        "print('from stderr', file=sys.stderr, flush=True); "
        f"print({long_line!r}, flush=True); sys.stdout.write('final fragment')"
    )
    stage = Stage("Sample stage", [sys.executable, "-c", source], key="features")
    data = tmp_path / "data"
    environment = {**os.environ, DATA_DIR_ENV: str(data)}
    assert cli._run_console([stage], tmp_path, environment, False) == 0
    assert stage.status == "done"
    logs = list((data / "logs").glob("install-*/*.log"))
    assert len(logs) == 1
    assert logs[0].name == "01-features.log"
    recorded = logs[0].read_text()
    displayed = capsys.readouterr().out
    after_command = displayed.split("+ ", 1)[1].split("\n", 1)[1]
    for message in ("from stdout", "from stderr", long_line, "final fragment"):
        assert message in recorded
        assert message in after_command
    assert all(re.match(r"\d{4}-\d{2}-\d{2} .* INFO ", line) for line in recorded.splitlines())
    assert str(logs[0].parent) in displayed


def test_console_preserves_progress_records_in_log_but_prints_summary(tmp_path, capsys):
    """Structured progress stays machine-readable on disk and readable in the console.

    A literal two-of-three record emulates a container that exports progress on
    stdout. It must update the Stage, remain present in its .log file, and be
    replaced by an ordinary progress summary in terminal output. The command is
    printed by the console before execution, so only output after its command
    line is checked for absence of the raw JSON payload.
    """
    payload = '{"task":"overall","completed":2,"total":3}'
    stage = Stage(
        "Progress stage",
        [sys.executable, "-c", f"print('ordinary line'); print({payload!r})"],
        progress_stdout=True,
    )
    environment = {**os.environ, DATA_DIR_ENV: str(tmp_path)}
    assert cli._run_console([stage], tmp_path, environment, False) == 0
    assert stage.progress == {"task": "overall", "completed": 2, "total": 3}
    recorded = next((tmp_path / "logs").glob("install-*/*.log")).read_text()
    assert payload in recorded
    displayed = capsys.readouterr().out
    after_command = displayed.split("ordinary line\n", 1)[1]
    assert payload not in after_command
    assert "2 / 3 items" in after_command


def test_repeated_console_runs_keep_separate_log_files(tmp_path):
    """Two runs in the same host process preserve separate stage log histories.

    Each invocation uses the same stage index and key but emits a distinct
    literal message. Both records must survive under separate installation
    directories, with no append mixing or overwriting between repeated runs.
    """
    environment = {**os.environ, DATA_DIR_ENV: str(tmp_path)}
    for message in ("first invocation", "second invocation"):
        stage = Stage("Repeated stage", [sys.executable, "-c", f"print({message!r})"], key="trades")
        assert cli._run_console([stage], tmp_path, environment, False) == 0
    logs = list((tmp_path / "logs").glob("install-*/*.log"))
    assert len(logs) == 2
    assert len({path.parent for path in logs}) == 2
    records = [path.read_text() for path in logs]
    assert sum("first invocation" in record for record in records) == 1
    assert sum("second invocation" in record for record in records) == 1


def test_console_signal_exit_keeps_shell_status_and_stops_pipeline(tmp_path):
    """A child terminated by SIGTERM returns 143 and never starts a later stage.

    The first child logs a message then terminates itself with signal 15. The
    second would create a sentinel file if launched. Absence of that sentinel
    and the pending state of its Stage prove the failure stops the pipeline,
    while the first stage's .log still contains its pre-failure output.
    """
    first = Stage(
        "Terminated stage",
        [
            sys.executable,
            "-c",
            "import os, signal; print('before signal', flush=True); os.kill(os.getpid(), signal.SIGTERM)",
        ],
    )
    sentinel = tmp_path / "later-stage"
    second = Stage("Unstarted stage", [sys.executable, "-c", f"open({str(sentinel)!r}, 'w').close()"])
    environment = {**os.environ, DATA_DIR_ENV: str(tmp_path)}
    assert cli._run_console([first, second], tmp_path, environment, False) == 143
    assert first.status == "failed"
    assert second.status == "pending"
    assert not sentinel.exists()
    logs = list((tmp_path / "logs").glob("install-*/*.log"))
    assert len(logs) == 1
    assert "before signal" in logs[0].read_text()


def test_console_dry_run_creates_no_logs_or_processes(tmp_path, monkeypatch):
    """A preview must not open a stage log or launch its process monitor.

    A missing data path and failing monitor stub expose both possible side
    effects. Printing a dry-run command must succeed while leaving the entire
    prospective data directory, including its logs subdirectory, absent.
    """

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry run started the process monitor")

    monkeypatch.setattr(cli, "monitor_stage", forbidden)
    data = tmp_path / "not-created"
    stage = Stage("Preview", ["unused-command"])
    assert cli._run_console([stage], tmp_path, {DATA_DIR_ENV: str(data)}, True) == 0
    assert not data.exists()
