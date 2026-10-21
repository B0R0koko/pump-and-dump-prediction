"""Tiny subprocess outputs verify persistent logs without Docker or network access."""

from collections import deque
import logging
import os
import re
import sys

import pytest

from installer.logging_utils import file_logger, log_path
from installer.runner import _StageOutput, monitor_stage
from installer.state import Stage


def test_file_logger_normalizes_suffix_flushes_and_closes(tmp_path):
    """Every installer log is a .log file managed by a standard FileHandler.

    Two separate sessions append one message each to a requested .txt path.
    Inspect the first message before closing to verify immediate flushing, and
    inspect both saved handlers afterward to verify resource release. Exactly
    two records prove repeated use does not retain and duplicate handlers.
    """
    requested = tmp_path / "logs" / "setup.txt"
    destination = log_path(requested)
    assert destination == tmp_path / "logs" / "setup.log"
    handlers = []
    for message in ("install started", "install completed"):
        with file_logger(requested, mode="a") as logger:
            assert logger.propagate is False
            assert isinstance(logger.handlers[0], logging.FileHandler)
            handlers.append(logger.handlers[0])
            logger.info("%s", message)
            assert message in destination.read_text()
        assert not logger.handlers
    lines = destination.read_text().splitlines()
    assert len(lines) == 2
    assert all(re.match(r"\d{4}-\d{2}-\d{2} .* INFO install ", line) for line in lines)
    assert all(handler.stream is None for handler in handlers)
    assert not requested.exists()


def test_file_logger_propagates_write_errors_and_closes(tmp_path, monkeypatch):
    """A full or failed log destination cannot silently produce an unaudited run.

    Replace only the real file's write method with an OSError. Standard logging
    normally swallows that failure; our FileHandler must propagate it while the
    context manager still closes the underlying file and removes its handler.
    """
    with pytest.raises(OSError, match="disk full"):
        with file_logger(tmp_path / "failure.log") as logger:
            handler = logger.handlers[0]

            def failed_write(_message):
                raise OSError("disk full")

            monkeypatch.setattr(handler.stream, "write", failed_write)
            logger.info("must be saved")
    assert handler.stream is None
    assert not logger.handlers


def test_output_retains_long_and_unterminated_lines(tmp_path):
    """Chunk boundaries and a missing final newline do not discard stage output.

    A long diagnostic exceeds the dashboard's 16 KiB partial-line buffer. Its
    complete text must still reach the log, followed by a split UTF-8 character
    and an unterminated final message. UI entries remain bounded to 2,000
    characters even though the file retains the long diagnostic.
    """
    destination = tmp_path / "output.log"
    stage = Stage("Dummy", [])
    logs = deque()
    diagnostic = "x" * 20000
    with file_logger(destination) as logger:
        output = _StageOutput(stage, logs, logger)
        output.feed("log", diagnostic.encode())
        output.feed("log", b"\nprice \xe2")
        output.feed("log", b"\x82\xac")
        output.flush()
    saved = destination.read_text()
    assert diagnostic in saved
    assert "price €" in saved
    assert all(len(line) <= 2000 for line in logs)
    assert logs[-1] == "price €"


def test_monitor_persists_stdout_stderr_and_final_partial_line(tmp_path):
    """A real child process leaves readable .log records and its original exit code.

    The subprocess writes one stdout line, one stderr line and a final stdout
    fragment without a newline, then exits with status seven. Both streams
    must be captured, the final fragment flushed, and the failed stage status
    preserved independently of the logging changes.
    """
    stage = Stage(
        "Dummy",
        [
            sys.executable,
            "-c",
            "import sys; print('stdout'); print('stderr', file=sys.stderr); "
            "sys.stdout.write('last fragment'); sys.exit(7)",
        ],
    )
    result = monitor_stage(
        stage,
        root=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "child.txt",
        logs=deque(),
        tick=lambda: None,
    )
    assert result == 7
    assert stage.status == "failed"
    lines = (tmp_path / "child.log").read_text().splitlines()
    assert len(lines) == 3
    assert {line.split(" INFO ", 1)[1] for line in lines} == {"stdout", "stderr", "last fragment"}
    assert not (tmp_path / "child.txt").exists()
