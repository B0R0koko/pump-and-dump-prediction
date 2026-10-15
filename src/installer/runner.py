"""Headless subprocess supervision and progress/log transport for installer stages."""

from collections import deque
from contextlib import contextmanager
import os
import json
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time
from typing import BinaryIO, Callable, Protocol

from pumps_and_dumps.core.progress import PROGRESS_FD_ENV
from installer.state import Stage
from installer.containers import cleanup_container

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_line(line: str) -> str:
    return "".join(char for char in _ANSI.sub("", line) if char.isprintable() or char == "\t")


class StageRunner(Protocol):
    def __call__(
        self,
        stage: Stage,
        *,
        root: Path,
        environment: dict[str, str],
        log_path: Path,
        logs: deque,
        tick: Callable[[], object],
    ) -> int: ...


def stop_process_group(process: subprocess.Popen) -> None:
    """Stop the stage and its multiprocessing workers, including after a parent crash."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class _StageOutput:
    """Decode the two child streams and retain bounded log/progress buffers."""

    def __init__(self, stage: Stage, logs: deque, log_file: BinaryIO):
        self.stage = stage
        self.logs = logs
        self.log_file = log_file
        self.buffers: dict[str, bytes] = {}

    def feed(self, channel: str, chunk: bytes) -> None:
        if channel == "log" and chunk:
            self.log_file.write(chunk)
            self.log_file.flush()
        buffer = self.buffers.get(channel, b"") + chunk
        lines = buffer.split(b"\n")
        remainder = lines.pop()
        if not chunk and remainder:
            lines.append(remainder)
            remainder = b""
        self.buffers[channel] = remainder[-16384:]
        for line in lines:
            if channel == "progress":
                self.stage.receive(line)
            elif self.stage.progress_stdout and self._is_progress(line):
                self.stage.receive(line)
            else:
                self.logs.append(clean_line(line.decode("utf-8", errors="replace"))[-2000:])

    @staticmethod
    def _is_progress(line: bytes) -> bool:
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            return False
        return isinstance(event, dict) and isinstance(event.get("task"), str) and "completed" in event


@contextmanager
def _stage_process(command: list[str], root: Path, environment: dict[str, str]):
    """Own the child, its process group, and both ends of the progress pipe."""
    progress_read, progress_write = os.pipe()
    process = None
    try:
        try:
            os.set_blocking(progress_write, False)
            process = subprocess.Popen(
                command,
                cwd=root,
                env={
                    **environment,
                    PROGRESS_FD_ENV: str(progress_write),
                    "PYTHONUNBUFFERED": "1",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(progress_write,),
            )
        finally:
            os.close(progress_write)
        yield process, progress_read
    finally:
        if process is not None:
            stop_process_group(process)
            if process.stdout is not None:
                process.stdout.close()
        os.close(progress_read)


def _drain_output(
    process: subprocess.Popen,
    progress_read: int,
    output: _StageOutput,
    tick: Callable[[], object],
):
    if process.stdout is None:
        raise RuntimeError("Stage stdout pipe was not created")
    exited_at = None
    next_draw = 0.0
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, "log")
        selector.register(progress_read, selectors.EVENT_READ, "progress")
        while selector.get_map() or process.poll() is None:
            for key, _ in selector.select(timeout=0.1):
                chunk = os.read(key.fd, 65536)
                output.feed(key.data, chunk)
                if not chunk:
                    selector.unregister(key.fileobj)
            now = time.monotonic()
            if now >= next_draw:
                tick()
                next_draw = now + 0.1
            if process.poll() is None:
                continue
            exited_at = exited_at or now
            # Orphaned workers must not hold a finished stage's pipes open forever.
            if now - exited_at > 2:
                break


def monitor_stage(
    stage: Stage,
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
    logs: deque,
    tick: Callable[[], object],
) -> int:
    """Run one stage while independently draining its logs and progress."""
    stage.status = "running"
    stage.started = stage.last_update = time.monotonic()
    try:
        with (
            log_path.open("wb") as log_file,
            _stage_process(stage.command, root, environment) as (
                process,
                progress_read,
            ),
        ):
            stage.pid = process.pid
            _drain_output(process, progress_read, _StageOutput(stage, logs, log_file), tick)
            code = process.wait()
        stage.status = "done" if code == 0 else "failed"
        return code
    except KeyboardInterrupt:
        stage.status = "cancelled"
        raise
    except BaseException:
        stage.status = "failed"
        raise
    finally:
        cleanup_container(stage)
        stage.ended = time.monotonic()
