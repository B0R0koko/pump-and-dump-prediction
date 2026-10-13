"""Progress snapshots for the installer's pipe, separate from ordinary logs."""

import json
import os
import select
import time
from typing import Any

PROGRESS_FD_ENV = "PUMPS_AND_DUMPS_PROGRESS_FD"


def progress_enabled() -> bool:
    return PROGRESS_FD_ENV in os.environ


class ProgressReporter:
    def __init__(
        self,
        task: str = "overall",
        *,
        total: int | None = None,
        unit: str = "items",
        detail: str = "",
    ):
        self.fields: dict[str, Any] = dict(
            task=task,
            completed=0,
            total=total,
            unit=unit,
            detail=detail,
            status="running",
        )
        self._last_sent = 0.0
        self.update(force=True)

    def update(self, advance: int = 0, *, force: bool = False, **fields) -> None:
        self.fields["completed"] += advance
        self.fields.update(fields)
        now = time.monotonic()
        if not progress_enabled() or (not force and now - self._last_sent < 0.2):
            return
        self._last_sent = now
        # Stay below PIPE_BUF so independent forked workers cannot interleave records.
        snapshot = {**self.fields, "detail": str(self.fields["detail"])[:240]}
        record = (json.dumps(snapshot, ensure_ascii=True) + "\n").encode()
        if len(record) > 4096:
            return
        try:
            fd = int(os.environ[PROGRESS_FD_ENV])
            # Give the final aggregate snapshot a chance to drain after a burst of worker updates.
            if self.fields["task"] == "overall" and self.fields["status"] in (
                "done",
                "failed",
            ):
                select.select([], [fd], [], 0.2)
            os.write(fd, record)
        except (OSError, ValueError):
            # A closed/full monitoring pipe must not break data processing.
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self.update(status="done" if exc_type is None else "failed", force=True)


def worker_progress(
    *, unit: str, detail: str, total: int | None = None
) -> ProgressReporter:
    return ProgressReporter(
        f"worker:{os.getpid()}", total=total, unit=unit, detail=detail
    )
