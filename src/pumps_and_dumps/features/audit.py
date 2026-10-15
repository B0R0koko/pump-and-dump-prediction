"""Append complete JSON audit records safely from independent feature workers."""

import fcntl
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


class _LockedFileHandler(logging.FileHandler):
    """Keep each multi-record batch together across processes and surface errors."""

    def emit(self, record: logging.LogRecord) -> None:
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        try:
            super().emit(record)
        finally:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)

    def handleError(self, record: logging.LogRecord) -> None:
        raise


@dataclass(frozen=True)
class FeatureAuditLog:
    """A picklable destination; each batch is locked, appended and flushed.

    File locking prevents interleaved JSON when spawned workers (or independent
    runs explicitly sharing a path) append concurrently. No open handle crosses
    a process boundary, and write failures propagate to the feature writer.
    """

    path: Path
    run_id: str

    @classmethod
    def for_run(cls, feature_dir: Path, path: Path | None = None) -> "FeatureAuditLog":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid4().hex[:8]
        destination = path if path is not None else feature_dir / "logs" / f"feature_audit_{run_id}.log"
        return cls(Path(destination).expanduser().with_suffix(".log").resolve(), run_id)

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        logged_at = datetime.now(timezone.utc).isoformat()
        payload = "\n".join(
            json.dumps(
                {"run_id": self.run_id, "logged_at": logged_at, "pid": os.getpid(), **record},
                ensure_ascii=False,
                allow_nan=False,
            )
            for record in records
        )
        if not payload:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handler = _LockedFileHandler(self.path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.Logger(f"feature-audit.{self.run_id}", level=logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)
        try:
            logger.info("%s", payload)
        finally:
            logger.removeHandler(handler)
            handler.close()
