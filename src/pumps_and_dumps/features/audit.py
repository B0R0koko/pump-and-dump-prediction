"""Append complete JSON audit records safely from independent feature workers."""

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


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
        destination = path if path is not None else feature_dir / "logs" / f"feature_audit_{run_id}.jsonl"
        return cls(Path(destination).expanduser().resolve(), run_id)

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        logged_at = datetime.now(timezone.utc).isoformat()
        payload = "".join(
            json.dumps(
                {"run_id": self.run_id, "logged_at": logged_at, "pid": os.getpid(), **record},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for record in records
        )
        if not payload:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.write(payload)
                stream.flush()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
