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

from pumps_and_dumps.features.provenance import timestamp_ns, utc_timestamp


@dataclass
class AssetInputAudit:
    """Reduce calculation provenance to one temporal check for an asset.

    Predictor inputs include their return-price lags and normalizer history;
    target inputs include the baseline price used by the target return. Keep
    integer nanoseconds until formatting, so equal endpoints count as overlap.
    Missing dependency timestamps cannot produce a successful certification,
    but known timestamps still establish a violation when provenance is partial.
    """

    feature_max_ns: int | None = None
    target_min_ns: int | None = None
    feature_provenance_complete: bool = True
    target_provenance_complete: bool = True

    def observe(self, record: dict[str, Any]) -> None:
        """Consume a calculation callback without retaining per-feature records."""
        if record["role"] == "predictor":
            end = record["data_end_ns"]
            if end is not None:
                self.feature_max_ns = end if self.feature_max_ns is None else max(self.feature_max_ns, end)
            self.feature_provenance_complete &= record["provenance_complete"]
        elif record["role"] == "outcome":
            start = record["data_start_ns"]
            if start is not None:
                self.target_min_ns = start if self.target_min_ns is None else min(self.target_min_ns, start)
            self.target_provenance_complete &= record["provenance_complete"]
        else:
            raise ValueError(f"Unknown audit role: {record['role']!r}")

    def summary(self, cutoff: datetime) -> dict[str, Any]:
        """Return actual bounds and conservative checks for one pump/asset pair."""
        complete = self.feature_provenance_complete and self.target_provenance_complete
        before_cutoff = None
        if self.feature_max_ns is not None:
            if self.feature_max_ns >= timestamp_ns(cutoff):
                before_cutoff = False
            elif self.feature_provenance_complete:
                before_cutoff = True

        non_overlapping = None
        if self.feature_max_ns is not None and self.target_min_ns is not None:
            if self.feature_max_ns >= self.target_min_ns:
                non_overlapping = False
            elif complete:
                non_overlapping = True

        if before_cutoff is False:
            status = "LEAKAGE"
        elif non_overlapping is False:
            status = "OVERLAP"
        elif not complete:
            status = "UNKNOWN"
        elif self.feature_max_ns is None or self.target_min_ns is None:
            status = "NO_DATA"
        else:
            status = "OK"
        return {
            "feature_max_time": utc_timestamp(self.feature_max_ns),
            "target_min_time": utc_timestamp(self.target_min_ns),
            "non_overlapping": non_overlapping,
            "features_before_cutoff": before_cutoff,
            "provenance_complete": complete,
            "status": status,
        }


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
