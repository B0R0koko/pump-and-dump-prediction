"""Execution state shared by console, process supervision, and terminal views."""

from dataclasses import dataclass, field
import json
import time

from installer.plan import STAGES


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


@dataclass
class Stage:
    name: str
    command: list[str]
    status: str = "pending"
    progress: dict = field(default_factory=dict)
    workers: dict = field(default_factory=dict)
    started: float = 0.0
    ended: float = 0.0
    last_update: float = 0.0
    pid: int | None = None
    key: str = ""

    @property
    def title(self) -> str:
        return next(
            (option.title for option in STAGES if option.key == self.key), self.name
        )

    def receive(self, record: bytes) -> None:
        try:
            event = json.loads(record)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict) or not isinstance(event.get("task"), str):
            return
        self.last_update = time.monotonic()
        if event["task"] == "overall":
            self.progress = event
        elif event.get("status") == "running":
            self.workers[event["task"]] = event
        else:
            self.workers.pop(event["task"], None)

    def elapsed(self) -> float:
        return (self.ended or time.monotonic()) - self.started if self.started else 0.0

    def summary(self) -> str:
        completed = self.progress.get("completed", 0)
        total = self.progress.get("total")
        unit = self.progress.get("unit", "items")
        known = total is not None and self.progress.get("total_known", True)
        count = (
            f"{completed:,} / {total:,} {unit}"
            if total is not None
            else f"{completed:,} {unit}"
        )
        if not known:
            return f"{count} | discovering total | {duration(self.elapsed())}"
        rate = completed / max(self.elapsed(), 0.001)
        eta = (
            duration((total - completed) / rate)
            if rate > 0 and total >= completed
            else "--:--:--"
        )
        return (
            f"{count} | {rate:,.1f}/s | elapsed {duration(self.elapsed())} | ETA {eta}"
        )
