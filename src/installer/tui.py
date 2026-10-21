"""Terminal dashboard for installer stages."""

from collections import deque
from datetime import datetime
import os
from pathlib import Path
import sys
import shutil
import time

from installer.dashboard import render
from installer.ui import Theme, prepare_screen
from installer.state import Stage
from installer.runner import StageRunner, monitor_stage


def tui_available() -> bool:
    if not sys.stdin.isatty() or not sys.stdout.isatty() or os.environ.get("TERM", "dumb") == "dumb":
        return False
    try:
        import curses
    except ImportError:
        return False
    try:
        curses.setupterm()
        return True
    except curses.error:
        return False


class Dashboard:
    def __init__(
        self,
        stages: list[Stage],
        root: Path,
        environment: dict[str, str],
        data_dir: Path,
        dates: str,
        runner: StageRunner | None = None,
    ):
        self.stages = stages
        self.root = root
        self.environment = environment
        self.data_dir = data_dir
        self.dates = dates
        self.runner = monitor_stage if runner is None else runner
        self.started = time.monotonic()
        self.ended = 0.0
        self.logs: deque[str] = deque(maxlen=2000)
        self.scroll = 0
        self.finished = False
        self.log_dir = data_dir / "logs" / f"install-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
        self._disk_checked = 0.0
        self.free_space = ""
        self.theme = Theme()

    def draw(self, screen) -> None:
        now = time.monotonic()
        if now - self._disk_checked > 5:
            self.free_space = f"{shutil.disk_usage(self.data_dir).free / 1024**3:,.1f} GiB"
            self._disk_checked = now
        render(self, screen)

    def tick(self, screen) -> bool:
        import curses

        key = screen.getch()
        if key in (ord("q"), 3) or (self.finished and key in (10, 13)):
            if not self.finished:
                raise KeyboardInterrupt
            return False
        if key in (curses.KEY_UP, curses.KEY_PPAGE):
            self.scroll = min(
                max(0, len(self.logs) - 1),
                self.scroll + (1 if key == curses.KEY_UP else 10),
            )
        elif key in (curses.KEY_DOWN, curses.KEY_NPAGE):
            self.scroll = max(0, self.scroll - (1 if key == curses.KEY_DOWN else 10))
        elif key == curses.KEY_END:
            self.scroll = 0
        self.draw(screen)
        return True

    def run(self, screen) -> int:
        self.theme = prepare_screen(screen)
        self.log_dir.mkdir(parents=True)
        code = 0
        try:
            for index, stage in enumerate(self.stages, start=1):
                self.logs.append(f"Starting: {stage.name}")
                code = self.runner(
                    stage,
                    root=self.root,
                    environment=self.environment,
                    log_path=self.log_dir / f"{index:02}.log",
                    logs=self.logs,
                    tick=lambda: self.tick(screen),
                )
                if code:
                    self.logs.append(f"Failed: {stage.name} (exit {code}). Later stages were not started.")
                    break
                self.logs.append(f"Finished: {stage.name}")
        except KeyboardInterrupt:
            self.logs.append("Installation cancelled. Downloaded ZIPs can be reused on the next run.")
            code = 130
        self.finished = True
        self.ended = time.monotonic()
        for stage in self.stages:
            if stage.status == "pending":
                stage.status = "not run"
        while self.tick(screen):
            time.sleep(0.1)
        return code if code >= 0 else 128 - code


def run_tui(stages: list[Stage], **kwargs) -> int:
    import curses

    dashboard = Dashboard(stages, **kwargs)
    try:
        return curses.wrapper(dashboard.run)
    finally:
        print(f"Installer logs: {dashboard.log_dir}")
