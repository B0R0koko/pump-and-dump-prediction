"""Interactive choice of research environment setup or data pipeline stages."""

from dataclasses import dataclass
from pathlib import Path
import time
import textwrap

from installer.plan import STAGES, STAGE_KEYS, ordered_stages, validate_inputs
from installer.ui import Canvas, Theme, prepare_screen


@dataclass(frozen=True)
class Selection:
    data_dir: Path
    stages: list[str]
    setup_environment: bool = False


class SetupMenu:
    def __init__(self, data_dir: Path, dates: str, *, configured_stages=()):
        self.data_dir = data_dir
        self.dates = dates
        self.configured_stages = configured_stages
        self.custom = False
        self.focus = 0
        self.selected = set(STAGE_KEYS)
        self.editing = False
        self.path_text = str(data_dir)
        self.cursor = len(self.path_text)
        self.error = ""
        self.cancelled = False
        self.theme = Theme()

    def submit(self) -> Selection | None:
        selected = ordered_stages(self.selected) if self.custom else list(STAGE_KEYS)
        try:
            error = validate_inputs(selected, self.data_dir, skip_stages=self.configured_stages)
        except OSError as exc:
            error = str(exc)
        if error:
            self.error = error
            return None
        return Selection(self.data_dir, selected)

    def handle(self, key: str | int) -> Selection | None:
        import curses

        if self.editing:
            self._edit_directory(key)
            return None

        if key in ("q", "\x03"):
            self.cancelled = True
        elif key == "\x1b":
            if self.custom:
                self.custom, self.focus = False, 1
            else:
                self.cancelled = True
        elif key == "d":
            self.editing = True
            self.path_text = str(self.data_dir)
            self.cursor = len(self.path_text)
        elif key in (curses.KEY_UP, curses.KEY_DOWN, "\t", "j", "k"):
            direction = -1 if key in (curses.KEY_UP, "k") else 1
            self.focus = (self.focus + direction) % (len(STAGES) + 1 if self.custom else 3)
        elif self.custom and key in ("a", "n"):
            self.selected = set(STAGE_KEYS) if key == "a" else set()
            self.error = ""
        elif self.custom and key == "s":
            return self.submit()
        elif key in ("\n", "\r", " "):
            return self._activate_selection()
        return None

    def _activate_selection(self) -> Selection | None:
        if not self.custom:
            if self.focus == 0:
                return self.submit()
            if self.focus == 2:
                return Selection(self.data_dir, [], setup_environment=True)
            self.custom, self.focus = True, 0
            return None
        if self.focus == len(STAGES):
            return self.submit()
        chosen = STAGES[self.focus].key
        self.selected.symmetric_difference_update({chosen})
        self.error = ""
        return None

    def _edit_directory(self, key: str | int):
        import curses

        if key == "\x1b":
            self.editing = False
        elif key in ("\n", "\r"):
            self._save_directory()
        elif key in (curses.KEY_BACKSPACE, "\x7f", "\b") and self.cursor:
            self.path_text = self.path_text[: self.cursor - 1] + self.path_text[self.cursor :]
            self.cursor -= 1
        elif key == curses.KEY_DC:
            self.path_text = self.path_text[: self.cursor] + self.path_text[self.cursor + 1 :]
        elif key == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor = min(len(self.path_text), self.cursor + 1)
        elif key == curses.KEY_HOME:
            self.cursor = 0
        elif key == curses.KEY_END:
            self.cursor = len(self.path_text)
        elif key == "\x15":
            self.path_text = ""
            self.cursor = 0
        elif isinstance(key, str) and key.isprintable():
            self.path_text = self.path_text[: self.cursor] + key + self.path_text[self.cursor :]
            self.cursor += len(key)

    def _save_directory(self):
        if not self.path_text.strip():
            self.error = "Enter a data directory."
            return
        try:
            self.data_dir = Path(self.path_text).expanduser().resolve()
        except (ValueError, OSError, RuntimeError) as exc:
            self.error = str(exc)
            return
        self.editing = False
        self.error = ""

    def draw(self, screen):
        screen.erase()
        view = Canvas(screen, self.theme)
        view.header("Choose the work to run. Your existing dataset can be reused.")
        if view.too_small():
            screen.refresh()
            return
        width = min(100, view.width - 4)
        x = (view.width - width) // 2
        self._draw_directory(view, x, width)
        if self.custom:
            self._draw_stages(view, x, width)
            footer = "↑↓ Move  Space Toggle  s Run  a All  n None  d Directory  Esc Back"
        else:
            self._draw_installation_choices(view, x, width)
            footer = "↑↓ Navigate   Enter Select   d Directory   q Quit"
        if self.error:
            for offset, line in enumerate(textwrap.wrap(self.error, width - 4)[:2]):
                view.text(view.height - 3 + offset, x + 2, line, "danger", width=width - 4)
        view.text(view.height - 1, 2, footer, "accent", width=view.width - 4)
        screen.refresh()

    def _draw_directory(self, view: Canvas, x: int, width: int):
        view.panel(4, x, 4, width, "DATA DIRECTORY", focused=self.editing)
        if not self.editing:
            view.text(5, x + 2, str(self.data_dir), width=width - 4)
            view.text(
                6,
                x + 2,
                f"{self.dates}   |   d  Change directory",
                "muted",
                width=width - 4,
            )
            return
        offset = max(0, self.cursor - (width - 8))
        view.text(5, x + 2, self.path_text[offset:], width=width - 4)
        view.text(
            5,
            x + 2 + self.cursor - offset,
            self.path_text[self.cursor : self.cursor + 1] or " ",
            "selected",
            width=1,
        )
        view.text(6, x + 2, "Enter save   Esc cancel   Ctrl-U clear", "accent")

    def _draw_installation_choices(self, view: Canvas, x: int, width: int):
        for index, (title, detail) in enumerate(
            (
                (
                    "Full installation",
                    "Trade archives → BTC candles → Hive Parquet → event features",
                ),
                (
                    "Choose stages",
                    "Download, convert, or compute features independently.",
                ),
                (
                    "Set up research environment",
                    "Update code and install .venv dependencies; no Docker needed.",
                ),
            )
        ):
            y = 9 + index * 4
            active = self.focus == index and not self.editing
            view.panel(y, x, 4, width, f"0{index + 1}", focused=active)
            view.text(
                y + 1,
                x + 2,
                f"  {title}  ",
                "selected" if active else "text",
                bold=True,
            )
            view.text(y + 2, x + 2, detail, "muted", width=width - 4)
        if not self.error:
            view.text(
                21,
                x + 2,
                "Data stages validate ZIPs and rebuild selected Parquet and features.",
                "muted",
                width=width - 4,
            )

    def _draw_stages(self, view: Canvas, x: int, width: int):
        view.panel(9, x, 2 * len(STAGES) + 3, width, "CHOOSE STAGES", focused=True)
        for index, option in enumerate(STAGES):
            checked = "x" if option.key in self.selected else " "
            label = f" [{checked}] {option.title} "
            role = "selected" if self.focus == index and not self.editing else "text"
            view.text(10 + index * 2, x + 2, label, role, bold=True)
            view.text(11 + index * 2, x + 3, option.description, "muted", width=width - 6)
        label = f"  Run {len(self.selected)} selected stage{'s' if len(self.selected) != 1 else ''}  "
        view.text(
            10 + 2 * len(STAGES),
            x + 2,
            label,
            "selected" if self.focus == len(STAGES) else "accent",
            bold=True,
        )

    def run(self, screen) -> Selection | None:
        import curses

        self.theme = prepare_screen(screen)
        while not self.cancelled:
            self.draw(screen)
            try:
                key = screen.get_wch()
            except curses.error:
                time.sleep(0.05)
                continue
            if Canvas(screen, self.theme).too_small() and key not in (
                "q",
                "\x03",
                "\x1b",
            ):
                continue
            selection = self.handle(key)
            if selection:
                return selection
        return None


def choose_installation(data_dir: Path, dates: str, *, configured_stages=()) -> Selection | None:
    import curses

    return curses.wrapper(SetupMenu(data_dir, dates, configured_stages=configured_stages).run)
