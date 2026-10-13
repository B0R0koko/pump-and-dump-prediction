"""Shared terminal theme and drawing primitives for setup and progress screens."""

from dataclasses import dataclass, field
import sys
import time
import unicodedata


def clip(text: str, width: int) -> str:
    result = []
    used = 0
    for char in str(text):
        if not char.isprintable():
            continue
        size = (
            0
            if unicodedata.combining(char)
            else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
        )
        if used + size > width:
            break
        result.append(char)
        used += size
    return "".join(result)


@dataclass
class Theme:
    styles: dict[str, int] = field(default_factory=dict)
    unicode: bool = True


def prepare_screen(screen) -> Theme:
    import curses

    theme = Theme()
    try:
        "╭─╮│╰╯━✓›".encode(sys.stdout.encoding or "ascii")
    except UnicodeEncodeError:
        theme.unicode = False
    screen.nodelay(True)
    screen.keypad(True)
    try:
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            background = 234 if curses.COLORS >= 256 else curses.COLOR_BLACK
            colors = {
                "text": (253, curses.COLOR_WHITE),
                "muted": (245, curses.COLOR_WHITE),
                "border": (239, curses.COLOR_BLUE),
                "accent": (80, curses.COLOR_CYAN),
                "success": (114, curses.COLOR_GREEN),
                "warning": (221, curses.COLOR_YELLOW),
                "danger": (210, curses.COLOR_RED),
            }
            for index, (role, (extended, basic)) in enumerate(colors.items(), start=1):
                curses.init_pair(
                    index, extended if curses.COLORS >= 256 else basic, background
                )
                theme.styles[role] = curses.color_pair(index)
            curses.init_pair(
                8,
                234 if curses.COLORS >= 256 else curses.COLOR_BLACK,
                80 if curses.COLORS >= 256 else curses.COLOR_CYAN,
            )
            theme.styles["selected"] = curses.color_pair(8) | curses.A_BOLD
            screen.bkgd(" ", theme.styles["text"])
    except curses.error:
        pass  # The same layout remains usable in monochrome terminals.
    return theme


class Canvas:
    def __init__(self, screen, theme: Theme):
        self.screen = screen
        self.theme = theme
        self.height, self.width = screen.getmaxyx()

    def text(
        self,
        y: int,
        x: int,
        text: str,
        role: str = "text",
        *,
        width: int | None = None,
        bold=False,
    ):
        import curses

        if not 0 <= y < self.height or not 0 <= x < self.width - 1:
            return
        available = min(width if width is not None else self.width, self.width - x - 1)
        value = clip(text, available)
        style = self.theme.styles.get(
            role, curses.A_REVERSE if role == "selected" else 0
        )
        if bold:
            style |= curses.A_BOLD
        try:
            self.screen.addnstr(y, x, value, len(value), style)
        except curses.error:
            pass

    def panel(
        self, y: int, x: int, height: int, width: int, title: str, *, focused=False
    ):
        if height < 2 or width < 4:
            return
        tl, tr, bl, br, horizontal, vertical = (
            "╭╮╰╯─│" if self.theme.unicode else "++++-|"
        )
        role = "accent" if focused else "border"
        self.text(y, x, tl + horizontal * (width - 2) + tr, role, width=width)
        for row in range(y + 1, y + height - 1):
            self.text(row, x, vertical, role)
            self.text(row, x + width - 1, vertical, role)
        self.text(
            y + height - 1, x, bl + horizontal * (width - 2) + br, role, width=width
        )
        self.text(
            y,
            x + 2,
            f" {title} ",
            "accent" if focused else "muted",
            width=width - 5,
            bold=True,
        )

    def bar(self, y: int, x: int, width: int, fraction: float | None, role="accent"):
        width = max(1, width)
        filled, empty = ("━", "─") if self.theme.unicode else ("#", "-")
        self.text(y, x, empty * width, "border", width=width)
        if fraction is None:
            count = min(3, width)
            start = int(time.monotonic() * 6) % max(1, width - count + 1)
        else:
            count = min(width, max(0, int(width * fraction)))
            start = 0
        self.text(y, x + start, filled * count, role, width=count)

    def header(self, subtitle: str):
        self.text(1, 2, "PUMPS & DUMPS", "accent", bold=True)
        self.text(1, max(22, self.width - 18), "DATA INSTALLER", "muted")
        self.text(2, 2, subtitle, "muted", width=self.width - 4)

    def too_small(self) -> bool:
        if self.width >= 72 and self.height >= 24:
            return False
        self.text(4, 2, "Resize to at least 72 x 24", "warning")
        self.text(5, 2, "q: cancel / close", "muted")
        return True
