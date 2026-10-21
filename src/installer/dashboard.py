"""Responsive layout for stage progress, live activity, and logs."""

from dataclasses import dataclass
import textwrap
import time

from installer.ui import Canvas
from installer.state import duration


@dataclass(frozen=True)
class _ActivityRegion:
    y: int
    x: int
    width: int
    end: int


def _stage_bar(stage):
    if stage.status == "done":
        return 1.0
    total = stage.progress.get("total")
    if total is None or not stage.progress.get("total_known", True):
        return None if stage.status == "running" else 0.0
    return min(0.96, stage.progress.get("completed", 0) / total) if total else 0.0


def _role(status):
    return {
        "done": "success",
        "running": "accent",
        "failed": "danger",
        "cancelled": "warning",
    }.get(status, "muted")


def _progress_count(progress):
    count = f"{progress.get('completed', 0):,}"
    if progress.get("total") is not None:
        count += f" / {progress['total']:,}"
    return count


def _timing(stage, compact=False):
    completed = stage.progress.get("completed", 0)
    total = stage.progress.get("total")
    if total is None or not stage.progress.get("total_known", True):
        timing = "discovering total"
    else:
        rate = completed / max(stage.elapsed(), 0.001)
        eta = duration(max(0, total - completed) / rate) if rate else "--:--:--"
        timing = f"{rate:,.1f}/s · ETA {eta}"
    if compact:
        return f"{_progress_count(stage.progress)} {stage.progress.get('unit', '')} · {timing}"
    return f"{duration(stage.elapsed())} · {timing}"


def _render_summary(view, dashboard, current, wide, now):
    elapsed = (dashboard.ended or now) - dashboard.started
    done = sum(stage.status == "done" for stage in dashboard.stages)
    if not wide:
        view.text(
            3,
            2,
            f"Stages finished: {done}/{len(dashboard.stages)}   {duration(elapsed)}   {dashboard.free_space} free",
            "accent",
        )
        return
    status = current.status if current else "ready"
    tile_width = (view.width - 7) // 4
    tiles = (
        ("STAGES FINISHED", f"{done} / {len(dashboard.stages)}", "accent"),
        ("ELAPSED", duration(elapsed), "text"),
        ("DISK AVAILABLE", dashboard.free_space, "text"),
        ("STATUS", status.upper(), _role(status)),
    )
    for index, (label, value, role) in enumerate(tiles):
        x = 2 + index * (tile_width + 1)
        view.panel(4, x, 4, tile_width, label)
        view.text(6, x + 2, value, role, bold=True, width=tile_width - 4)


def _render_pipeline(view, stages, panel_y, width, wide):
    for index, stage in enumerate(stages):
        y = panel_y + 1 + index * (3 if wide else 2)
        role = _role(stage.status)
        view.text(y, 4, f"{index + 1:02}  {stage.title}", bold=stage.status == "running", width=width - 17)
        view.text(y, width - 10, f"{stage.status.upper():>10}", role, width=10, bold=True)
        view.bar(y + 1, 4, 12 if wide else 8, _stage_bar(stage), role)
        timing = _timing(stage, compact=not wide) if stage.started else "Waiting to start"
        if wide:
            view.text(
                y + 1, 18, f"{_progress_count(stage.progress)} {stage.progress.get('unit', '')}", role, width=width - 18
            )
            view.text(y + 2, 4, timing, "muted", width=width - 4)
        else:
            view.text(y + 1, 14, timing, "muted", width=width - 14)


def _activity_counters(stage):
    counters = [
        f"{key.replace('_', ' ')} {stage.progress[key]:,}"
        for key in ("verified", "downloaded", "missing", "damaged", "reused", "failed", "written", "unavailable")
        if key in stage.progress
    ]
    if "received_bytes" in stage.progress:
        received = stage.progress["received_bytes"] / 1024**2
        counters.append(f"{received:,.1f} MiB  /  {received / max(stage.elapsed(), 0.001):.1f} MiB/s")
    return counters


def _render_workers(view, stage, region, y):
    view.text(y, region.x, "WORKERS", "muted", bold=True)
    y += 1
    workers = list(stage.workers.items())
    slots = max(0, (region.end - y - 1) // 2)
    for task, progress in workers[:slots]:
        view.text(y, region.x, f"{task}  ·  {_progress_count(progress)} {progress.get('unit', '')}", width=region.width)
        view.text(y + 1, region.x, progress.get("detail", ""), "muted", width=region.width)
        y += 2
    if len(workers) > slots:
        view.text(y, region.x, f"+ {len(workers) - slots} other active workers", "muted")


def _render_activity(view, current, active, region, wide, now):
    y, x, width = region.y, region.x, region.width
    if current is None:
        if wide:
            view.text(y, x, "Waiting for the first stage", "muted", width=width)
        return
    if wide:
        age = duration((current.ended or now) - current.last_update)
        view.text(y, x, f"PID {current.pid}  ·  Updated {age} ago", "muted", width=width)
        y += 1
    detail = current.progress.get("detail", "Starting the selected stage…")
    lines = textwrap.wrap(detail, width)[:2] if wide else [detail]
    for line in lines:
        view.text(y, x, line, "accent", width=width)
        y += 1
    for line in textwrap.wrap("  ·  ".join(_activity_counters(current)), width)[: 3 if wide else 1]:
        view.text(y, x, line, width=width)
        y += 1
    if wide and active:
        _render_workers(view, active, region, y + 1)


def _render_logs(view, dashboard, y):
    height = view.height - y - 2
    view.panel(y, 2, height, view.width - 4, "LOG HISTORY" if dashboard.scroll else "LIVE LOG")
    visible = max(0, height - 2)
    end = max(0, len(dashboard.logs) - dashboard.scroll)
    for offset, line in enumerate(list(dashboard.logs)[max(0, end - visible) : end]):
        role = "danger" if any(word in line.lower() for word in ("error", "failed", "traceback")) else "muted"
        view.text(y + 1 + offset, 4, line, role, width=view.width - 8)
    view.text(view.height - 2, 2, f"Logs: {dashboard.log_dir}", "muted", width=view.width - 4)
    controls = "Enter/q: close" if dashboard.finished else "q / Ctrl-C Cancel"
    view.text(view.height - 1, 2, f"{controls}   ↑↓ Scroll   PgUp/PgDn Page   End Live", "accent", width=view.width - 4)


def render(dashboard, screen):
    screen.erase()
    view = Canvas(screen, dashboard.theme)
    view.header(f"{dashboard.dates} · {dashboard.data_dir}")
    if view.too_small():
        screen.refresh()
        return
    now = time.monotonic()
    active = next((stage for stage in dashboard.stages if stage.status == "running"), None)
    current = active or next((stage for stage in reversed(dashboard.stages) if stage.started), None)
    wide = view.width >= 110 and view.height >= 33
    _render_summary(view, dashboard, current, wide, now)

    if wide:
        panel_y = 9
        left_width = (view.width - 5) // 2
        panel_height = max(9, 3 * len(dashboard.stages) + 3)
        right_x = left_width + 3
        right_width = view.width - right_x - 2
        region = _ActivityRegion(panel_y + 1, right_x + 2, right_width - 4, panel_y + panel_height - 1)
    else:
        panel_y = 5
        left_width = view.width - 4
        panel_height = 2 * len(dashboard.stages) + 3
        region = _ActivityRegion(panel_y + panel_height, 2, view.width - 4, panel_y + panel_height + 2)
    view.panel(panel_y, 2, panel_height, left_width, "PIPELINE", focused=active is not None)
    if wide:
        view.panel(panel_y, right_x, panel_height, right_width, "LIVE ACTIVITY", focused=active is not None)
    _render_pipeline(view, dashboard.stages, panel_y, left_width, wide)
    _render_activity(view, current, active, region, wide, now)
    _render_logs(view, dashboard, panel_y + panel_height + (1 if wide else 2))
    screen.refresh()
