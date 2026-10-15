"""Responsive layout for stage progress, live activity, and logs."""

import textwrap
import time

from installer.ui import Canvas
from installer.state import duration


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


def _timing(stage, *, compact=False):
    completed = stage.progress.get("completed", 0)
    total = stage.progress.get("total")
    count = f"{completed:,}" + (f" / {total:,}" if total is not None else "")
    if total is None or not stage.progress.get("total_known", True):
        timing = "discovering total"
    else:
        rate = completed / max(stage.elapsed(), 0.001)
        eta = duration(max(0, total - completed) / rate) if rate else "--:--:--"
        timing = f"{rate:,.1f}/s · ETA {eta}"
    if compact:
        return f"{count} {stage.progress.get('unit', '')} · {timing}"
    return f"{duration(stage.elapsed())} · {timing}"


def render(dashboard, screen):
    screen.erase()
    view = Canvas(screen, dashboard.theme)
    view.header(f"{dashboard.dates} · {dashboard.data_dir}")
    if view.too_small():
        screen.refresh()
        return
    now = time.monotonic()
    elapsed = (dashboard.ended or now) - dashboard.started
    done = sum(stage.status == "done" for stage in dashboard.stages)
    active = next((stage for stage in dashboard.stages if stage.status == "running"), None)
    current = active or next((stage for stage in reversed(dashboard.stages) if stage.started), None)
    wide = view.width >= 110 and view.height >= 33
    status = (current.status if current else "ready").upper()

    if wide:
        tile_width = (view.width - 7) // 4
        for index, (label, value, role) in enumerate(
            (
                ("STAGES FINISHED", f"{done} / {len(dashboard.stages)}", "accent"),
                ("ELAPSED", duration(elapsed), "text"),
                ("DISK AVAILABLE", dashboard.free_space, "text"),
                ("STATUS", status, _role(current.status if current else "ready")),
            )
        ):
            x = 2 + index * (tile_width + 1)
            view.panel(4, x, 4, tile_width, label)
            view.text(6, x + 2, value, role, bold=True, width=tile_width - 4)
        panel_y = 9
        left_width = (view.width - 5) // 2
        panel_height = max(9, 3 * len(dashboard.stages) + 3)
        view.panel(panel_y, 2, panel_height, left_width, "PIPELINE", focused=active is not None)
        right_x = left_width + 3
        right_width = view.width - right_x - 2
        view.panel(
            panel_y,
            right_x,
            panel_height,
            right_width,
            "LIVE ACTIVITY",
            focused=active is not None,
        )
    else:
        view.text(
            3,
            2,
            f"Stages finished: {done}/{len(dashboard.stages)}   {duration(elapsed)}   {dashboard.free_space} free",
            "accent",
        )
        panel_y = 5
        left_width = view.width - 4
        panel_height = 2 * len(dashboard.stages) + 3
        view.panel(panel_y, 2, panel_height, left_width, "PIPELINE", focused=active is not None)
        right_x, right_width = 0, view.width

    for index, stage in enumerate(dashboard.stages):
        y = panel_y + 1 + index * (3 if wide else 2)
        role = _role(stage.status)
        view.text(
            y,
            4,
            f"{index + 1:02}  {stage.title}",
            bold=stage.status == "running",
            width=left_width - 17,
        )
        view.text(y, left_width - 10, f"{stage.status.upper():>10}", role, width=10, bold=True)
        if wide:
            view.bar(y + 1, 4, 12, _stage_bar(stage), role)
            completed = stage.progress.get("completed", 0)
            total = stage.progress.get("total")
            count = f"{completed:,}" + (f" / {total:,}" if total is not None else "")
            view.text(
                y + 1,
                18,
                f"{count} {stage.progress.get('unit', '')}",
                role,
                width=left_width - 18,
            )
            view.text(
                y + 2,
                4,
                _timing(stage) if stage.started else "Waiting to start",
                "muted",
                width=left_width - 4,
            )
        else:
            view.bar(y + 1, 4, 8, _stage_bar(stage), role)
            view.text(
                y + 1,
                14,
                _timing(stage, compact=True) if stage.started else "Waiting to start",
                "muted",
                width=left_width - 14,
            )

    activity_y = panel_y + 1 if wide else panel_y + panel_height
    activity_x = right_x + 2
    content_width = right_width - 4
    activity_end = panel_y + panel_height - 1 if wide else activity_y + 2
    if current:
        counters = [
            f"{key.replace('_', ' ')} {current.progress[key]:,}"
            for key in (
                "verified",
                "downloaded",
                "missing",
                "damaged",
                "reused",
                "failed",
                "written",
                "unavailable",
            )
            if key in current.progress
        ]
        if "received_bytes" in current.progress:
            received = current.progress["received_bytes"] / 1024**2
            counters.append(f"{received:,.1f} MiB  /  {received / max(current.elapsed(), 0.001):.1f} MiB/s")
        age = duration((current.ended or now) - current.last_update)
        if wide:
            view.text(
                activity_y,
                activity_x,
                f"PID {current.pid}  ·  Updated {age} ago",
                "muted",
                width=content_width,
            )
            activity_y += 1
        detail = current.progress.get("detail", "Starting the selected stage…")
        lines = textwrap.wrap(detail, content_width)[:2] if wide else [detail]
        for line in lines:
            view.text(activity_y, activity_x, line, "accent", width=content_width)
            activity_y += 1
        if counters:
            for line in textwrap.wrap("  ·  ".join(counters), content_width)[: 3 if wide else 1]:
                view.text(activity_y, activity_x, line, width=content_width)
                activity_y += 1
        if wide and active:
            activity_y += 1
            view.text(activity_y, activity_x, "WORKERS", "muted", bold=True)
            activity_y += 1
            workers = list(active.workers.items())
            slots = max(0, (activity_end - activity_y - 1) // 2)
            for task, progress in workers[:slots]:
                count = f"{progress.get('completed', 0):,}" + (
                    f" / {progress['total']:,}" if progress.get("total") is not None else ""
                )
                view.text(
                    activity_y,
                    activity_x,
                    f"{task}  ·  {count} {progress.get('unit', '')}",
                    width=content_width,
                )
                view.text(
                    activity_y + 1,
                    activity_x,
                    progress.get("detail", ""),
                    "muted",
                    width=content_width,
                )
                activity_y += 2
            if len(workers) > slots:
                view.text(
                    activity_y,
                    activity_x,
                    f"+ {len(workers) - slots} other active workers",
                    "muted",
                )
    elif wide:
        view.text(
            activity_y,
            activity_x,
            "Waiting for the first stage",
            "muted",
            width=content_width,
        )

    log_y = panel_y + panel_height + (1 if wide else 2)
    log_height = view.height - log_y - 2
    view.panel(
        log_y,
        2,
        log_height,
        view.width - 4,
        "LIVE LOG" if not dashboard.scroll else "LOG HISTORY",
    )
    visible = max(0, log_height - 2)
    end = max(0, len(dashboard.logs) - dashboard.scroll)
    for offset, line in enumerate(list(dashboard.logs)[max(0, end - visible) : end]):
        role = "danger" if any(word in line.lower() for word in ("error", "failed", "traceback")) else "muted"
        view.text(log_y + 1 + offset, 4, line, role, width=view.width - 8)
    view.text(view.height - 2, 2, f"Logs: {dashboard.log_dir}", "muted", width=view.width - 4)
    view.text(
        view.height - 1,
        2,
        (
            "Enter/q: close   ↑↓ Scroll   PgUp/PgDn Page   End Live"
            if dashboard.finished
            else "q / Ctrl-C Cancel   ↑↓ Scroll   PgUp/PgDn Page   End Live"
        ),
        "accent",
        width=view.width - 4,
    )
    screen.refresh()
