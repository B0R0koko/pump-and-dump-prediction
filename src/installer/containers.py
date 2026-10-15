"""Construct stage containers and remove them when the installer is interrupted.

The host installer uses only the standard library. Scientific dependencies and
the data commands themselves come from the published images, pulled on every
stage invocation. Image resources are read inside /app, while all output is
kept in the caller's mounted data directory.
"""

import csv
from io import StringIO
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from installer.state import Stage

IMAGES = {
    "trades": "b0r0koko/pds-dataloader:latest",
    "candles": "b0r0koko/pds-dataloader:latest",
    "parquet": "b0r0koko/pdf-preprocessor:latest",
    "features": "b0r0koko/pdf-feature-writer:latest",
}


def bind_mount(source: Path, destination: str, *, readonly: bool = False) -> str:
    """Encode Docker's CSV mount grammar, including host paths with commas/spaces."""
    result = StringIO()
    fields = ["type=bind", f"src={source}", f"dst={destination}"]
    if readonly:
        fields.append("readonly")
    csv.writer(result, lineterminator="").writerow(fields)
    return result.getvalue()


def container_stage(
    key: str,
    description: str,
    options: list[str],
    data_dir: Path,
    *,
    config: Path | None = None,
    progress: bool = False,
) -> Stage:
    """Run a freshly pulled image as the caller with persistent host output."""
    name = f"pds-{key}-{os.getpid()}-{uuid4().hex[:12]}"
    command = ["docker", "run", "--pull", "always", "--rm", "--init", "--name", name]
    if hasattr(os, "getuid"):
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    command.extend(["--mount", bind_mount(data_dir, "/data")])
    if config is not None:
        command.extend(["--mount", bind_mount(config.parent, "/config", readonly=True)])
        options = ["--config", f"/config/{config.name}", *options]
    if progress:
        # Docker cannot inherit the host's private pipe descriptor. Existing
        # images can emit the same JSON records on fd 1 for the host to demux.
        command.extend(["--env", "PUMPS_AND_DUMPS_PROGRESS_FD=1"])
    if "POLARS_MAX_THREADS" in os.environ and key in {"parquet", "features"}:
        command.extend(["--env", "POLARS_MAX_THREADS"])
    if key == "candles":
        command.extend(["--entrypoint", "pumps-download-klines"])
    command.extend([IMAGES[key], *options])
    return Stage(description, command, key=key, container_name=name, progress_stdout=progress)


def cleanup_container(stage: Stage) -> None:
    """Remove this stage's unique container even if its Docker client was killed."""
    if stage.container_name is None:
        return
    try:
        subprocess.run(
            ["docker", "rm", "--force", stage.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Cleanup is best effort if the daemon itself becomes unavailable; do
        # not replace a useful parser failure or cancellation with this error.
        pass
