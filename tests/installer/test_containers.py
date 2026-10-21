"""Hand-checkable installer tests using fake Docker commands and tiny log records.

No image is pulled and no exchange request is made. The tests exercise the host
command boundary, configuration precedence, progress transport and interruption
cleanup that would otherwise be easy to miss in image-only tests.
"""

from collections import deque
import csv
from io import StringIO
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from installer import cli
from installer.containers import IMAGES, container_stage
from installer.menu import SetupMenu
from installer.runner import _StageOutput, monitor_stage

ROOT = Path(__file__).resolve().parents[2]


def stages_for(arguments, tmp_path, progress=False):
    """Parse real host CLI arguments and return the four generated container stages."""
    args = cli._build_parser().parse_args(arguments)
    return cli._build_stages(args, list(cli.STAGE_KEYS), tmp_path, progress=progress)


def test_pipeline_uses_public_images_and_image_manifests(tmp_path):
    """The four pipeline stages map to exactly three requested DockerHub images.

    Both downloads must use pds-dataloader, while conversion and features use
    the corresponding pds-prefixed names. Every run pulls, removes its finished
    container, runs as the caller and mounts the same host data. The trade list
    must resolve inside /app in the image, never to an unmounted host checkout.
    """
    stages = stages_for(["--start", "2021-01-01", "--end", "2021-01-02"], tmp_path)
    assert [stage.key for stage in stages] == ["trades", "candles", "parquet", "features"]
    assert [next(value for value in stage.command if value.endswith(":latest")) for stage in stages] == [
        "b0r0koko/pds-dataloader:latest",
        "b0r0koko/pds-dataloader:latest",
        "b0r0koko/pds-preprocessor:latest",
        "b0r0koko/pds-feature-writer:latest",
    ]
    for stage in stages:
        assert stage.command[:6] == ["docker", "run", "--pull", "always", "--rm", "--init"]
        assert f"{os.getuid()}:{os.getgid()}" in stage.command
        assert f"type=bind,src={tmp_path},dst=/data" in stage.command
    assert "/app/resources/btc_currencies.json" in stages[0].command
    assert "--start-date" not in stages[3].command
    assert len({stage.container_name for stage in stages}) == 4


@pytest.mark.parametrize("key", ["trades", "candles", "parquet", "features"])
def test_stage_uses_direct_cli_entrypoint(key, tmp_path):
    """Each image receives parser options directly, without a shell dispatch alias.

    A single --help option makes the Docker/CLI boundary hand-checkable: the
    image name must be followed immediately by --help and nothing else. Trade
    downloads, preprocessing and features use their Dockerfile entrypoint.
    Candles share the downloader image and select its other installed command
    with Docker's --entrypoint option, which must precede the image reference.
    """
    stage = container_stage(key, "Preview command", ["--help"], tmp_path)
    image_index = stage.command.index(IMAGES[key])
    assert stage.command[image_index + 1 :] == ["--help"]
    if key == "candles":
        assert stage.command[image_index - 2 : image_index] == ["--entrypoint", "pumps-download-klines"]
    else:
        assert "--entrypoint" not in stage.command


def test_config_only_passes_explicit_overrides_and_mounts_siblings(tmp_path):
    """Config-only invocation preserves its dates, pair selection and worker count.

    A file beside a sibling manifest is mounted through its entire parent at
    /config, read-only, so relative references still resolve. No installer
    default date, default pair list or CPU count may silently overwrite values
    in that config. Explicit host options are checked separately below.
    """
    config = tmp_path / "my config, $literal.conf"
    config.write_text("cpu_count=1")
    stages = stages_for(["--trades-config", str(config), "--features-config", str(config)], tmp_path)
    for stage in (stages[0], stages[3]):
        image_index = stage.command.index(IMAGES[stage.key])
        assert stage.command[image_index + 1 :] == ["--config", f"/config/{config.name}"]
        assert f"type=bind,src={tmp_path},dst=/config,readonly" in stage.command


def test_explicit_dates_workers_and_skip_override_config(tmp_path):
    """Explicit host dates and worker flags follow the config in the container argv.

    The expected four-token date override is easy to inspect and uses the
    image's accepted start/end aliases. Feature dates remain in the feature
    config, since installer dates describe archives; only its explicitly chosen
    worker count and reuse flag are forwarded to the feature writer.
    """
    config = tmp_path / "pipeline.conf"
    config.write_text("{}")
    arguments = [
        "--start",
        "2021-01-01",
        "--end",
        "2021-01-02",
        "--hive-workers",
        "2",
        "--feature-workers",
        "3",
        "--skip-existing-features",
    ]
    for key in cli.STAGE_KEYS:
        arguments.extend([f"--{key}-config", str(config)])
    stages = stages_for(arguments, tmp_path)
    assert stages[2].command[-6:] == ["--start", "2021-01-01", "--end", "2021-01-02", "--processes", "2"]
    assert stages[3].command[-3:] == ["--cpu-count", "3", "--skip-existing"]
    assert "--start" not in stages[3].command


def test_mount_grammar_preserves_comma_quote_space_and_dollar(tmp_path):
    """Host paths remain one literal bind source even with Docker CSV metacharacters.

    Docker parses --mount as CSV; ordinary shell quoting alone is insufficient
    for commas inside a path. Decode the generated field with the same CSV
    grammar and compare it to the exact requested source, including a quote,
    comma, dollar sign and space. No shell expansion is involved.
    """
    data_dir = tmp_path / 'data, "quoted" $literal'
    stage = container_stage("trades", "Download", [], data_dir)
    mount = stage.command[stage.command.index("--mount") + 1]
    assert next(csv.reader([mount])) == ["type=bind", f"src={data_dir}", "dst=/data"]


def test_dry_run_does_not_pull_save_or_create_data(tmp_path, monkeypatch, capsys):
    """Previewing a full pipeline prints all image references without side effects.

    A failing subprocess stub proves neither Docker runs nor cleanup commands
    are invoked. The requested data directory and saved-path function must also
    remain untouched, so a dry run is usable before Docker or a dataset exists.
    """

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry run attempted a subprocess or saved configuration")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(cli, "save_data_dir", forbidden)
    data_dir = tmp_path / "not-created"
    assert cli.main(["--all", "--dry-run", "--data-dir", str(data_dir)]) == 0
    assert not data_dir.exists()
    output = capsys.readouterr().out
    for image in set(IMAGES.values()):
        assert image in output


def test_console_failure_stops_pipeline_and_cleans_its_container(tmp_path, monkeypatch):
    """A failed first container returns its status and prevents every later stage.

    A tiny Python child substitutes for the first Docker client and returns 23.
    The only daemon operation must be forced removal of that stage's unique
    name; no second downloader or derived-data stage may start after failure.
    This exercises the same logged process monitor used by actual console runs.
    """
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    stages = stages_for([], tmp_path)
    stages[0].command = [sys.executable, "-c", "raise SystemExit(23)"]
    assert cli._run_console(stages, tmp_path, dict(os.environ), False) == 23
    assert calls == [["docker", "rm", "--force", stages[0].container_name]]


def test_console_interrupt_removes_container_before_propagating(tmp_path, monkeypatch):
    """Ctrl-C cannot leave a container downloading after its host client exits.

    Raising KeyboardInterrupt on the real monitor's first tick models Ctrl-C
    and the installer's SIGTERM handler while its child is still running. The
    interrupt must propagate only after the process group is stopped and the
    stage's unique container name is force-removed.
    """
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    def interrupted_tick():
        raise KeyboardInterrupt

    def interrupted_monitor(stage, **kwargs):
        kwargs["tick"] = interrupted_tick
        return monitor_stage(stage, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "monitor_stage", interrupted_monitor)
    stages = stages_for([], tmp_path)
    stages[0].command = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(KeyboardInterrupt):
        cli._run_console(stages, tmp_path, dict(os.environ), False)
    assert stages[0].status == "cancelled"
    assert calls[-1] == ["docker", "rm", "--force", stages[0].container_name]


def test_progress_json_is_separated_from_container_logs(tmp_path):
    """Container stdout transports progress without filling the dashboard log view.

    A split JSON record represents a real pipe read boundary; the normal log
    before it and unrelated JSON after it must remain visible. Exactly one
    overall update reaches the stage state, while the full original stream is
    preserved as logging records for the on-disk installer log. A string
    stream with the same standard handler interface avoids real disk I/O here.
    """
    stage = container_stage("trades", "Download", [], tmp_path, progress=True)
    assert "PUMPS_AND_DUMPS_PROGRESS_FD=1" in stage.command
    logs = deque()
    stored = StringIO()
    logger = logging.Logger("test-stage-output")
    logger.addHandler(logging.StreamHandler(stored))
    output = _StageOutput(stage, logs, logger)
    chunks = [b'Downloading archive\n{"task":"overall",', b'"completed":2,"total":3}\n{"message":"ordinary JSON"}\n']
    for chunk in chunks:
        output.feed("log", chunk)
    assert stage.progress == {"task": "overall", "completed": 2, "total": 3}
    assert list(logs) == ["Downloading archive", '{"message":"ordinary JSON"}']
    assert stored.getvalue() == b"".join(chunks).decode()


def test_dashboard_monitor_cleans_container_after_interrupt(tmp_path, monkeypatch):
    """The real monitor cleans up a Docker stage when its dashboard is interrupted.

    A tiny sleeping Python subprocess substitutes for the Docker client. Its
    first UI tick raises Ctrl-C; the monitor must stop the process group, mark
    the stage cancelled, and force-remove its named container. This covers the
    dashboard path independently from the ordinary console cleanup test.
    """
    from installer import containers

    stage = container_stage("features", "Features", [], tmp_path, progress=True)
    stage.command = [sys.executable, "-c", "import time; time.sleep(30)"]
    calls = []
    monkeypatch.setattr(containers.subprocess, "run", lambda command, **kwargs: calls.append(command))

    def interrupted_tick():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        monitor_stage(
            stage,
            root=ROOT,
            environment=dict(os.environ),
            log_path=tmp_path / "log",
            logs=deque(),
            tick=interrupted_tick,
        )
    assert stage.status == "cancelled"
    assert calls == [["docker", "rm", "--force", stage.container_name]]


def test_configured_stage_menu_defers_custom_input_checks(tmp_path):
    """A configured feature stage may read a custom path instead of the default Hive root.

    With no default dataset, the normal feature-only menu rejects its missing
    input. Marking that stage as configured lets its real CLI validate its own
    input path after the read-only config is mounted. Unconfigured stages still
    retain the existing prerequisite check.
    """
    menu = SetupMenu(tmp_path, "Dates", configured_stages={"features"})
    menu.custom = True
    menu.selected = {"features"}
    assert menu.submit().stages == ["features"]
    menu.configured_stages = set()
    assert menu.submit() is None
    assert "No Hive Parquet" in menu.error


def test_launcher_works_without_poetry_or_site_packages(tmp_path):
    """The shell launcher needs only standard-library Python to preview the pipeline.

    An isolated PATH exposes Bash utilities and a Python wrapper using -S, which
    disables site packages. There is deliberately no Poetry or Docker command.
    Running from another directory must still find the checkout and resolve a
    relative data path against that caller directory, without creating it.
    """
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("bash", "dirname"):
        import shutil

        (binaries / name).symlink_to(shutil.which(name))
    python = binaries / "python3"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "scripts" / "install.sh"), "--all", "--dry-run", "--data-dir", "relative-data"],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(binaries), "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / "relative-data") in result.stdout
    assert "b0r0koko/pds-feature-writer:latest" in result.stdout
    assert not (tmp_path / "relative-data").exists()


def test_python_before_313_uses_cpu_affinity(monkeypatch):
    """Older supported host Pythons preserve the installer's available-CPU limit.

    Pretend process_cpu_count is unavailable and the OS exposes 64 physical
    logical CPUs but allows this process to run on only CPUs 1 and 3. The
    expected worker default is two, preventing an older host interpreter from
    oversubscribing a limited container or batch allocation.
    """
    from installer.plan import available_cpu_count

    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {1, 3}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert available_cpu_count() == 2


def test_explicit_polars_limit_reaches_processing_containers(tmp_path, monkeypatch):
    """A user-specified Polars thread budget still reaches the containerized stages.

    Docker's bare --env form imports the named host variable without putting its
    value into a shell command. Both Parquet and features must receive the
    setting; a downloader has no Polars dependency and should not receive it.
    """
    monkeypatch.setenv("POLARS_MAX_THREADS", "2")
    stages = stages_for([], tmp_path)
    assert "POLARS_MAX_THREADS" not in stages[0].command
    for stage in stages[2:]:
        assert "POLARS_MAX_THREADS" in stage.command
