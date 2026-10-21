"""Small, inspectable setup examples without downloading scientific dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from installer import environment


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project with spaces"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "0.1"\n[dependency-groups]\ndev = ["pytest"]\n'
    )
    return root


def _fake_venv(root: Path) -> Path:
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /python\n")
    (venv / "bin" / "python").touch()
    return venv


def test_preview_does_not_execute_or_write_anything(tmp_path, monkeypatch, capsys):
    """Preview is useful before Git, Python, or data prerequisites are available.

    Give setup a path that does not even exist and prohibit subprocess calls.
    The displayed plan must include the Git update, editable install, development
    group, isolated import check, and notebook kernel, without creating files.
    """
    root = tmp_path / "absent project"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("preview ran a subprocess"))
    assert environment.setup_environment(root, dry_run=True) == 0
    assert not root.exists()
    output = capsys.readouterr().out
    assert "pull --ff-only --no-rebase" in output
    assert "--editable" in output and "pyproject.toml:dev" in output
    assert "-I -c" in output and "--sys-prefix" in output
    assert "no code, environment, configuration, or logs were changed" in output


def test_fresh_setup_uses_pyproject_groups_and_local_kernel(tmp_path, monkeypatch, capsys):
    """An empty checkout needs one venv and dependencies taken from pyproject.

    Record commands instead of installing packages. The path contains spaces to
    expose accidental shell splitting. Check exact editable/group arguments,
    dependency validation, isolated imports, and an environment-local kernel.
    --no-update must skip Git, and successful setup must leave a real .log file.
    """
    root = _project(tmp_path)
    commands = []
    monkeypatch.setattr(environment, "_find_python", lambda requested: "/Python 3.13/bin/python")
    monkeypatch.setattr(environment, "_update_checkout", lambda *a: pytest.fail("--no-update called Git"))
    monkeypatch.setattr(environment, "_run", lambda command, *a, **kw: commands.append(command))
    assert environment.setup_environment(root, update=False) == 0
    python = str(root / ".venv/bin/python")
    assert commands[0] == ["/Python 3.13/bin/python", "-m", "venv", str(root / ".venv")]
    assert commands[1] == [python, "-m", "pip", "install", "--upgrade", "pip>=25.1"]
    assert commands[2] == [
        python,
        "-m",
        "pip",
        "install",
        "--editable",
        str(root),
        "--group",
        f"{root}/pyproject.toml:dev",
    ]
    assert commands[3] == [python, "-m", "pip", "check"]
    assert commands[4][1:3] == ["-I", "-c"]
    assert commands[4][-1] == str(root)
    assert "--sys-prefix" in commands[5] and "--user" not in commands[5]
    logs = list((root / "logs").iterdir())
    assert len(logs) == 1 and logs[0].suffix == ".log"
    assert "Research environment setup completed" in logs[0].read_text()
    assert "Activate: source '" in capsys.readouterr().out


def test_valid_existing_environment_is_reused_without_clearing(tmp_path, monkeypatch):
    """Rerunning setup should update packages while preserving a valid .venv.

    A dummy venv reports Python 3.13 and a prefix matching this checkout. A marker
    represents user files. No interpreter search or venv creation may occur;
    dependency installation still runs and the marker must retain its contents.
    """
    root = _project(tmp_path)
    venv = _fake_venv(root)
    marker = venv / "keep.txt"
    marker.write_text("keep me")
    monkeypatch.setattr(
        environment,
        "_python_info",
        lambda executable: dict(version=[3, 13, 7], prefix=str(venv), base_prefix="/python"),
    )
    monkeypatch.setattr(environment, "_find_python", lambda *a: pytest.fail("searched for another interpreter"))
    commands = []
    monkeypatch.setattr(environment, "_run", lambda command, *a, **kw: commands.append(command))
    assert environment.setup_environment(root, update=False) == 0
    assert all("venv" not in command and "--clear" not in command for command in commands)
    assert any("--editable" in command for command in commands)
    assert marker.read_text() == "keep me"


@pytest.mark.parametrize("prefix", ["/another/venv", "/python"])
def test_wrong_existing_environment_fails_before_package_install(tmp_path, monkeypatch, prefix):
    """A Python executable inside .venv is insufficient proof of isolation.

    Simulate either a different environment or the base interpreter being
    returned by that executable. Both must fail before pip runs so setup cannot
    modify someone else's environment or the host's Python installation.
    """
    root = _project(tmp_path)
    _fake_venv(root)
    monkeypatch.setattr(
        environment, "_python_info", lambda executable: dict(version=[3, 13, 7], prefix=prefix, base_prefix="/python")
    )
    monkeypatch.setattr(environment, "_run", lambda *a, **kw: pytest.fail("invalid venv ran pip"))
    assert environment.setup_environment(root, update=False) == 1


def test_arbitrary_existing_directory_is_not_overwritten(tmp_path, monkeypatch):
    """A directory called .venv may contain unrelated files rather than Python.

    Put one human-readable file there without a pyvenv.cfg. Setup must fail,
    leave the file byte-for-byte unchanged, and never create or clear a venv.
    """
    root = _project(tmp_path)
    (root / ".venv").mkdir()
    marker = root / ".venv/notes.txt"
    marker.write_text("valuable notes")
    monkeypatch.setattr(environment, "_run", lambda *a, **kw: pytest.fail("overwrote an existing directory"))
    assert environment.setup_environment(root, update=False) == 1
    assert marker.read_text() == "valuable notes"


def test_failed_dependency_install_stops_and_is_logged(tmp_path, monkeypatch, capsys):
    """A pip failure must not be reported as a ready research environment.

    Simulate successful venv creation followed by failure upgrading pip. No
    later install, validation, or kernel command may run, and the failure text
    must be present both in stderr and in the .log file for later inspection.
    """
    root = _project(tmp_path)
    monkeypatch.setattr(environment, "_find_python", lambda requested: sys.executable)
    commands = []

    def fail_pip(command, *args, **kwargs):
        commands.append(command)
        if "pip" in command:
            raise RuntimeError("dummy dependency error")

    monkeypatch.setattr(environment, "_run", fail_pip)
    assert environment.setup_environment(root, update=False) == 1
    assert len(commands) == 2
    assert "dummy dependency error" in capsys.readouterr().err
    assert "dummy dependency error" in next((root / "logs").glob("*.log")).read_text()


def test_changed_checkout_restarts_updated_installer_with_same_log(tmp_path, monkeypatch):
    """Updating source files must not keep executing already-imported old code.

    Pretend a fast-forward changed HEAD and intercept execve before it replaces
    pytest. The new interpreter must load this checkout, continue without a
    second pull, preserve an explicit Python path, and append to the same log.
    """
    root = _project(tmp_path)
    monkeypatch.setattr(environment, "_update_checkout", lambda *a: True)
    monkeypatch.setattr(environment, "_finish_setup", lambda *a: pytest.fail("ran stale installer code"))
    captured = {}

    class Restarted(Exception):
        pass

    def restart(executable, command, env):
        captured.update(executable=executable, command=command, env=env)
        raise Restarted

    monkeypatch.setattr(os, "execve", restart)
    with pytest.raises(Restarted):
        environment.setup_environment(root, python="/my python/bin/python")
    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "installer.environment"]
    assert command[command.index("--root") + 1] == str(root)
    assert Path(command[command.index("--log-file") + 1]).is_file()
    assert command[-2:] == ["--python", "/my python/bin/python"]
    assert "--setup-environment" not in command
    assert captured["env"]["PYTHONPATH"] == str(root / "src")


def test_subprocess_output_is_saved_using_logging(tmp_path, capsys):
    """A real tiny subprocess proves both output streams reach the same .log.

    The child prints one stdout line and one stderr line, then exits with 7.
    Verify the exception, console output, and logged command/exit code, so a
    future buffering or subprocess change cannot silently hide installation
    diagnostics. No package installation or network access is involved.
    """
    destination = tmp_path / "setup.log"
    with environment.file_logger(destination) as logger:
        with pytest.raises(RuntimeError, match="exit 7"):
            environment._run(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('from stdout'); print('from stderr', file=sys.stderr); sys.exit(7)",
                ],
                tmp_path,
                logger,
            )
    logged = destination.read_text()
    output = capsys.readouterr().out
    assert "INFO from stdout" in logged and "INFO from stderr" in logged
    assert "from stdout" in output and "from stderr" in output
    assert "Exit code: 7" in logged


def test_dependency_commands_remove_inherited_import_overrides(tmp_path, monkeypatch):
    """The installer's bootstrap PYTHONPATH must not mask a broken editable install.

    Set both Python path overrides and an unrelated active environment. The
    command environment must discard them, point VIRTUAL_ENV/PATH to the new
    venv, and require pip to run in a virtual environment; os.environ stays intact.
    """
    monkeypatch.setenv("PYTHONPATH", "/wrong/source")
    monkeypatch.setenv("PYTHONHOME", "/wrong/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/other/venv")
    target = tmp_path / ".venv"
    clean = environment._clean_environment(target)
    assert "PYTHONPATH" not in clean and "PYTHONHOME" not in clean
    assert clean["VIRTUAL_ENV"] == str(target)
    assert clean["PATH"].split(os.pathsep)[0] == str(target / "bin")
    assert clean["PIP_REQUIRE_VIRTUALENV"] == "true"
    assert os.environ["PYTHONPATH"] == "/wrong/source"


def test_python_version_is_checked_in_isolation():
    """The real current test interpreter supplies its own version and prefixes.

    This exercises the JSON subprocess protocol instead of a fake version
    object. Project tests run on supported Python, so the result must agree
    exactly with sys.version_info and the active interpreter's environment.
    """
    info = environment._python_info(sys.executable)
    assert info["version"] == list(sys.version_info[:3])
    assert info["prefix"] == sys.prefix
