"""Set up a local, editable research environment without host dependencies.

The launcher can run on Python 3.10; research commands run on the project's
Python 3.13+ interpreter in .venv. Git updates never stash or reset user work.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from installer.logging_utils import file_logger
from installer.runner import stop_process_group


def _clean_environment(venv: Path | None = None) -> dict[str, str]:
    """Keep launcher/import overrides out of dependency installation and checks."""
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    if venv is not None:
        environment["VIRTUAL_ENV"] = str(venv)
        environment["PATH"] = str(venv / "bin") + os.pathsep + environment.get("PATH", "")
        environment["PIP_REQUIRE_VIRTUALENV"] = "true"
    return environment


def _run(command: list[str], root: Path, logger: logging.Logger, venv: Path | None = None) -> None:
    """Stream a command to the console and log, terminating its children on abort."""
    display = shlex.join(command)
    print(f"+ {display}", flush=True)
    logger.info("Command: %s", display)
    with subprocess.Popen(
        command,
        cwd=root,
        env=_clean_environment(venv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                logger.info("%s", line.rstrip("\r\n"))
            code = process.wait()
        except BaseException:
            stop_process_group(process)
            raise
    logger.info("Exit code: %s", code)
    if code:
        raise RuntimeError(f"Command failed (exit {code}): {display}")


def _git(root: Path, logger: logging.Logger, *arguments: str) -> str:
    command = ["git", "-C", str(root), *arguments]
    logger.info("Command: %s", shlex.join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git update failed: {detail}. Use --no-update to install the current checkout.")
    return result.stdout.strip()


def _update_checkout(root: Path, logger: logging.Logger) -> bool:
    """Fast-forward a clean tracking branch; return whether HEAD changed.

    Local edits skip the pull so environment setup can use the current code.
    Detached or divergent branches still require an explicit --no-update.
    """
    if Path(_git(root, logger, "rev-parse", "--show-toplevel")).resolve() != root.resolve():
        raise RuntimeError("Environment setup must run from the project checkout root.")
    if _git(root, logger, "status", "--porcelain", "--untracked-files=all"):
        message = "Skipping Git pull: the checkout has local changes. Installing the current local code."
        print(message, flush=True)
        logger.warning(message)
        return False
    if _git(root, logger, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD":
        raise RuntimeError("Detached HEAD: check out a tracking branch or use --no-update.")
    _git(root, logger, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    before = _git(root, logger, "rev-parse", "HEAD")
    # Override git configuration that might otherwise enable rebasing/autostash.
    _run(["git", "-c", "merge.autoStash=false", "pull", "--ff-only", "--no-rebase"], root, logger)
    return before != _git(root, logger, "rev-parse", "HEAD")


def _python_info(executable: str) -> dict:
    result = subprocess.run(
        [
            executable,
            "-I",
            "-c",
            "import json, sys; print(json.dumps(dict(version=list(sys.version_info[:3]), "
            "prefix=sys.prefix, base_prefix=sys.base_prefix)))",
        ],
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot run Python at {executable}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    version = tuple(info["version"])
    if not (3, 13) <= version < (4, 0):
        raise RuntimeError(
            f"Python at {executable} is {'.'.join(map(str, version))}; research needs Python 3.13+ (<4)."
        )
    return info


def _find_python(requested: str | None) -> str:
    candidates = [requested] if requested else ["python3.13", sys.executable, "python3"]
    failures = []
    for candidate in dict.fromkeys(candidates):
        executable = shutil.which(os.path.expanduser(candidate))
        if executable is None:
            failures.append(f"{candidate}: not found")
            continue
        executable = os.path.abspath(executable)
        try:
            _python_info(executable)
        except (OSError, RuntimeError) as exc:
            failures.append(str(exc))
            continue
        return executable
    raise RuntimeError("Install Python 3.13+ and select it with --python /path/to/python. " + "; ".join(failures))


def _environment_commands(root: Path) -> list[list[str]]:
    python = str(root / ".venv" / "bin" / "python")
    verify_imports = (
        "from pathlib import Path; import sys, pumps_and_dumps, installer; "
        "source = (Path(sys.argv[1]) / 'src').resolve(); "
        "assert all(source in Path(module.__file__).resolve().parents "
        "for module in (pumps_and_dumps, installer)), 'Editable imports point to another checkout'; "
        "print('Editable imports verified:', source)"
    )
    return [
        [python, "-m", "pip", "install", "--upgrade", "pip>=25.1"],
        [python, "-m", "pip", "install", "--editable", str(root), "--group", f"{root / 'pyproject.toml'}:dev"],
        [python, "-m", "pip", "check"],
        [python, "-I", "-c", verify_imports, str(root)],
        [
            python,
            "-m",
            "ipykernel",
            "install",
            "--sys-prefix",
            "--name",
            "pumps-and-dumps",
            "--display-name",
            "Pumps and dumps (.venv)",
        ],
    ]


def _install_environment(root: Path, requested: str | None, logger: logging.Logger) -> None:
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"Project metadata not found: {root / 'pyproject.toml'}")
    venv = root / ".venv"
    python = venv / "bin" / "python"
    if venv.exists() or venv.is_symlink():
        if venv.is_symlink() or not (venv / "pyvenv.cfg").is_file() or not python.is_file():
            raise RuntimeError(f"{venv} is not a usable virtual environment. Move it aside before retrying.")
        info = _python_info(str(python))
        if Path(info["prefix"]).resolve() != venv.resolve() or info["prefix"] == info["base_prefix"]:
            raise RuntimeError(
                f"{python} does not belong to this virtual environment. Move .venv aside before retrying."
            )
        if requested is not None:
            selected = _python_info(_find_python(requested))
            if selected["version"][:2] != info["version"][:2]:
                raise RuntimeError("--python differs from the existing .venv Python version. Move .venv aside first.")
        logger.info("Reusing virtual environment: %s", venv)
    else:
        _run([_find_python(requested), "-m", "venv", str(venv)], root, logger)
    for command in _environment_commands(root):
        _run(command, root, logger, venv=venv)


def _finish_setup(root: Path, python: str | None, logger: logging.Logger) -> int:
    _install_environment(root, python, logger)
    logger.info("Research environment setup completed.")
    print(f"\nResearch environment ready: {root / '.venv'}")
    print(f"Activate: source {shlex.quote(str(root / '.venv' / 'bin' / 'activate'))}")
    print(f"Start notebooks: {shlex.quote(str(root / '.venv' / 'bin' / 'jupyter'))} lab")
    print('Select the "Pumps and dumps (.venv)" notebook kernel. Imports work from any directory.')
    return 0


def setup_environment(root: Path, python: str | None = None, dry_run: bool = False, update: bool = True) -> int:
    """Update this checkout and install runtime/dev dependencies from pyproject.

    Dry runs perform no subprocess calls or writes. Local edits automatically
    skip Git updates; --no-update skips them explicitly. A changed checkout
    resumes in a fresh process so the updated installer and metadata take effect.
    """
    root = root.expanduser().resolve()
    if dry_run:
        print(f"Research environment: {root / '.venv'}")
        if update:
            print("Pull a clean tracking branch; if local edits are present, install the current code without pulling.")
            print(f"+ git -C {shlex.quote(str(root))} pull --ff-only --no-rebase")
        print(f"+ {shlex.join([python or 'python3.13', '-m', 'venv', str(root / '.venv')])}  # reuse if valid")
        for command in _environment_commands(root):
            print(f"+ {shlex.join(command)}")
        print("Preview only; no code, environment, configuration, or logs were changed.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_file = root / "logs" / f"setup-environment-{stamp}-{os.getpid()}.log"
    print(f"Setup log: {log_file}", flush=True)
    return _setup_logged(root, python, update, log_file)


def _setup_logged(root: Path, python: str | None, update: bool, log_file: Path) -> int:
    """Share logging and failure handling between initial setup and continuation."""
    with file_logger(log_file, mode="a") as logger:
        try:
            if not update or not _update_checkout(root, logger):
                return _finish_setup(root, python, logger)
            logger.info("Checkout updated; continuing with the updated installer in a fresh process.")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Environment setup failed: %s", exc)
            print(f"Environment setup failed: {exc}\nSee {log_file}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            logger.warning("Environment setup interrupted.")
            print(f"Environment setup interrupted. See {log_file}", file=sys.stderr)
            return 130

    # Re-import the updated installer rather than executing stale loaded modules.
    command = [sys.executable, "-m", "installer.environment", "--root", str(root), "--log-file", str(log_file)]
    if python is not None:
        command.extend(["--python", python])
    environment = _clean_environment()
    environment["PYTHONPATH"] = str(root / "src")
    # Replace this process so cancellation has one owner, including pip workers.
    os.execve(sys.executable, command, environment)


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _main() -> int:
    """Internal continuation after a Git update, sharing the original log file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--python")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _interrupt)
    return _setup_logged(args.root, args.python, False, args.log_file)


if __name__ == "__main__":
    raise SystemExit(_main())
