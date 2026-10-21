"""Small child processes verify search logs without training models or fetching data."""

import json
import subprocess
import sys

import optuna
import pytest

from pumps_and_dumps.backtest.research import retune


@pytest.mark.parametrize("exit_code", [0, 7])
def test_searches_capture_each_process_and_preserve_failure(tmp_path, monkeypatch, exit_code):
    """Concurrent searches retain separate complete .log files and failure status.

    Replace the expensive training command with two tiny Python children. Each
    prints its own candidate name on stdout, an stderr line, and an unterminated
    final line. A successful run must write completion metadata; an exit-seven
    run must raise the existing search failure and leave no success marker.
    This exercises pipe draining and logging without changing or running models.
    """
    real_popen = subprocess.Popen
    launched = []

    def small_child(command, **kwargs):
        candidate = command[command.index("--candidate") + 1]
        child = real_popen(
            [
                sys.executable,
                "-c",
                f"import sys; print({candidate!r}); print('stderr', file=sys.stderr); "
                f"sys.stdout.write('last fragment'); sys.exit({exit_code})",
            ],
            **kwargs,
        )
        launched.append(child)
        return child

    monkeypatch.setattr(retune, "CANDIDATES", {"first": ("A", "A", 1), "second": ("B", "B", 1)})
    monkeypatch.setattr(retune.subprocess, "Popen", small_child)
    monkeypatch.setattr(optuna.storages, "RDBStorage", lambda _storage: None)
    output = tmp_path / "searches"
    storage = f"sqlite:///{tmp_path / 'study.sqlite'}"
    if exit_code:
        with pytest.raises(RuntimeError, match="Search failed"):
            retune.run_searches(storage, output, trials=3)
        assert not (output / "complete.json").exists()
    else:
        retune.run_searches(storage, output, trials=3)
        complete = json.loads((output / "complete.json").read_text())
        assert complete["models"] == 2
        assert complete["trials_per_model"] == 3
    for candidate in ("first", "second"):
        lines = (output / f"{candidate}.log").read_text().splitlines()
        assert len(lines) == 3
        assert {line.split(" INFO ", 1)[1] for line in lines} == {candidate, "stderr", "last fragment"}
    assert all(child.poll() is not None for child in launched)


def test_capture_failure_stops_running_searches(tmp_path, monkeypatch):
    """A log-writer exception is reported promptly and stops its search process.

    Use a sleeping child so it cannot finish by itself during the test, and a
    capture worker that raises the same OSError as a full disk. The caller must
    receive that exception, terminate its process group, and avoid publishing
    completion metadata. This guards against ignoring background thread errors
    or waiting for a long training job after its persistent logging has failed.
    """
    real_popen = subprocess.Popen
    launched = []

    def sleeping_child(_command, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        launched.append(child)
        return child

    def failed_capture(process, _path):
        process.stdout.close()
        raise OSError("disk full")

    monkeypatch.setattr(retune, "CANDIDATES", {"first": ("A", "A", 1)})
    monkeypatch.setattr(retune.subprocess, "Popen", sleeping_child)
    monkeypatch.setattr(retune, "_capture_output", failed_capture)
    monkeypatch.setattr(optuna.storages, "RDBStorage", lambda _storage: None)
    output = tmp_path / "searches"
    with pytest.raises(OSError, match="disk full"):
        retune.run_searches(f"sqlite:///{tmp_path / 'study.sqlite'}", output, trials=3)
    assert all(child.poll() is not None for child in launched)
    assert not (output / "complete.json").exists()
