"""Standalone pipeline imports must work without notebook import ordering."""

import subprocess
import sys


def test_random_forest_import_in_fresh_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from pumps_and_dumps.backtest.pipelines.random_forest.pipeline import RandomForestPipeline; "
            "assert 'pumps_and_dumps.backtest.portfolio' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
