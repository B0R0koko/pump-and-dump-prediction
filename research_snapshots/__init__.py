"""Import paths for the historical notebook's model definitions.

The installed ``pumps_and_dumps`` package and installer keep the latest code.
Only the notebook's old top-level imports use the frozen research snapshot.
"""

import json
from pathlib import Path

_SNAPSHOT_DIR = Path(__file__).resolve().parent / "73ba716"


def package_path(package: str) -> str:
    archive = _SNAPSHOT_DIR / "runtime.zip"
    if not archive.is_file():
        raise ImportError("Historical runtime missing; run python research_snapshots/build_snapshot.py")
    return f"{archive}/{package}"


def research_inputs() -> tuple[Path, dict]:
    """Resolve the pinned paper cohort without embedding revision IDs in notebooks."""
    provenance = json.loads((_SNAPSHOT_DIR / "provenance.json").read_text())
    root = Path(__file__).resolve().parents[1]
    return root / provenance["active_pumps_path"], provenance


def input_provenance(provenance: dict, manifest_sha256: str) -> dict:
    """Keep exact revision identifiers in exported run records."""
    return {
        "pump_commit": provenance["active_pumps_commit"],
        "paper_cohort": provenance.get("paper_cohort"),
        "pump_manifest_sha256": manifest_sha256,
        "model_code_commit": provenance["commit"],
        "runtime_sha256": provenance["runtime_sha256"],
    }
