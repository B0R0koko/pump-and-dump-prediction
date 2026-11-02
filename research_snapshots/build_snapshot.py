"""Rebuild the historical runtime and the exact cohort in the 0167ddd paper."""

import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

COMMIT = "73ba716dfe39be3f2da9b0ccd7d6663ff544b570"
PAPER_COMMIT = "0167dddfa6b55819e276812836ac7537c5d3a00b"
PACKAGES = {"backtest", "core", "features"}
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research_snapshots" / "73ba716"
COHORT = ROOT / "research_snapshots" / "paper_0167ddd"


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def main():
    manifest = git_bytes("resources/pumps.json")
    exclusions = git_bytes("resources/dropped_pumps.json")
    events = json.loads(manifest)
    dropped = json.loads(exclusions)

    def event_key(event):
        return (
            event.get("symbol", event.get("currency_pair")),
            event["exchange"],
            datetime.fromisoformat(event["time"]),
        )

    excluded = {event_key(event) for event in dropped}
    if not excluded <= {event_key(event) for event in events}:
        raise ValueError("Historical exclusions contain events absent from the source manifest")
    retained = [event for event in events if event_key(event) not in excluded]
    active_manifest = (json.dumps(retained, indent=2) + "\n").encode()
    paper_blobs = [
        subprocess.check_output(["git", "rev-parse", f"{commit}:paper/access.pdf"], cwd=ROOT, text=True).strip()
        for commit in (COMMIT, PAPER_COMMIT)
    ]
    if paper_blobs[0] != paper_blobs[1]:
        raise ValueError("The source cohort's paper differs from the requested paper version")
    COHORT.mkdir(parents=True, exist_ok=True)
    (COHORT / "pumps.json").write_bytes(active_manifest)
    (COHORT / "excluded_pumps.json").write_bytes(exclusions)
    cohort_provenance = {
        "paper_commit": PAPER_COMMIT,
        "paper_pdf_git_blob": paper_blobs[0],
        "source_commit": COMMIT,
        "source_manifest_path": "research_snapshots/73ba716/pumps.json",
        "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "historical_exclusions_path": "research_snapshots/paper_0167ddd/excluded_pumps.json",
        "historical_exclusions_sha256": hashlib.sha256(exclusions).hexdigest(),
        "construction": "Original 498-event manifest minus the 25 recorded unavailable events; preserve original order",
        "source_events": len(events),
        "excluded_events": len(excluded),
        "retained_events": len(retained),
        "manifest_path": "research_snapshots/paper_0167ddd/pumps.json",
        "manifest_sha256": hashlib.sha256(active_manifest).hexdigest(),
    }
    (COHORT / "provenance.json").write_text(json.dumps(cohort_provenance, indent=2) + "\n")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", COMMIT], cwd=ROOT, text=True).splitlines()
    records = []
    archive_path = OUTPUT / "runtime.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.split("/")[0] not in PACKAGES or not path.endswith(".py"):
                continue
            if path in {"backtest/utils/graphs.py", "backtest/utils/impact_example.py", "features/feature_exprs.py"}:
                continue
            original = git_bytes(path)
            content = original
            changes = []
            if path == "features/FeatureWriter.py":
                offsets = next(
                    node
                    for node in ast.parse(original).body
                    if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "REGRESSOR_OFFSETS"
                )
                content = (
                    "from typing import List\nfrom core.time_utils import NamedTimeDelta\n\n"
                    + ast.get_source_segment(original.decode(), offsets)
                    + "\n"
                ).encode()
                changes.append("Retain only original model feature offsets; data generation uses the current installer")
            elif path == "core/paths.py":
                content = (
                    '"""Historical research paths with the current installer data directory."""\n'
                    "from pumps_and_dumps.core.paths import (\n"
                    "    get_root_dir, DATA_DIR, RAW_DATA_DIR, TRANSFORMED_DATA_DIR, FEATURE_DIR,\n"
                    "    BINANCE_SPOT_RAW_TRADES, BINANCE_SPOT_RAW_KLINES, BINANCE_SPOT_HIVE_TRADES,\n"
                    ")\n"
                    "SQLITE_URL = f\"sqlite:///{DATA_DIR / 'studies.db'}\"\n"
                ).encode()
                changes.append("Use the installer's configured data directory and the original studies.db")
            elif path == "backtest/utils/build_dataset.py":
                source = content.decode()
                source = source.replace(
                    'get_root_dir() / "resources/pumps.json"',
                    'get_root_dir() / "research_snapshots/paper_0167ddd/pumps.json"',
                )
                source = source.replace(
                    'get_root_dir() / "resources/dropped_pumps.json"',
                    'get_root_dir() / "notebooks/analysis_outputs/research_73ba716/dropped_pumps.json"',
                )
                content = source.encode()
                changes.append(
                    "Use the exact 473-event historical paper cohort and isolate generated missing-event reports"
                )
            elif path == "backtest/portfolio/BasePortfolio.py":
                source = content.decode()
                source = source.replace(
                    "self._hive: pl.LazyFrame = pl.scan_parquet(Exchange.BINANCE_SPOT.get_hive_location(), hive_partitioning=True)",
                    "from pumps_and_dumps.backtest.portfolio.trades import HiveTradeSource\n"
                    "        self._trade_source = HiveTradeSource()",
                )
                node = next(
                    member
                    for item in ast.parse(source).body
                    if isinstance(item, ast.ClassDef)
                    for member in item.body
                    if isinstance(member, ast.FunctionDef) and member.name == "load_trades"
                )
                lines = source.splitlines(keepends=True)
                lines[node.lineno - 1 : node.end_lineno] = [
                    "    def load_trades(self, bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:\n"
                    "        return self._trade_source.load_trades(bounds, currency_pair)\n"
                ]
                content = "".join(lines).encode()
                changes.append("Share partitioned trade reads with the refactored pipeline; preserve execution logic")
            elif path == "backtest/portfolio/TOPKPortfolio.py":
                source = content.decode()
                warning = next(
                    node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "warning"
                    and node.value.args
                    and isinstance(node.value.args[0], ast.Constant)
                    and str(node.value.args[0].value).startswith("Manipulated impact model has only")
                )
                lines = source.splitlines(keepends=True)
                del lines[warning.lineno - 1 : warning.end_lineno]
                content = "".join(lines).encode()
                changes.append("Silence the expected small-sample impact fallback; preserve execution and error warnings")
            info = zipfile.ZipInfo(path, date_time=(2026, 8, 16, 13, 5, 6))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
            records.append(
                {
                    "path": path,
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "runtime_sha256": hashlib.sha256(content).hexdigest(),
                    "adaptations": changes,
                }
            )
    (OUTPUT / "pumps.json").write_bytes(manifest)
    (OUTPUT / "provenance.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "runtime_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "pumps_sha256": hashlib.sha256(manifest).hexdigest(),
                "active_pumps_commit": COMMIT,
                "active_pumps_path": cohort_provenance["manifest_path"],
                "active_pumps_sha256": hashlib.sha256(active_manifest).hexdigest(),
                "paper_cohort": cohort_provenance,
                "files": records,
                "purpose": "Preserve historical model/split code and the exact 473-event cohort analyzed in the 0167ddd paper",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Archived {len(records)} historical modules in {archive_path}")


if __name__ == "__main__":
    main()
