"""Versioned feature caches with atomic writes and validated read compatibility."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
import pyarrow.parquet as pq

from pumps_and_dumps.features.schema import (
    FEATURE_SCHEMA_METADATA_KEY,
    FEATURE_SCHEMA_VERSION,
    regressor_columns,
)


def validate_feature_file(path: Path, allow_compatible: bool = False) -> None:
    schema = pq.read_schema(path)
    version = (schema.metadata or {}).get(FEATURE_SCHEMA_METADATA_KEY)
    accepted_versions = {FEATURE_SCHEMA_VERSION.encode()}
    if allow_compatible:
        # These versions contain the same original predictors and trade counts.
        # Readers discard their extra columns; new writes use the reduced schema.
        accepted_versions.update({b"2", b"3"})
    missing = set(regressor_columns()) - set(schema.names)
    if version not in accepted_versions or missing:
        raise ValueError(
            f"Outdated feature schema in {path}: expected version {FEATURE_SCHEMA_VERSION}; "
            f"{len(missing)} regressor columns missing. Regenerate with "
            "`poetry run python -m pumps_and_dumps.features.writer --skip-existing`, "
            "then restart the notebook kernel."
        )


def is_current_feature_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        validate_feature_file(path)
    except (ValueError, OSError):
        return False
    return True


def write_feature_file(frame: pl.DataFrame, path: Path) -> None:
    missing = set(regressor_columns()) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot write incomplete feature frame: missing {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Even an all-null regressor must round-trip as numeric NaN, not an object
    # column of None values that native CatBoost cannot consistently consume.
    table = frame.with_columns(pl.col(regressor_columns()).cast(pl.Float64)).to_arrow()
    metadata = (table.schema.metadata or {}) | {FEATURE_SCHEMA_METADATA_KEY: FEATURE_SCHEMA_VERSION.encode()}
    table = table.replace_schema_metadata(metadata)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
