import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_PUMP_HASH,
    COL_PUMP_TIME,
    COL_PUMPED_CURRENCY_PAIR,
)
from pumps_and_dumps.core.feature_type import FeatureType
from pumps_and_dumps.core.paths import get_root_dir, FEATURE_DIR
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.core.utils import configure_logging
from pumps_and_dumps.features.utils import load_pumps
from pumps_and_dumps.features.storage import validate_feature_file
from pumps_and_dumps.features.schema import regressor_columns


def _read_cross_section(pump: PumpEvent, *, feature_dir: Path | None = None) -> tuple[PumpEvent, pd.DataFrame | None]:
    feature_dir = FEATURE_DIR if feature_dir is None else feature_dir
    cross_section_path = feature_dir / "pumps" / f"{pump}.parquet"

    if not cross_section_path.exists():
        return pump, None

    validate_feature_file(cross_section_path, allow_compatible=True)
    df_cross_section: pd.DataFrame = pd.read_parquet(cross_section_path)
    # Older versioned caches contain additional predictors. Keep only the
    # active catalog, candidate identity and outcome labels for every reader.
    retained_columns = set(regressor_columns()) | {COL_CURRENCY_PAIR}
    df_cross_section = df_cross_section.loc[
        :,
        [
            column
            for column in df_cross_section.columns
            if column in retained_columns or column.startswith("target_return@")
        ],
    ]

    if not (df_cross_section[COL_CURRENCY_PAIR] == pump.currency_pair.name).any():
        return pump, None

    # Add additional columns
    df_cross_section[COL_PUMP_HASH] = str(pump)
    df_cross_section[COL_PUMP_TIME] = pump.time
    df_cross_section[COL_PUMPED_CURRENCY_PAIR] = pump.currency_pair.name
    return pump, df_cross_section


def _write_dropped_pumps(pumps: List[PumpEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([pump.as_dict() for pump in pumps], indent=2), encoding="utf-8")
    logging.info("Wrote %d dropped pump events to %s", len(pumps), path)


def create_dataset(
    max_workers: int | None = None,
    *,
    pumps_path: Path | None = None,
    feature_dir: Path | None = None,
    dropped_path: Path | None = None,
) -> pd.DataFrame:
    """
    Builds dataset using features computed by the feature writer
    """
    configure_logging()
    pumps_path = get_root_dir() / "resources/pumps.json" if pumps_path is None else pumps_path
    feature_dir = FEATURE_DIR if feature_dir is None else feature_dir
    pump_events: List[PumpEvent] = load_pumps(path=pumps_path)

    dfs: List[pd.DataFrame] = []
    skipped_pumps: List[PumpEvent] = []

    workers = min(32, max(4, (os.cpu_count() or 1) * 2)) if max_workers is None else max_workers
    if workers < 1:
        raise ValueError("max_workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(partial(_read_cross_section, feature_dir=feature_dir), pump_events)
        for pump, df_cross_section in tqdm(iterator, total=len(pump_events), desc="Building dataset"):
            if df_cross_section is None:
                skipped_pumps.append(pump)
                continue
            dfs.append(df_cross_section)

    if not dfs:
        raise ValueError(f"No usable pump-event features found in {feature_dir}")
    logging.warning("No data present for %s pumps", len(skipped_pumps))
    dropped_path = get_root_dir() / "resources/dropped_pumps.json" if dropped_path is None else dropped_path
    _write_dropped_pumps(skipped_pumps, dropped_path)
    return pd.concat(dfs, ignore_index=True)


def main():
    configure_logging()
    df: pd.DataFrame = create_dataset()
    logging.info("%s", df[FeatureType.NUM_PREV_PUMP.lower()].value_counts())


if __name__ == "__main__":
    main()
