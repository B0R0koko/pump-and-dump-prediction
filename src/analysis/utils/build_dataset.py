import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from analysis.utils.columns import *
from core.feature_type import FeatureType
from core.paths import get_root_dir, FEATURE_DIR
from core.pump_event import PumpEvent
from core.utils import configure_logging
from feature_writer.utils import load_pumps


def _read_cross_section(pump: PumpEvent) -> tuple[PumpEvent, pd.DataFrame | None]:
    cross_section_path: Path = FEATURE_DIR / "pumps" / f"{str(pump)}.parquet"

    if not cross_section_path.exists():
        return pump, None

    df_cross_section: pd.DataFrame = pd.read_parquet(cross_section_path)

    if not (df_cross_section[COL_CURRENCY_PAIR] == pump.currency_pair.name).any():
        return pump, None

    # Add additional columns
    df_cross_section[COL_PUMP_HASH] = str(pump)
    df_cross_section[COL_PUMP_TIME] = pump.time
    df_cross_section[COL_PUMPED_CURRENCY_PAIR] = pump.currency_pair.name
    return pump, df_cross_section


def create_dataset(max_workers: int | None = None) -> pd.DataFrame:
    """
    Builds dataset using features computed by feature writer
    """
    configure_logging()
    path: Path = get_root_dir() / "src/resources/pumps.json"
    pump_events: List[PumpEvent] = load_pumps(path=path)

    dfs: List[pd.DataFrame] = []
    skipped_pumps: List[PumpEvent] = []

    workers: int = max_workers or min(32, max(4, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_read_cross_section, pump_events)
        for pump, df_cross_section in tqdm(
            iterator, total=len(pump_events), desc="Building dataset"
        ):
            if df_cross_section is None:
                skipped_pumps.append(pump)
                continue
            dfs.append(df_cross_section)

    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    logging.warning("No data present for %s pumps", len(skipped_pumps))
    return df


def main():
    configure_logging()
    df: pd.DataFrame = create_dataset()
    logging.info("%s", df[FeatureType.NUM_PREV_PUMP.lower()].value_counts())


if __name__ == "__main__":
    main()
