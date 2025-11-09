import logging
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


def create_dataset() -> pd.DataFrame:
    """
    Builds dataset using features computed by feature writer
    """
    configure_logging()
    path: Path = get_root_dir() / "src/resources/pumps.json"
    pump_events: List[PumpEvent] = load_pumps(path=path)

    dfs: List[pd.DataFrame] = []
    skipped_pumps: List[PumpEvent] = []

    for pump in tqdm(pump_events, desc="Building dataset"):
        cross_section_path: Path = FEATURE_DIR / "pumps" / f"{str(pump)}.parquet"

        if not cross_section_path.exists():
            skipped_pumps.append(pump)
            continue

        df_cross_section: pd.DataFrame = pd.read_parquet(cross_section_path)

        if not (df_cross_section[COL_CURRENCY_PAIR] == pump.currency_pair.name).any():
            skipped_pumps.append(pump)
            continue

        # Add additional columns
        df_cross_section[COL_PUMP_HASH] = str(pump)
        df_cross_section[COL_PUMP_TIME] = pump.time
        df_cross_section[COL_PUMPED_CURRENCY_PAIR] = pump.currency_pair.name

        dfs.append(df_cross_section)

    df: pd.DataFrame = pd.concat(dfs)
    df = df.reset_index(drop=True)
    logging.warn("No data present for %s pumps", len(skipped_pumps))
    return df


def main():
    configure_logging()
    df: pd.DataFrame = create_dataset()
    logging.info("%s", df[FeatureType.NUM_PREV_PUMP.lower()].value_counts())


if __name__ == "__main__":
    main()
