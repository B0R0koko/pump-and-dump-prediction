import logging
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from analysis.utils.columns import *
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

    for pump in tqdm(pump_events, desc="Building dataset"):
        cross_section_path: Path = FEATURE_DIR / "pumps" / f"{str(pump)}.parquet"

        if not cross_section_path.exists():
            logging.info("No cross section found for pump %s", pump)
            continue

        df_cross_section: pd.DataFrame = pd.read_parquet(cross_section_path)

        # Add additional columns
        df_cross_section[COL_PUMP_HASH] = str(pump)
        df_cross_section[COL_PUMP_TIME] = pump.time
        df_cross_section[COL_PUMPED_CURRENCY_PAIR] = pump.currency_pair.name

        dfs.append(df_cross_section)

    df: pd.DataFrame = pd.concat(dfs)
    df = df.reset_index(drop=True)
    return df


def main():
    configure_logging()
    df: pd.DataFrame = create_dataset()
    logging.info("%s", df[COL_PUMP_HASH].nunique())


if __name__ == "__main__":
    main()
