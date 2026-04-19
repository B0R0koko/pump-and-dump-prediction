from typing import List

import pandas as pd
from tqdm import tqdm


def cross_section_standardize(
    df: pd.DataFrame,
    cols_to_scale: List[str],
    group_col: str = "pump_hash",
) -> pd.DataFrame:
    """
    Apply cross-sectional z-score standardisation within each pump group.

    For every cross-section (identified by *group_col*) each column in
    *cols_to_scale* is replaced by (x - mean) / std computed over that group.

    Returns a new DataFrame with an additional ``pump_id`` column (integer
    index of the cross-section in iteration order).
    """
    dfs: List[pd.DataFrame] = []

    for i, (_pump_hash, df_cs) in tqdm(enumerate(df.groupby(group_col)), desc="Cross-section standardisation"):
        df_cs = df_cs.reset_index(drop=True)
        for col in cols_to_scale:
            df_cs[col] = (df_cs[col] - df_cs[col].mean()) / df_cs[col].std()
        df_cs["pump_id"] = i
        dfs.append(df_cs)

    return pd.concat(dfs).reset_index(drop=True)
