from typing import List

import pandas as pd


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
    result = standardize_columns(df, cols_to_scale, group_col)
    result["pump_id"] = result.groupby(group_col, sort=False).ngroup()
    return result.reset_index(drop=True)


def standardize_columns(
    df: pd.DataFrame, columns: List[str], group_col: str
) -> pd.DataFrame:
    """Standardize selected columns without changing row order, indices or IDs."""
    result = df.copy()
    if not columns:
        return result
    grouped = result.groupby(group_col, sort=False)[columns]
    means = grouped.transform("mean")
    stds = grouped.transform("std").mask(lambda values: values == 0)
    nuniques = grouped.transform("nunique")
    scaled = (result[columns] - means).div(stds)
    result[columns] = scaled.mask(nuniques == 1, 0.0)
    return result
