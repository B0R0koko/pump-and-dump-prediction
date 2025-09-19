import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from analysis.utils.build_dataset import create_dataset
from analysis.utils.columns import COL_IS_PUMPED, COL_CURRENCY_PAIR, COL_PUMPED_CURRENCY_PAIR, COL_PUMP_TIME, \
    COL_PUMP_HASH, COL_PUMP_ID
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import split_by_time, DatasetType
from core.feature_type import FeatureType
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS


def cross_section_standardisation(df: pd.DataFrame) -> pd.DataFrame:
    asset_return_cols: List[str] = FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)
    asset_return_zscore_cols: List[str] = FeatureType.ASSET_RETURN_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    quote_abs_zscore_cols: List[str] = FeatureType.QUOTE_ABS_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)

    cols_to_scale: List[str] = asset_return_cols + asset_return_zscore_cols + quote_abs_zscore_cols + powerlaw_cols
    dfs: List[pd.DataFrame] = []

    for i, (pump_hash, df_cross_section) in tqdm(
            enumerate(df.groupby(COL_PUMP_HASH, sort=False)),
            total=df[COL_PUMP_HASH].nunique(),
            desc="Applying cross section standardisation",
    ):
        df_cross_section = df_cross_section.reset_index(drop=True)
        # Apply cross-sectional standardisation
        for col in cols_to_scale:
            df_cross_section[col] = (df_cross_section[col] - df_cross_section[col].mean()) / df_cross_section[col].std()
        df_cross_section[COL_PUMP_ID] = i
        dfs.append(df_cross_section)

    return pd.concat(dfs).reset_index(drop=True)


def fillna_with_median(df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    for col in feature_set.regressors:
        df[col] = df[col].fillna(value=df[col].median())
    return df


class BasePipeline(ABC):

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here"""
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        return df_scaled

    def build_datasets(self) -> Dict[DatasetType, pd.DataFrame]:
        logging.info("Building dataset and preprocessing data")
        df: pd.DataFrame = create_dataset()
        df[COL_IS_PUMPED] = df[COL_CURRENCY_PAIR] == df[COL_PUMPED_CURRENCY_PAIR]  # attach binary target

        df = self.preprocess_data(df=df)
        datasets: Dict[DatasetType, pd.DataFrame] = split_by_time(
            df=df,
            time_bins=[datetime(2020, 9, 1), datetime(2021, 5, 1)],
            names=[DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST],
            time_col=COL_PUMP_TIME,
        )
        return datasets

    @abstractmethod
    def build_model(self) -> None:
        ...
