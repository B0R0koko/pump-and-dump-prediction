import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any

import optuna
import pandas as pd
from optuna import Study
from tqdm import tqdm

from analysis.pipelines.BaseModel import BaseModel
from analysis.utils.build_dataset import create_dataset
from analysis.utils.columns import COL_IS_PUMPED, COL_CURRENCY_PAIR, COL_PUMPED_CURRENCY_PAIR, COL_PUMP_TIME, \
    COL_PUMP_HASH, COL_PUMP_ID, COL_ASSET_RETURN_RANK
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import split_by_time, DatasetType, Sample
from core.feature_type import FeatureType
from core.paths import SQLITE_URL
from core.time_utils import NamedTimeDelta
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS


def cross_section_standardisation(df: pd.DataFrame) -> pd.DataFrame:
    asset_return_cols: List[str] = FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)
    asset_return_zscore_cols: List[str] = FeatureType.ASSET_RETURN_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    quote_abs_zscore_cols: List[str] = FeatureType.QUOTE_ABS_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)

    cols_to_scale: List[str] = asset_return_cols + asset_return_zscore_cols + quote_abs_zscore_cols + powerlaw_cols
    dfs: List[pd.DataFrame] = []

    for pump_hash, df_cross_section in tqdm(
            df.groupby(COL_PUMP_ID, sort=False), total=df[COL_PUMP_ID].nunique(),
            desc="Applying cross section standardisation",
    ):
        df_cross_section = df_cross_section.reset_index(drop=True)
        # Apply cross-sectional standardisation
        for col in cols_to_scale:
            # check that X is not const, otherwise we will have nans in our data
            if df_cross_section[col].nunique() == 1:
                continue
            df_cross_section[col] = (df_cross_section[col] - df_cross_section[col].mean()) / df_cross_section[col].std()

        dfs.append(df_cross_section)

    return pd.concat(dfs).reset_index(drop=True)


def remove_failed_pump_cross_sections(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Removing failed pump cross sections")
    target_return_col: str = f"target_return@{NamedTimeDelta.ONE_MINUTE.get_slug()}"
    df[COL_ASSET_RETURN_RANK] = (
        df.groupby(COL_PUMP_ID, sort=False)[target_return_col].rank(ascending=False, method="dense")
    )
    df_cross_sections: List[pd.DataFrame] = []
    pumps_removed: int = 0

    for pump_hash, df_cross_section in df.groupby(COL_PUMP_HASH, sort=False):
        pump: pd.Series = df_cross_section.loc[df_cross_section[COL_IS_PUMPED], COL_ASSET_RETURN_RANK]
        assert len(pump) == 1, "Found many pumps within one cross-section"
        pump_rank: int = pump.iloc[0]
        # if pump is not even in top-10 based on returns, then it is a failed pump, and we should remove it from
        # our analysis
        if pump_rank >= 10:
            pumps_removed += 1
            continue
        df_cross_sections.append(df_cross_section)

    df = pd.concat(df_cross_sections).reset_index(drop=True).drop(columns=[COL_ASSET_RETURN_RANK])

    if pumps_removed > 0:
        logging.warning("Removed %s failed pumps", pumps_removed)

    return df


def fillna_with_median_by_cross_section(df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    """Group by PUMP_HASH and fill missing values with cross-section median values"""
    dfs: List[pd.DataFrame] = []
    global_medians: pd.Series = df[feature_set.regressors].median()  # -> median values for the features

    for pump_hash, df_cross_section in tqdm(
            df.groupby(COL_PUMP_HASH), desc="Filling missing values with cross-sectional medians"
    ):
        # Fill regressors and numeric target with median value
        for col in feature_set.regressors:
            # if we don't have any data for the feature within the current cross-section, then fill missing values
            # with global median
            if df_cross_section[col].isna().all():
                df_cross_section[col] = df_cross_section[col].fillna(global_medians.loc[col])
            # Otherwise fill with cross-sectional median
            else:
                df_cross_section[col] = df_cross_section[col].fillna(df_cross_section[col].median())

        dfs.append(df_cross_section)

    df_nonans: pd.DataFrame = pd.concat(dfs).reset_index(drop=True)
    logging.info(
        "Nans\n%s",
        df_nonans[feature_set.regressors].isna().sum().sort_values(ascending=False)
    )
    return df_nonans


def add_col_pump_id(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_PUMP_ID] = df.groupby(by=COL_PUMP_HASH, sort=False).ngroup()
    return df


class BasePipeline(ABC):

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here"""
        df = add_col_pump_id(df=df)
        df = remove_failed_pump_cross_sections(df=df)
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
        for ds_type, dataset in datasets.items():
            logging.info("Dataset %s. Shape %s", ds_type, dataset.shape)

        return datasets

    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        logging.info("Loading parameters from %s", study_name)
        study: Study = optuna.load_study(study_name=study_name, storage=SQLITE_URL)
        return base_params | study.best_params

    @abstractmethod
    def train(self, sample: Sample, tuned: bool = True):
        ...

    @abstractmethod
    def build_model(self) -> BaseModel:
        ...

    @abstractmethod
    def optimize_parameters(self):
        ...
