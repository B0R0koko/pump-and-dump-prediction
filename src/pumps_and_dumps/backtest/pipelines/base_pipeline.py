import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from functools import partial
from typing import List, Dict, Any

import optuna
import pandas as pd
from optuna import Study

from pumps_and_dumps.backtest.pipelines.base_model import BaseModel, ImplementsRank
from pumps_and_dumps.backtest.pipelines.study import create_study, validate_study_split
from pumps_and_dumps.backtest.datasets.build_dataset import create_dataset
from pumps_and_dumps.backtest.datasets.columns import (
    COL_IS_PUMPED,
    COL_CURRENCY_PAIR,
    COL_PUMPED_CURRENCY_PAIR,
    COL_PUMP_HASH,
    COL_PUMP_ID,
)
from pumps_and_dumps.backtest.datasets.feature_set import FeatureSet
from pumps_and_dumps.backtest.datasets.preprocessing import standardize_columns
from pumps_and_dumps.backtest.datasets.sample import (
    DatasetType,
    Sample,
)
from pumps_and_dumps.backtest.datasets.splitting import (
    MODEL_SELECTION_SPLIT_POLICY,
    split_model_selection_datasets,
)
from pumps_and_dumps.backtest.metrics import calculate_topk_percent
from pumps_and_dumps.core.feature_type import FeatureType
from pumps_and_dumps.core.paths import SQLITE_URL
from pumps_and_dumps.features.schema import REGRESSOR_OFFSETS

_RAW_DATASET_CACHE: pd.DataFrame | None = None
_PREPROCESSED_DATASETS_CACHE: dict[tuple, Dict[DatasetType, pd.DataFrame]] = {}

def _copy_datasets(
    datasets: Dict[DatasetType, pd.DataFrame],
) -> Dict[DatasetType, pd.DataFrame]:
    return {ds_type: dataset.copy(deep=True) for ds_type, dataset in datasets.items()}


def _get_raw_dataset_cached() -> pd.DataFrame:
    global _RAW_DATASET_CACHE

    if _RAW_DATASET_CACHE is None:
        logging.info("Building raw dataset from feature files")
        _RAW_DATASET_CACHE = _with_pump_labels(create_dataset())
    else:
        logging.info("Using cached raw dataset")

    return _RAW_DATASET_CACHE.copy(deep=True)


def _with_pump_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    df[COL_IS_PUMPED] = df[COL_CURRENCY_PAIR] == df[COL_PUMPED_CURRENCY_PAIR]
    return df


def cross_section_standardisation(df: pd.DataFrame) -> pd.DataFrame:
    asset_return_cols: List[str] = FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)
    asset_return_zscore_cols: List[str] = FeatureType.ASSET_RETURN_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    quote_abs_zscore_cols: List[str] = FeatureType.QUOTE_ABS_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
    powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)

    cols_to_scale: List[str] = asset_return_cols + asset_return_zscore_cols + quote_abs_zscore_cols + powerlaw_cols
    # A constant feature contains no within-event ranking information, so its
    # mathematically neutral standardized value is zero. Keeping the raw level
    # would silently reintroduce cross-event scale into a transform advertised
    # as cross-sectional. All-NaN columns remain NaN for estimators with native
    # missing-value handling (and are imputed later for sklearn/SMOTE paths).
    return standardize_columns(df, cols_to_scale, COL_PUMP_ID)


def fillna_with_median_by_cross_section(
    df: pd.DataFrame, feature_set: FeatureSet, *, train_events: Collection[str]
) -> pd.DataFrame:
    """Group by PUMP_HASH and fill missing values with cross-section median values.

    The global fallback medians used when an entire cross-section is NaN are computed
    from explicitly supplied training events only. Using train rows as
    the source of the global prior avoids leaking validation/test information into
    the imputation step.

    Some regressor columns can be entirely NaN across the whole panel when the
    required source history is absent. When that happens both the
    cross-section median and the global-train median are NaN. To keep the
    downstream sklearn estimators happy (they refuse NaN inputs) we fall back
    to 0.0 for the residual, which is neutral under the downstream cross-
    sectional standardisation (Eq. ~cs\\_standardization) — a filled-with-mean
    column becomes a zero column after standardisation and cannot inject
    signal.
    """
    regressors: List[str] = feature_set.regressors
    train_mask: pd.Series = df[COL_PUMP_HASH].isin(train_events)
    train_source: pd.DataFrame = df.loc[train_mask, regressors]
    # Never substitute validation/test rows when a caller supplies a frame with
    # no training period. In that edge case, unresolved missing values fall
    # through to the neutral 0.0 fallback below.
    global_medians: pd.Series = (
        train_source.median() if not train_source.empty else pd.Series(index=regressors, dtype=float)
    )
    cross_section_medians: pd.DataFrame = df.groupby(COL_PUMP_HASH, sort=False)[regressors].transform("median")

    df_nonans: pd.DataFrame = df.copy()
    df_nonans[regressors] = df_nonans[regressors].fillna(cross_section_medians).fillna(global_medians).fillna(0.0)

    logging.info("Nans\n%s", df_nonans[regressors].isna().sum().sort_values(ascending=False))
    return df_nonans


def impute_training_features(df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    """Fit missing-value fallback medians on the chronological training split."""
    train = split_model_selection_datasets(df)[DatasetType.TRAIN]
    return fillna_with_median_by_cross_section(
        df, feature_set, train_events=train[COL_PUMP_HASH].unique()
    )


def add_col_pump_id(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_PUMP_ID] = df.groupby(by=COL_PUMP_HASH, sort=False).ngroup()
    return df


def prepare_cross_section_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach event groups and clip power-law features before model-specific imputation."""
    df = add_col_pump_id(df)
    powerlaw_cols = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
    df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
    return df


class BasePipeline(ABC):
    def __init__(
        self,
        feature_set: FeatureSet | None = None,
        *,
        dataset_loader: Callable[[], pd.DataFrame] | None = None,
        study_storage: str | None = None,
    ):
        self.feature_set = FeatureSet.auto() if feature_set is None else feature_set
        self._dataset_loader = dataset_loader
        self.study_storage = SQLITE_URL if study_storage is None else study_storage

    def create_sample(self) -> Sample:
        return self.create_sample_from_datasets(self.build_datasets())

    def create_sample_from_datasets(self, datasets: Dict[DatasetType, pd.DataFrame]) -> Sample:
        """Prepare supplied splits without rebuilding or resplitting the dataset."""
        return Sample.from_pandas(datasets=_copy_datasets(datasets), feature_set=self.feature_set)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define past-only preprocessing steps shared by all dataset splits."""
        return cross_section_standardisation(prepare_cross_section_features(df))

    def build_datasets(self) -> Dict[DatasetType, pd.DataFrame]:
        cache_key = (
            type(self),
            tuple(self.feature_set.regressors),
            self.feature_set.target,
            MODEL_SELECTION_SPLIT_POLICY,
        )
        use_cache = self._dataset_loader is None
        if use_cache and cache_key in _PREPROCESSED_DATASETS_CACHE:
            logging.info("Using cached preprocessed datasets for %s", cache_key)
            return _copy_datasets(_PREPROCESSED_DATASETS_CACHE[cache_key])

        logging.info("Building dataset and preprocessing data")
        df = self._load_raw_dataset()
        df = self.preprocess_data(df=df)
        datasets = split_model_selection_datasets(df)
        for ds_type, dataset in datasets.items():
            logging.info("Dataset %s. Shape %s", ds_type, dataset.shape)

        if use_cache:
            _PREPROCESSED_DATASETS_CACHE[cache_key] = _copy_datasets(datasets)
        return _copy_datasets(datasets)

    def _load_raw_dataset(self) -> pd.DataFrame:
        if self._dataset_loader is None:
            return _get_raw_dataset_cached()
        return _with_pump_labels(self._dataset_loader())

    def get_model_params(self, base_params: Dict[str, Any], study_name: str) -> Dict[str, Any]:
        logging.info("Loading parameters from %s", study_name)
        study: Study = optuna.load_study(study_name=study_name, storage=self.study_storage)
        validate_study_split(study)
        return base_params | study.best_params

    @abstractmethod
    def train(self, sample: Sample, tuned: bool = True) -> BaseModel: ...

    def build_model(self, tuned: bool = True) -> BaseModel:
        """Train a model and report development accuracy on the tuning split."""
        sample = self.create_sample()
        model = self.train(sample=sample, tuned=tuned)
        topk_values = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(DatasetType.TUNING),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2],
        )
        logging.info("Tuning Top@k%% accuracy:\n%s", topk_values)
        return model

    @abstractmethod
    def optimize_parameters(self, n_trials: int = 100): ...

    def optimize_portfolio_strategy(self) -> None:
        """Tune portfolio timing / Top@k size against the TUNING split.

        The Optuna objective :func:`portfolio_pnl_objective` evaluates each candidate
        configuration on ``DatasetType.TUNING``. The independent VALIDATION and
        TEST splits are never touched during tuning.
        """
        from pumps_and_dumps.backtest.portfolio.topk import portfolio_pnl_objective

        sample: Sample = self.create_sample()
        model: ImplementsRank = self.train(sample=sample)
        study: Study = create_study(
            study_name="TOPKPortfolioStrategy",
            start_new=False,
            storage=self.study_storage,
        )
        study.optimize(partial(portfolio_pnl_objective, model=model, sample=sample), n_trials=20)
