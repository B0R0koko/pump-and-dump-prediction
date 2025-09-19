import logging
from typing import Dict, List

import pandas as pd
from catboost import Pool
from imblearn.over_sampling import SMOTE
from overrides import overrides

from analysis.pipelines.BasePipeline import BasePipeline, cross_section_standardisation, fillna_with_median
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.utils.columns import *
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent
from analysis.utils.sample import DatasetType, Sample
from core.feature_type import FeatureType
from core.utils import configure_logging
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS


class CatboostClassifierSMOTEPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here"""
        df[COL_IS_PUMPED] = df[COL_CURRENCY_PAIR] == df[COL_PUMPED_CURRENCY_PAIR]  # attach binary target
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        df_scaled = fillna_with_median(df=df_scaled, feature_set=self.feature_set)
        return df_scaled

    def apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the SMOTE algorithm to the Dataset
        """
        logging.info("Applying SMOTE")
        X, y = SMOTE().fit_resample(X=df[self.feature_set.regressors], y=df[self.feature_set.target])
        df_train: pd.DataFrame = X
        df_train[self.feature_set.target] = y
        return df_train

    def build_model(self) -> None:
        logging.info("Building Random Forest Model")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()

        df_train: pd.DataFrame = self.apply_smote(df=datasets.get(DatasetType.TRAIN))
        datasets[DatasetType.TRAIN] = df_train

        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        # we also need to set_pools as Catboost uses Pool under the hood
        for ds_type, dataset in sample.iter_datasets():
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features
                )
            )

        model: CatboostClassifierModel = CatboostClassifierModel()
        model.train(sample=sample)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.TEST),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2]
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)


def main():
    configure_logging()
    pipe = CatboostClassifierSMOTEPipeline()
    pipe.build_model()


if __name__ == "__main__":
    main()
