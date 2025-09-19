import logging
from typing import Dict, List

import pandas as pd
from catboost import Pool
from overrides import overrides

from analysis.pipelines.BasePipeline import BasePipeline, cross_section_standardisation
from analysis.pipelines.CatboostRanker.model import CatboostRankerModel
from analysis.utils.columns import *
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import DatasetType, Sample
from core.feature_type import FeatureType
from feature_writer.FeatureWriter import REGRESSOR_OFFSETS


class CatboostRankerPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()
        self.feature_set.target = "target_return@5MIN"

    @overrides
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define all data preprocessing steps here"""
        powerlaw_cols: List[str] = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
        df[powerlaw_cols] = df[powerlaw_cols].clip(1, 2)
        df_scaled: pd.DataFrame = cross_section_standardisation(df=df)
        df_scaled[COL_PUMP_ID] = df_scaled.groupby(COL_PUMP_HASH).ngroup()
        return df_scaled

    def build_model(self) -> None:
        logging.info("Building Random Forest Model")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()

        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        # we also need to set_pools as Catboost uses Pool under the hood
        for ds_type, dataset in sample.iter_datasets():
            dataset.set_pool(
                Pool(
                    data=dataset.get_data(),
                    label=dataset.get_label(),
                    cat_features=self.feature_set.categorical_features,
                    group_id=dataset.all_data()[COL_PUMP_ID]
                )
            )

        model: CatboostRankerModel = CatboostRankerModel()
        model.train(sample=sample)


def main():
    pipe = CatboostRankerPipeline()
    pipe.build_model()


if __name__ == "__main__":
    main()
