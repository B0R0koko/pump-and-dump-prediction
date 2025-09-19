import logging
from typing import Dict

import pandas as pd
from catboost import Pool

from analysis.pipelines.BaseModel import BaseModel
from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import DatasetType, Sample
from core.utils import configure_logging


class CatboostClassifierPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    def build_model(self) -> BaseModel:
        logging.info("Building Random Forest Model")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
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
        return model


def main():
    configure_logging()
    pipe = CatboostClassifierPipeline()
    pipe.build_model()


if __name__ == "__main__":
    main()
