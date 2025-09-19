import logging
from typing import Dict

import pandas as pd

from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.utils.feature_set import FeatureSet
from analysis.utils.sample import DatasetType, Sample


class CatboostClassifierPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    def build_model(self) -> None:
        logging.info("Building Random Forest Model")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)

        model: CatboostClassifierModel = CatboostClassifierModel()
        model.train(sample=sample)
