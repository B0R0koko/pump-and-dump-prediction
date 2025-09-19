import logging
from typing import Dict

import pandas as pd

from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.RandomForest.model import RandomForestModel
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent
from analysis.utils.sample import DatasetType, Sample
from core.utils import configure_logging


class RandomForestPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

    def build_model(self) -> None:
        logging.info("Building Random Forest Model")
        datasets: Dict[DatasetType, pd.DataFrame] = self.build_datasets()
        sample: Sample = Sample.from_pandas(datasets=datasets, feature_set=self.feature_set)
        model: RandomForestModel = RandomForestModel()
        model.train(sample=sample)

        topk_vals: pd.Series = calculate_topk_percent(
            model=model,
            dataset=sample.get_dataset(ds_type=DatasetType.VALIDATION),
            bins=[0.01, 0.02, 0.05, 0.1, 0.2]
        )
        logging.info(f"TopK Accuracy:\n%s", topk_vals)


if __name__ == "__main__":
    configure_logging()
    pipeline = RandomForestPipeline()
    pipeline.build_model()
