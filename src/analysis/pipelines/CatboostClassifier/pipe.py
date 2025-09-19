import logging
from typing import Dict

import pandas as pd
from catboost import Pool

from analysis.pipelines.BasePipeline import BasePipeline
from analysis.pipelines.CatboostClassifier.model import CatboostClassifierModel
from analysis.utils.feature_set import FeatureSet
from analysis.utils.metrics import calculate_topk_percent
from analysis.utils.sample import DatasetType, Sample
from core.utils import configure_logging


class CatboostClassifierPipeline(BasePipeline):

    def __init__(self):
        self.feature_set: FeatureSet = FeatureSet.auto()

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
    pipe = CatboostClassifierPipeline()
    pipe.build_model()


if __name__ == "__main__":
    main()
