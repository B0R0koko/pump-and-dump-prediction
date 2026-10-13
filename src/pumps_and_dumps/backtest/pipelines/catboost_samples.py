"""Adapt dataset splits to the pools required by CatBoost estimators."""

from catboost import Pool

from pumps_and_dumps.backtest.datasets.sample import Sample


def add_catboost_pools(sample: Sample, *, group_column: str | None = None) -> Sample:
    for _, dataset in sample.iter_datasets():
        group_id = None if group_column is None else dataset.all_data()[group_column]
        dataset.set_pool(
            Pool(
                data=dataset.get_data(),
                label=dataset.get_label(),
                cat_features=dataset.feature_set.categorical_features,
                group_id=group_id,
            )
        )
    return sample
