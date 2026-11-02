"""The frozen notebooks and the refactored package can coexist in one checkout."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_notebook_hides_missing_data_notices_but_preserves_other_warnings(caplog):
    import logging
    from tests.notebook_helpers import analysis

    logger = logging.getLogger()
    previous_filters = logger.filters.copy()
    try:
        analysis.silence_missing_data_logs()
        analysis.silence_missing_data_logs()
        with caplog.at_level(logging.INFO):
            logger.info("No data to get prices for %s", "EVX-BTC")
            logger.warning("No data present for %s pumps", 0)
            logger.warning("Impact fitting failed")
            logger.error("No data present for required input")
        assert "EVX-BTC" not in caplog.text
        assert "0 pumps" not in caplog.text
        assert "Impact fitting failed" in caplog.text
        assert "required input" in caplog.text
        assert sum(item.name == "notebook_missing_data" for item in logger.filters) == 1
    finally:
        logger.filters[:] = previous_filters


def test_all_historical_notebook_imports_resolve_without_changing_modern_split():
    from pumps_and_dumps.backtest.datasets import splitting as modern

    modern_split = modern.split_model_selection_datasets
    root = Path(__file__).resolve().parents[3]
    for name in ("research_notebook", "visualisations"):
        notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
        namespace = {}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{name}:{index}", "exec"), namespace)
    from backtest.pipelines.BasePipeline import TRAIN_END, VALIDATION_END
    from backtest.utils.sample import DatasetType as HistoricalDatasetType
    from core.paths import DATA_DIR as historical_data, SQLITE_URL as historical_studies
    from pumps_and_dumps.core.paths import DATA_DIR as modern_data, SQLITE_URL as modern_studies

    assert str(TRAIN_END) == "2020-09-01 00:00:00"
    assert str(VALIDATION_END) == "2021-05-01 00:00:00"
    assert not hasattr(HistoricalDatasetType, "TUNING")
    assert modern.split_model_selection_datasets is modern_split
    assert modern.MODEL_SELECTION_SPLIT_POLICY == "chronological_events_60_10_20_10_v1"
    assert historical_data == modern_data
    assert historical_studies.endswith("/studies.db")
    assert modern_studies.endswith("/studies_split_60_10_20_10_v1.db")


def test_historical_catboost_trains_from_snapshot_without_test_rows(monkeypatch):
    from backtest.pipelines.CatboostClassifierTOPKAUC import pipe
    from backtest.utils.columns import COL_IS_PUMPED, COL_PUMP_HASH
    from backtest.utils.feature_set import FeatureSet
    from backtest.utils.sample import DatasetType, Sample

    monkeypatch.setattr(
        pipe,
        "_BASE_PARAMS",
        pipe._BASE_PARAMS
        | {"num_boost_round": 8, "max_depth": 2, "verbose": False, "thread_count": 1, "allow_writing_files": False},
    )
    rng = np.random.default_rng(42)
    frames = {}
    for kind, events in ((DatasetType.TRAIN, 12), (DatasetType.VALIDATION, 5)):
        signal = rng.normal(size=(events, 20))
        frames[kind] = pd.DataFrame(
            {
                COL_PUMP_HASH: np.repeat(np.arange(events), 20),
                COL_IS_PUMPED: (np.arange(20)[None, :] == signal.argmax(axis=1)[:, None]).ravel(),
                "signal": signal.ravel(),
            }
        )
    features = FeatureSet(["signal"], COL_IS_PUMPED)
    sample = Sample.from_pandas(frames, features)
    for _, dataset in sample.iter_datasets():
        dataset.add_pool()
    pipeline = pipe.CatboostClassifierTOPKAUCPipeline()
    pipeline.feature_set = features
    model = pipeline.train(sample, tuned=False)
    assert model._model.tree_count_ > 0
    assert "TOPKAUCMetric" in model._model.get_best_score()["validation"]
    assert np.isfinite(model.rank(sample.get_dataset(DatasetType.VALIDATION))).all()
