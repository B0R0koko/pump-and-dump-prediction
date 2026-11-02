"""L2 selection must preserve the L1 fit and never inspect TEST."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.pipelines.LogisticRegression.model import LogisticRegressionModel
from backtest.utils.sample import DatasetType
from tests.notebook_helpers import analysis


def test_l2_search_is_validation_only_preserves_l1_and_breaks_ties_to_smaller_c(monkeypatch):
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(240, 3)), columns=["signal", "correlated", "noise"])
    y = pd.Series((X.signal + rng.normal(size=len(X)) > 1).astype(int))
    requests = []
    validation = SimpleNamespace(name="validation")

    def get_data(ds_type):
        requests.append(ds_type)
        assert ds_type is DatasetType.TRAIN
        return X

    def get_label(ds_type):
        requests.append(ds_type)
        assert ds_type is DatasetType.TRAIN
        return y

    def get_dataset(kind):
        requests.append(kind)
        assert kind is DatasetType.VALIDATION
        return validation

    sample = SimpleNamespace(get_data=get_data, get_label=get_label, get_dataset=get_dataset)
    params = {
        "penalty": "l1",
        "C": 0.1,
        "class_weight": {0: 1, 1: 4},
        "solver": "liblinear",
        "random_state": 42,
        "max_iter": 500,
    }
    l1 = LogisticRegressionModel(params).train(sample)
    before = l1._model.predict_proba(X).copy()
    evaluated = []

    def validation_score(model, dataset):
        assert dataset is validation
        assert model._model.penalty == "l2"
        assert model._model.class_weight == params["class_weight"]
        assert model._model.n_features_in_ == X.shape[1]
        evaluated.append(model._model.C)
        return 0.7  # All tie: the strongest regularization must win.

    monkeypatch.setattr("backtest.utils.metrics.calculate_topk_percent_auc", validation_score)
    winner, path = analysis.fit_l2_logistic_path(sample, params, [1.0, 0.01, 0.1, 0.01])
    assert evaluated == [0.01, 0.1, 1.0]
    assert winner.params["C"] == 0.01
    assert len(path) == 3
    assert set(requests) == {DatasetType.TRAIN, DatasetType.VALIDATION}
    assert l1.params["penalty"] == "l1"
    assert l1.params["C"] == 0.1
    np.testing.assert_array_equal(l1._model.predict_proba(X), before)


@pytest.mark.parametrize("values", [[], [0.0], [-1], [np.nan], [np.inf]])
def test_l2_rejects_invalid_penalties_before_accessing_data(values):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        analysis.fit_l2_logistic_path(None, {}, values)


def test_subset_robustness_refits_the_selected_l2_parameters_without_loading_l1_study(monkeypatch):
    from copy import deepcopy
    from backtest.pipelines.LogisticRegression.pipe import LogisticRegressionPipeline

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["signal", "noise"])
    y = pd.Series((X.signal > 0.7).astype(int))

    def training_data(ds_type):
        assert ds_type is DatasetType.TRAIN
        return X

    def training_labels(ds_type):
        assert ds_type is DatasetType.TRAIN
        return y

    sample = SimpleNamespace(get_data=training_data, get_label=training_labels)
    params = {"penalty": "l2", "C": 0.001, "class_weight": {0: 1, 1: 4},
              "solver": "liblinear", "random_state": 42, "max_iter": 2000, "tol": 1e-8}
    before = deepcopy(params)

    def no_search(*args, **kwargs):
        raise AssertionError("Subset robustness must not reload the historical L1 study")

    def subset_runner(pipeline_factory, **kwargs):
        assert kwargs == {"subset_fraction": 0.7, "n_runs": 25, "tuned": True, "base_seed": 42}
        model = pipeline_factory().train(sample, tuned=True)
        assert model._model.penalty == "l2"
        assert model.params == before
        assert model._model.C == before["C"]
        assert model._model.class_weight == before["class_weight"]
        # A fitted model must own its parameter mapping, even across subset fits.
        model.params["class_weight"][1] = 999
        return "subset results"

    monkeypatch.setattr(LogisticRegressionPipeline, "get_model_params", no_search)
    monkeypatch.setattr("backtest.robust.robustness.run_cross_section_subset_robustness", subset_runner)
    result = analysis.run_fixed_logistic_subset_robustness(
        params, subset_fraction=0.7, n_runs=25, tuned=True, base_seed=42,
    )
    assert result == "subset results"
    assert params == before
