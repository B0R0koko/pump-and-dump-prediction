"""The coefficient report must use training rows and preserve the predictive fit."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from backtest.utils.sample import DatasetType
from pumps_and_dumps.backtest.pipelines.logistic_regression.statsmodels import StatsmodelsLogitModel
from tests.notebook_helpers import analysis


def test_logistic_report_uses_training_only_and_separates_predictive_and_explanatory_fits():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(900, 3)), columns=["signal", "small_units", "large_units"])
    X["small_units"] *= 0.1
    X["large_units"] *= 1000
    y = pd.Series(rng.binomial(1, expit(-1 + 0.8 * X.signal - 3 * X.small_units)), name="is_pumped")
    groups = pd.Series(np.repeat(np.arange(90), 10), name="pump_hash")
    estimator = LogisticRegression(solver="liblinear", penalty="l1", C=0.1, class_weight={0: 1, 1: 3}).fit(X, y)
    before = estimator.predict_proba(X).copy()
    requests = []

    def get_dataset(kind):
        requests.append(kind)
        assert kind is DatasetType.TRAIN
        return SimpleNamespace(get_data=lambda: X, get_label=lambda: y, all_data=lambda: groups.to_frame())

    experiment = SimpleNamespace(
        get_sample=lambda: SimpleNamespace(get_dataset=get_dataset),
        get_model=lambda: SimpleNamespace(_model=estimator),
        get_experiment_name=lambda: "Logistic regression + Tuned",
    )
    baseline = StatsmodelsLogitModel().train(experiment.get_sample(), train_type=DatasetType.TRAIN)
    coefficients, result, scaling, diagnostics = analysis.logistic_coefficient_report(experiment, baseline)
    assert requests == [DatasetType.TRAIN, DatasetType.TRAIN]
    assert result is baseline.result
    np.testing.assert_array_equal(estimator.predict_proba(X), before)
    np.testing.assert_array_equal(coefficients.loc[X.columns, "coefficient"], estimator.coef_[0])
    linear = X @ coefficients.loc[X.columns, "coefficient"] + coefficients.loc["intercept", "coefficient"]
    np.testing.assert_allclose(expit(linear), before[:, 1])
    np.testing.assert_allclose(coefficients.loc[X.columns, "coefficient_per_training_sd"], estimator.coef_[0] * X.std())
    np.testing.assert_allclose(result.model.exog[:, 1:], (X - scaling.training_mean) / scaling.training_sd)
    assert result.cov_type == "cluster"
    assert diagnostics["n_training_events"] == 90
    assert diagnostics["converged"]
    assert diagnostics["max_absolute_mean_score"] < 1e-7
    assert "Std.Err." in result.summary2().as_text()
    assert not np.allclose(result.predict(), before[:, 1])

    # Held-out distribution shifts and column order must not change training scaling.
    frozen_scaling = baseline.scaling.copy()
    shifted = X.iloc[:15].copy() * 4 + 10
    heldout = SimpleNamespace(get_data=lambda: shifted[shifted.columns[::-1]])
    expected = expit(
        (shifted - scaling.training_mean).div(scaling.training_sd) @ result.params.loc[X.columns]
        + result.params["const"]
    )
    np.testing.assert_allclose(baseline.rank(heldout), expected)
    np.testing.assert_allclose(baseline.predict_proba(heldout).sum(axis=1), 1)
    pd.testing.assert_frame_equal(baseline.scaling, frozen_scaling)


def test_statsmodels_baseline_rejects_nontraining_fit_and_unfitted_prediction():
    import pytest

    model = StatsmodelsLogitModel()
    with pytest.raises(ValueError, match="TRAIN only"):
        model.train(None, train_type=DatasetType.TEST)
    with pytest.raises(RuntimeError, match="fitted first"):
        model.rank(None)
