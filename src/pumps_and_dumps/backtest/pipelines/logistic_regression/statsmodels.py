"""Untuned, unweighted maximum-likelihood logistic baseline."""

from enum import Enum

import numpy as np
import statsmodels.api as sm

from pumps_and_dumps.backtest.datasets.sample import Dataset, DatasetType, Sample
from pumps_and_dumps.backtest.pipelines.base_model import BaseModel, require_fitted


class StatsmodelsLogitModel(BaseModel):
    """Fit/scaling use TRAIN only; clustered covariance changes inference, not scores."""

    def __init__(self):
        self.params = {
            "penalty": None,
            "class_weight": None,
            "solver": "newton",
            "start_params": "zeros",
            "max_iter": 100,
            "scaling": "training mean and sample SD",
            "covariance": "pump-event clustered, small-sample correction",
        }
        self.result = None
        self.scaling = None

    def train(self, sample: Sample, *, train_type: Enum = DatasetType.TRAIN) -> "StatsmodelsLogitModel":
        if train_type.name != "TRAIN":
            raise ValueError("Statsmodels baseline must be fitted on TRAIN only")
        self.result = None
        self.scaling = None
        training = sample.get_dataset(train_type)
        X = training.get_data().astype(float)
        sd = X.std(ddof=1)
        if not np.isfinite(X.to_numpy()).all() or not np.isfinite(sd).all() or (sd <= 0).any():
            raise ValueError("Statsmodels baseline requires finite, nonconstant training predictors")
        self.scaling = X.mean().to_frame("training_mean").assign(training_sd=sd).rename_axis("feature")
        design = self._design(training)
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("Statsmodels baseline design is rank deficient")
        self.result = sm.Logit(training.get_label().astype(int), design).fit(
            start_params=np.zeros(design.shape[1]),
            method=self.params["solver"],
            maxiter=self.params["max_iter"],
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": training.all_data()["pump_hash"], "use_correction": True},
        )
        if not self.result.mle_retvals["converged"] or not np.isfinite(self.result.bse).all():
            self.result = None
            raise RuntimeError("Statsmodels baseline failed to converge with finite standard errors")
        return self

    def _design(self, dataset: Dataset):
        scaling = require_fitted(self.scaling)
        X = dataset.get_data().loc[:, scaling.index].astype(float)
        return sm.add_constant((X - scaling.training_mean) / scaling.training_sd, has_constant="add")

    def rank(self, dataset: Dataset) -> np.ndarray:
        return np.asarray(require_fitted(self.result).predict(self._design(dataset)), dtype=float)

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        positive = self.rank(dataset)
        return np.column_stack((1 - positive, positive))

    def predict(self, dataset: Dataset) -> np.ndarray:
        return (self.rank(dataset) >= 0.5).astype(int)
