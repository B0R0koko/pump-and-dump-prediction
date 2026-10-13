"""Read feature importances through the model's public inspection interface."""

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.pipelines.base_model import HasFeatureImportances


def feature_importances(model: object) -> pd.Series:
    if isinstance(model, HasFeatureImportances):
        return model.get_feature_importances()

    # Keep existing notebooks using external estimator wrappers compatible.
    # Project models implement the public interface above.
    estimator = getattr(model, "_model", None)
    if estimator is None:
        estimator = model
    values = getattr(estimator, "feature_importances_", None)
    names = getattr(estimator, "feature_names_", None)
    if names is None:
        names = getattr(estimator, "feature_names_in_", None)
    if values is None or names is None:
        raise TypeError(
            f"{type(model).__name__} does not expose named feature importances"
        )
    return pd.Series(
        np.asarray(values, dtype=float),
        index=list(map(str, names)),
        name="feature_importance",
    )
