import numpy as np
import pandas as pd

from backtest.pipelines.BasePipeline import cross_section_standardisation
from backtest.utils.columns import COL_PUMP_ID
from core.feature_type import FeatureType
from features.FeatureWriter import REGRESSOR_OFFSETS


def _scale_columns() -> list[str]:
    return (
        FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)
        + FeatureType.ASSET_RETURN_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
        + FeatureType.QUOTE_ABS_ZSCORE.col_names(offsets=REGRESSOR_OFFSETS)
        + FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
    )


def test_cross_section_standardisation_skips_constant_features_without_dividing_by_zero() -> None:
    cols = _scale_columns()
    rows: list[dict[str, float]] = []

    for row_idx in range(4):
        pump_id = 0 if row_idx < 2 else 1
        row: dict[str, float] = {COL_PUMP_ID: float(pump_id)}
        for col in cols:
            if pump_id == 0:
                row[col] = 5.0
            else:
                row[col] = 1.0 if row_idx == 2 else 3.0
        rows.append(row)

    df = pd.DataFrame(rows)

    scaled = cross_section_standardisation(df=df)
    representative_col = cols[0]

    # Constant cross-section should be preserved exactly instead of raising.
    assert np.allclose(
        scaled.loc[scaled[COL_PUMP_ID] == 0, representative_col].to_numpy(),
        df.loc[df[COL_PUMP_ID] == 0, representative_col].to_numpy(),
    )

    # Variable cross-section should still be standardized.
    varying_values = scaled.loc[scaled[COL_PUMP_ID] == 1, representative_col].to_numpy()
    assert np.isclose(varying_values.mean(), 0.0)
    assert np.isclose(varying_values[0], -varying_values[1])
