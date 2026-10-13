"""Ordered feature catalog shared by extraction, storage and model input selection."""

from pumps_and_dumps.core.feature_type import FeatureType as F
from pumps_and_dumps.core.time_utils import NamedTimeDelta as W

REGRESSOR_OFFSETS = [
    W.FIVE_MINUTES,
    W.FIFTEEN_MINUTES,
    W.ONE_HOUR,
    W.TWO_HOURS,
    W.FOUR_HOURS,
    W.TWELVE_HOURS,
    W.ONE_DAY,
    W.TWO_DAYS,
    W.ONE_WEEK,
    W.TWO_WEEKS,
]

# Keep the original 81 predictors in a stable order across training processes.
FEATURE_WINDOWS = {
    feature: tuple(REGRESSOR_OFFSETS)
    for feature in (
        F.ASSET_RETURN,
        F.ASSET_RETURN_ZSCORE,
        F.QUOTE_ABS_ZSCORE,
        F.SHARE_OF_LONG_TRADES,
        F.POWERLAW_ALPHA,
        F.SLIPPAGE_IMBALANCE,
        F.FLOW_IMBALANCE,
        F.NUM_TRADES,
    )
}
FEATURE_WINDOWS[F.NUM_PREV_PUMP] = ()

FEATURE_SCHEMA_VERSION = "4"
FEATURE_SCHEMA_METADATA_KEY = b"pumps_and_dumps.feature_schema_version"


def regressor_columns() -> list[str]:
    return [
        column
        for feature, windows in FEATURE_WINDOWS.items()
        for column in (feature.col_names(list(windows)) if windows else [feature.lower()])
    ]
