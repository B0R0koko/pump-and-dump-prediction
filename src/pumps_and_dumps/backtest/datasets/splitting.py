"""Chronological model-selection splits measured in whole pump events."""

import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import COL_PUMP_HASH, COL_PUMP_TIME
from pumps_and_dumps.backtest.datasets.sample import DatasetType

MODEL_SELECTION_SPLIT_POLICY = "chronological_events_60_10_20_10_v1"
SPLIT_FRACTIONS = {
    DatasetType.TRAIN: 0.60,
    DatasetType.TUNING: 0.10,
    DatasetType.VALIDATION: 0.20,
    DatasetType.TEST: 0.10,
}


def _event_times(df: pd.DataFrame) -> pd.Series:
    """Treat naive timestamps as UTC without changing the supplied frame."""
    return pd.to_datetime(df[COL_PUMP_TIME], utc=True, format="mixed")


def _event_counts(df: pd.DataFrame, times: pd.Series) -> pd.Series:
    events = df[[COL_PUMP_HASH]].assign(**{COL_PUMP_TIME: times}).drop_duplicates()
    if events.isna().any().any():
        raise ValueError("Event identities and times must not be missing")
    if events[COL_PUMP_HASH].duplicated().any():
        raise ValueError("Each event must have exactly one timestamp")

    counts = events.groupby(COL_PUMP_TIME, sort=True).size()
    if len(counts) < len(SPLIT_FRACTIONS):
        raise ValueError("The 60/10/20/10 split requires at least four distinct event times")
    return counts


def _boundaries(counts: pd.Series) -> tuple[pd.Timestamp, ...]:
    cumulative = counts.cumsum()
    fractions = tuple(SPLIT_FRACTIONS.values())
    previous_cut = -1
    target_fraction = 0.0
    boundaries = []
    for split_index, fraction in enumerate(fractions[:-1]):
        target_fraction += fraction
        remaining_splits = len(fractions) - split_index - 1
        # Reserve at least one timestamp for every remaining partition.
        candidates = cumulative.iloc[previous_cut + 1 : len(counts) - remaining_splits]
        distances = (candidates - counts.sum() * target_fraction).abs()
        previous_cut += 1 + int(distances.argmin())
        boundaries.append(pd.Timestamp(counts.index[previous_cut + 1]))
    return tuple(boundaries)


def model_selection_boundaries(df: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    """Return the first UTC timestamp of tuning, comparison and test."""
    return _boundaries(_event_counts(df, _event_times(df)))


def split_model_selection_datasets(df: pd.DataFrame) -> dict[DatasetType, pd.DataFrame]:
    """Allocate 60/10/20/10% chronologically, keeping simultaneous events together.

    Candidate-asset counts and outcomes never influence membership. Each cut
    is nearest its cumulative event-count target among legal boundaries; ties
    go to the earlier boundary. Ratios are approximate when timestamps tie or
    rounding is necessary. Input row order and index are preserved.
    """
    times = _event_times(df)
    boundaries = _boundaries(_event_counts(df, times))
    partition = pd.Series(0, index=df.index)
    for boundary in boundaries:
        partition += times.ge(boundary).astype(int)
    return {kind: df.loc[partition.eq(index)].copy() for index, kind in enumerate(SPLIT_FRACTIONS)}
