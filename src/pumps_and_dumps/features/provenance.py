"""Timestamp-only provenance for feature inputs, with no logging or file IO.

Bounds describe rows actually passed to a calculation, including dependencies
outside its nominal window. Integer nanoseconds are retained for comparisons;
the UTC strings are intended for humans inspecting an audit log.
"""

from typing import Any, Callable

import pandas as pd
import polars as pl

AuditCallback = Callable[[dict[str, Any]], None]


def timestamp_ns(value: Any) -> int | None:
    """Return epoch nanoseconds, treating naive exchange timestamps as UTC."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, int):
        return value
    return int(pd.Timestamp(value).value)


def utc_timestamp(value: Any) -> str | None:
    """Format UTC timestamps without rounding away a boundary nanosecond."""
    nanoseconds = timestamp_ns(value)
    if nanoseconds is None:
        return None
    seconds, fraction = divmod(nanoseconds, 1_000_000_000)
    return f"{pd.Timestamp(seconds, unit='s', tz='UTC'):%Y-%m-%dT%H:%M:%S}.{fraction:09d}Z"


def timestamp_range(series: pl.Series) -> tuple[int | None, int | None]:
    """Find actual extrema without Polars' conversion to Python microseconds."""
    nanoseconds = series.cast(pl.Datetime("ns")).cast(pl.Int64)
    return nanoseconds.min(), nanoseconds.max()


def dependency(
    name: str,
    start_ns: int | None,
    end_ns: int | None,
    n_rows: int,
    *,
    provenance_complete: bool = True,
) -> dict[str, Any]:
    """Describe one shared input slice, or an untraceable lag dependency."""
    return {
        "name": name,
        "data_start": utc_timestamp(start_ns),
        "data_end": utc_timestamp(end_ns),
        "data_start_ns": start_ns,
        "data_end_ns": end_ns,
        "n_rows": n_rows,
        "provenance_complete": provenance_complete,
    }


def feature_provenance(
    feature: str,
    role: str,
    window_start: Any,
    window_end_exclusive: Any,
    dependencies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Union input bounds while keeping absent provenance explicitly unknown."""
    starts = [item["data_start_ns"] for item in dependencies if item["data_start_ns"] is not None]
    ends = [item["data_end_ns"] for item in dependencies if item["data_end_ns"] is not None]
    start, end = min(starts, default=None), max(ends, default=None)
    return {
        "feature": feature,
        "role": role,
        "window_start": utc_timestamp(window_start),
        "window_end_exclusive": utc_timestamp(window_end_exclusive),
        "data_start": utc_timestamp(start),
        "data_end": utc_timestamp(end),
        "data_start_ns": start,
        "data_end_ns": end,
        # A sink may enrich records without changing other features' inputs.
        "dependencies": [dict(item) for item in dependencies],
        "provenance_complete": all(item["provenance_complete"] for item in dependencies),
    }
