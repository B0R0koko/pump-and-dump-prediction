"""Pure feature calculations; no dataset paths, file writes, or worker management.

Regressors see only trades strictly before the decision cutoff. Hourly
normalizers use full bars anchored at that cutoff; shorter windows are converted
to hourly rates. Forward-looking outcome labels are calculated separately.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

import polars as pl

from pumps_and_dumps.core.columns import PRICE, QUANTITY, TRADE_TIME
from pumps_and_dumps.core.time_utils import NamedTimeDelta
from pumps_and_dumps.features.feature_exprs import (
    compute_asset_return_zscore,
    compute_flow_imbalance,
    compute_num_trades,
    compute_powerlaw_alpha,
    compute_quote_abs_zscore,
    compute_return,
    compute_share_of_long_trades,
    compute_slippage_imbalance,
)
from pumps_and_dumps.features.provenance import AuditCallback, dependency, feature_provenance, timestamp_range
from pumps_and_dumps.features.schema import FEATURE_WINDOWS
from pumps_and_dumps.features.utils import aggregate_into_trades


def prepare_trades(df: pl.DataFrame, side: pl.Expr) -> pl.DataFrame:
    ticks = df.with_columns(quote_abs=pl.col(PRICE) * pl.col(QUANTITY), side=side).with_columns(
        quote_sign=pl.col("quote_abs") * pl.col("side"),
        quantity_sign=pl.col(QUANTITY) * pl.col("side"),
    )
    trades = aggregate_into_trades(df_ticks=ticks)
    if not trades[TRADE_TIME].is_sorted(descending=False):
        raise ValueError("Data must be in ascending order by TRADE_TIME")
    return trades.with_columns(
        quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
    ).with_columns(
        quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign(),
        price_last_prev=pl.col("price_last").shift(1),
        trade_time_prev=pl.col(TRADE_TIME).shift(1),
    )


def has_recent_activity(ticks: pl.DataFrame, cutoff: datetime, lookback: timedelta) -> bool:
    return not ticks.filter(
        pl.col(TRADE_TIME).cast(pl.Datetime("ns")).is_between(cutoff - lookback, cutoff, closed="left")
    ).is_empty()


def anchored_hourly_bars(trades: pl.DataFrame, cutoff: datetime, include_provenance: bool = False) -> pl.DataFrame:
    """Keep complete pre-decision bars even when the cutoff falls between calendar hours."""
    offset_us = (cutoff.minute * 60 + cutoff.second) * 1_000_000 + cutoff.microsecond
    aggregations = {
        "asset_return_pips": (pl.col("price_last").last() / pl.col("price_first").first() - 1) * 1e4,
        "quote_abs": pl.col("quote_abs").sum(),
    }
    if include_provenance:
        # Bar labels are artificial boundaries; audit the contributing trade
        # timestamps from the very same grouping used to calculate the bars.
        nanoseconds = pl.col(TRADE_TIME).cast(pl.Datetime("ns")).cast(pl.Int64)
        aggregations.update(
            _source_start_ns=nanoseconds.min(),
            _source_end_ns=nanoseconds.max(),
            _source_n_rows=pl.len(),
        )
    bars = (
        trades.filter(pl.col(TRADE_TIME) < cutoff)
        .group_by_dynamic(
            index_column=TRADE_TIME,
            period=timedelta(hours=1),
            every=timedelta(hours=1),
            offset=f"{offset_us}us",
        )
        .agg(**aggregations)
    )
    return bars.filter(pl.col(TRADE_TIME) + pl.duration(hours=1) <= cutoff)


@dataclass(frozen=True)
class HourlyStatistics:
    return_mean: float | None
    return_std: float | None
    quote_mean: float | None
    quote_std: float | None

    @classmethod
    def from_bars(cls, bars: pl.DataFrame) -> "HourlyStatistics":
        return cls(
            return_mean=bars.select(pl.col("asset_return_pips").mean()).item(),
            return_std=bars.select(pl.col("asset_return_pips").std()).item(),
            quote_mean=bars.select(pl.col("quote_abs").mean()).item(),
            quote_std=bars.select(pl.col("quote_abs").std()).item(),
        )


def _hourly_rate_zscore(value: float | None, mean: float | None, std: float | None, hours: float) -> float | None:
    if value is None or mean is None or std in (None, 0):
        return None
    return (value / hours - mean) / std


def _window_zscores(
    trades: pl.DataFrame,
    window_bars: pl.DataFrame,
    statistics: HourlyStatistics,
    window: NamedTimeDelta,
    asset_return: float | None,
) -> dict[str, Any]:
    hours = window.get_td().total_seconds() / 3600
    if hours < 1:
        quote_abs = trades.select(pl.col("quote_abs").sum()).item()
        return {
            "asset_return_zscore": _hourly_rate_zscore(
                asset_return, statistics.return_mean, statistics.return_std, hours
            ),
            "quote_abs_zscore": _hourly_rate_zscore(quote_abs, statistics.quote_mean, statistics.quote_std, hours),
        }
    return window_bars.select(
        compute_asset_return_zscore(statistics.return_mean, statistics.return_std).alias("asset_return_zscore"),
        compute_quote_abs_zscore(statistics.quote_mean, statistics.quote_std).alias("quote_abs_zscore"),
    ).to_dicts()[0]


def _window_features(trades: pl.DataFrame) -> dict[str, Any]:
    return trades.select(
        compute_return().alias("asset_return"),
        compute_share_of_long_trades().alias("share_of_long_trades"),
        compute_powerlaw_alpha().alias("powerlaw_alpha"),
        compute_slippage_imbalance().alias("slippage_imbalance"),
        compute_flow_imbalance().alias("flow_imbalance"),
        compute_num_trades().alias("num_trades"),
    ).to_dicts()[0]


def _trade_dependency(trades: pl.DataFrame) -> dict[str, Any]:
    start, end = timestamp_range(trades[TRADE_TIME])
    return dependency(
        "window_trades", start, end, trades.height, provenance_complete=not trades[TRADE_TIME].has_nulls()
    )


def _return_dependencies(trades: pl.DataFrame, window: dict[str, Any]) -> list[dict[str, Any]]:
    """The denominator is the prior trade, which can precede the window."""
    dependencies = [window]
    if not trades.is_empty() and trades["price_last_prev"][0] is not None:
        previous_time = (
            timestamp_range(trades["trade_time_prev"].head(1))[0] if "trade_time_prev" in trades.columns else None
        )
        dependencies.append(
            dependency("previous_price", previous_time, previous_time, 1, provenance_complete=previous_time is not None)
        )
    return dependencies


def _bar_dependency(name: str, bars: pl.DataFrame) -> dict[str, Any]:
    return dependency(
        name,
        bars["_source_start_ns"].min(),
        bars["_source_end_ns"].max(),
        int(bars["_source_n_rows"].sum()),
    )


def compute_window_features(
    trades: pl.DataFrame,
    cutoff: datetime,
    windows: Sequence[NamedTimeDelta],
    audit: AuditCallback | None = None,
) -> dict[str, Any]:
    """Calculate predictors and optionally report each feature's actual inputs.

    The audit sees the same sliced trades and hourly groups used by the
    calculation. Return-price lags and normalizer history can extend before the
    requested window, so they are listed as separate dependencies. No audit
    records or provenance aggregation columns are built when ``audit`` is None.
    """
    # Widen timestamp precision before comparing sub-millisecond cutoffs.
    past_trades = trades.with_columns(pl.col(TRADE_TIME).cast(pl.Datetime("ns"))).filter(pl.col(TRADE_TIME) < cutoff)
    hourly_bars = anchored_hourly_bars(past_trades, cutoff, include_provenance=audit is not None)
    statistics = HourlyStatistics.from_bars(hourly_bars)
    normalizer = _bar_dependency("hourly_normalizer", hourly_bars) if audit is not None else None
    features = {}
    for window in windows:
        window_trades = past_trades.filter(pl.col(TRADE_TIME) >= cutoff - window.get_td())
        window_bars = (
            hourly_bars.filter(pl.col(TRADE_TIME) >= cutoff - window.get_td())
            if window.get_td() >= timedelta(hours=1)
            else hourly_bars.head(0)
        )
        values = _window_features(window_trades)
        values.update(
            _window_zscores(
                window_trades,
                window_bars,
                statistics,
                window,
                values["asset_return"],
            )
        )
        # Keep the existing Parquet column order while sharing the feature catalog.
        window_features = {
            feature.col_name(window): values[feature.lower()]
            for feature in FEATURE_WINDOWS
            if feature.lower() in values
        }
        features.update(window_features)
        if audit is not None:
            trade_input = _trade_dependency(window_trades)
            return_inputs = _return_dependencies(window_trades, trade_input)
            bar_input = _bar_dependency("hourly_window", window_bars)
            for feature in window_features:
                kind = feature.split("@", 1)[0]
                inputs = return_inputs if kind == "asset_return" else [trade_input]
                if kind in ("asset_return_zscore", "quote_abs_zscore"):
                    if window.get_td() >= timedelta(hours=1):
                        inputs = [bar_input, normalizer]
                    else:
                        numerator = return_inputs if kind == "asset_return_zscore" else [trade_input]
                        inputs = [*numerator, normalizer]
                audit(feature_provenance(feature, "predictor", cutoff - window.get_td(), cutoff, inputs))
    return features


def compute_targets(
    trades: pl.DataFrame,
    announcement: datetime,
    windows: Sequence[NamedTimeDelta],
    audit: AuditCallback | None = None,
) -> dict[str, Any]:
    """Outcome labels deliberately use future trades and never feed the regressors."""
    trades = trades.with_columns(pl.col(TRADE_TIME).cast(pl.Datetime("ns")))
    targets = {}
    for window in windows:
        window_trades = trades.filter(
            pl.col(TRADE_TIME).is_between(announcement, announcement + window.get_td(), closed="left")
        )
        feature = f"target_return@{window.get_slug()}"
        targets[feature] = window_trades.select(compute_return()).item()
        if audit is not None:
            audit(
                feature_provenance(
                    feature,
                    "outcome",
                    announcement,
                    announcement + window.get_td(),
                    _return_dependencies(window_trades, _trade_dependency(window_trades)),
                )
            )
    return targets
