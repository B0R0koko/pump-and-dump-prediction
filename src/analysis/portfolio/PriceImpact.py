from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.columns import TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER


@dataclass(frozen=True)
class PriceImpactModel:
    buy_intercept_bps: float
    buy_slope_bps_per_sqrt_notional: float
    sell_intercept_bps: float
    sell_slope_bps_per_sqrt_notional: float
    buy_capacity_quote: float
    sell_capacity_quote: float
    num_bars: int

    def predict_impact_bps(self, side: int, notional_quote: float) -> float:
        x: float = float(np.sqrt(max(notional_quote, 0.0)))
        if side >= 0:
            impact = self.buy_intercept_bps + self.buy_slope_bps_per_sqrt_notional * x
        else:
            impact = self.sell_intercept_bps + self.sell_slope_bps_per_sqrt_notional * x
        return max(0.0, float(impact))

    def estimate_fill_notional(self, side: int, intended_notional_quote: float) -> float:
        if intended_notional_quote <= 0:
            return 0.0
        capacity = self.buy_capacity_quote if side >= 0 else self.sell_capacity_quote
        if not np.isfinite(capacity) or capacity <= 0:
            return float(intended_notional_quote)
        return float(min(intended_notional_quote, capacity))


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1 or np.allclose(x, x[0]):
        slope = float(np.median(y / np.maximum(x, 1e-9)))
        return 0.0, max(0.0, slope)

    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept = max(0.0, float(beta[0]))
    slope = max(0.0, float(beta[1]))
    return intercept, slope


def _fit_side_model(df_side: pd.DataFrame, fallback_capacity: float, liquidity_quantile: float) -> tuple[
    float, float, float]:
    if df_side.empty:
        return 0.0, 0.0, fallback_capacity

    x = np.sqrt(df_side["notional_quote"].to_numpy(dtype=float))
    y = df_side["impact_bps"].to_numpy(dtype=float)
    intercept, slope = _fit_linear_regression(x=x, y=y)
    capacity = float(df_side["notional_quote"].quantile(liquidity_quantile))
    if not np.isfinite(capacity) or capacity <= 0:
        capacity = fallback_capacity
    return intercept, slope, capacity


def fit_price_impact_model(
        trades: pd.DataFrame,
        bar_minutes: int = 1,
        liquidity_quantile: float = 0.9,
) -> PriceImpactModel:
    """
    Fit a side-aware (buy/sell) regression:
        impact_bps ~ intercept + slope * sqrt(executed_notional_quote)
    using historical aggregated trade bars.
    """
    if trades.empty:
        return PriceImpactModel(
            buy_intercept_bps=0.0,
            buy_slope_bps_per_sqrt_notional=0.0,
            sell_intercept_bps=0.0,
            sell_slope_bps_per_sqrt_notional=0.0,
            buy_capacity_quote=np.inf,
            sell_capacity_quote=np.inf,
            num_bars=0,
        )

    df = trades[[TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER]].copy()
    df[TRADE_TIME] = pd.to_datetime(df[TRADE_TIME])
    df = df.sort_values(TRADE_TIME).reset_index(drop=True)

    # Aggressor buy -> is_buyer_maker=False, aggressor sell -> True
    df["side"] = np.where(df[IS_BUYER_MAKER].astype(bool), -1, 1)
    df["notional_quote_trade"] = df[PRICE].astype(float) * df[QUANTITY].astype(float)
    df["bucket_time"] = df[TRADE_TIME].dt.floor(f"{bar_minutes}min")

    grouped = (
        df.groupby(["bucket_time", "side"], sort=True)
        .agg(
            open_price=(PRICE, "first"),
            sum_quantity=(QUANTITY, "sum"),
            notional_quote=("notional_quote_trade", "sum"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["sum_quantity"] > 0].copy()
    if grouped.empty:
        return PriceImpactModel(
            buy_intercept_bps=0.0,
            buy_slope_bps_per_sqrt_notional=0.0,
            sell_intercept_bps=0.0,
            sell_slope_bps_per_sqrt_notional=0.0,
            buy_capacity_quote=np.inf,
            sell_capacity_quote=np.inf,
            num_bars=0,
        )

    grouped["vwap_price"] = grouped["notional_quote"] / grouped["sum_quantity"]
    grouped["impact_bps"] = grouped["side"] * (grouped["vwap_price"] / grouped["open_price"] - 1.0) * 1e4
    grouped["impact_bps"] = grouped["impact_bps"].clip(lower=0.0)

    fallback_capacity: float = float(grouped["notional_quote"].quantile(liquidity_quantile))
    if not np.isfinite(fallback_capacity) or fallback_capacity <= 0:
        fallback_capacity = np.inf

    buy_df = grouped[grouped["side"] == 1]
    sell_df = grouped[grouped["side"] == -1]

    buy_intercept, buy_slope, buy_capacity = _fit_side_model(
        df_side=buy_df, fallback_capacity=fallback_capacity, liquidity_quantile=liquidity_quantile
    )
    sell_intercept, sell_slope, sell_capacity = _fit_side_model(
        df_side=sell_df, fallback_capacity=fallback_capacity, liquidity_quantile=liquidity_quantile
    )

    return PriceImpactModel(
        buy_intercept_bps=buy_intercept,
        buy_slope_bps_per_sqrt_notional=buy_slope,
        sell_intercept_bps=sell_intercept,
        sell_slope_bps_per_sqrt_notional=sell_slope,
        buy_capacity_quote=buy_capacity,
        sell_capacity_quote=sell_capacity,
        num_bars=int(grouped.shape[0]),
    )
