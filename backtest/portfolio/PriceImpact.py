from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME


def _fit_sqrt_regression(notionals: np.ndarray, impacts: np.ndarray) -> tuple[float, float]:
    """Constrained OLS for I(Q) = max(0, a + beta * sqrt(Q)).

    The intercept a is unconstrained (negative = dead zone for small Q),
    while beta is constrained >= 0 (impact increases with size).
    """
    if len(notionals) == 0:
        return 0.0, 0.0
    X = np.column_stack([np.ones(len(notionals)), np.sqrt(notionals)])
    result = lsq_linear(X, impacts, bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    return float(result.x[0]), float(result.x[1])


@dataclass(frozen=True)
class PriceImpactModel:
    """
    Square-root market-impact model: I(Q) = max(0, a + beta * sqrt(Q)).

    Regression is fitted in USDT-normalised notional space so the impact
    curve is comparable across time (BTC price regimes). The public interface
    accepts notionals in quote currency; the model converts internally using
    the stored ``quote_to_usdt`` rate from fitting time.

    The intercept a can be negative, creating a dead zone for small orders
    that execute within the spread. beta >= 0 is enforced during fitting.
    """

    buy_beta0: float
    buy_beta1: float
    sell_beta0: float
    sell_beta1: float
    buy_capacity_usdt: float
    sell_capacity_usdt: float
    quote_to_usdt: float
    num_trades: int

    def predict_impact_bps(self, side: int, notional_quote: float) -> float:
        """Terminal impact: I(Q) = max(0, a + beta * sqrt(Q_usdt))."""
        if notional_quote <= 0:
            return 0.0
        q_usdt = notional_quote * self.quote_to_usdt
        a, b = (self.buy_beta0, self.buy_beta1) if side >= 0 else (self.sell_beta0, self.sell_beta1)
        return max(0.0, a + b * np.sqrt(q_usdt))

    def _impact_threshold_usdt(self, a: float, b: float) -> float:
        """USDT notional below which impact is zero: Q* = (a/b)^2 when a < 0."""
        if a >= 0 or b <= 0:
            return 0.0
        return (a / b) ** 2

    def predict_vwap_impact_bps(self, side: int, notional_quote: float) -> float:
        """
        VWAP impact accounting for the dead zone, in USDT space.

        I_vwap(Q) = (1/Q) * integral from Q* to Q of (a + beta*sqrt(q)) dq
                   = (1/Q) * [a*(Q - Q*) + 2/3*beta*(Q^1.5 - Q*^1.5)]

        where Q* = (a/beta)^2 is the USDT threshold below which impact is zero.
        """
        if notional_quote <= 0:
            return 0.0
        q_usdt = notional_quote * self.quote_to_usdt
        a, b = (self.buy_beta0, self.buy_beta1) if side >= 0 else (self.sell_beta0, self.sell_beta1)
        q_star = self._impact_threshold_usdt(a, b)
        if q_usdt <= q_star:
            return 0.0
        integral = a * (q_usdt - q_star) + (2.0 / 3.0) * b * (q_usdt**1.5 - q_star**1.5)
        return max(0.0, integral / q_usdt)

    def estimate_fill_notional(self, side: int, intended_notional_quote: float) -> float:
        """Cap intended notional by side-specific USDT liquidity capacity."""
        if intended_notional_quote <= 0:
            return 0.0
        intended_usdt = intended_notional_quote * self.quote_to_usdt
        capacity_usdt = self.buy_capacity_usdt if side >= 0 else self.sell_capacity_usdt
        if not np.isfinite(capacity_usdt) or capacity_usdt <= 0:
            return float(intended_notional_quote)
        filled_usdt = min(intended_usdt, capacity_usdt)
        return float(filled_usdt / self.quote_to_usdt)

    def estimate_vwap_price(self, base_price: float, side: int, notional_quote: float) -> tuple[float, float]:
        """
        VWAP execution price (Eqs. 10-11).

        Entry (side=1):  p_vwap = p * (1 + I_vwap / 1e4)
        Exit  (side=-1): p_vwap = p * (1 - I_vwap / 1e4)
        """
        if notional_quote <= 0:
            return max(1e-12, float(base_price)), 0.0
        vwap_impact_bps = self.predict_vwap_impact_bps(side=side, notional_quote=notional_quote)
        vwap_price = max(1e-12, float(base_price) * (1.0 + side * vwap_impact_bps / 1e4))
        return vwap_price, vwap_impact_bps


@dataclass(frozen=True)
class PriceImpactFitResult:
    model: PriceImpactModel
    samples: pd.DataFrame
    diagnostics: pd.DataFrame


def _empty_price_impact_model(quote_to_usdt: float = 1.0) -> PriceImpactModel:
    return PriceImpactModel(
        buy_beta0=0.0,
        buy_beta1=0.0,
        sell_beta0=0.0,
        sell_beta1=0.0,
        buy_capacity_usdt=np.inf,
        sell_capacity_usdt=np.inf,
        quote_to_usdt=quote_to_usdt,
        num_trades=0,
    )


_IMPACT_SAMPLE_COLUMNS = [
    "trade_time",
    "price_first",
    "price_last",
    "notional_quote",
    "notional_usdt",
    "side",
    "impact_bps",
]


def _empty_impact_samples() -> pd.DataFrame:
    return pd.DataFrame(columns=_IMPACT_SAMPLE_COLUMNS)


def aggregate_trades_to_orders(trades: pd.DataFrame, quote_to_usdt: float = 1.0) -> pd.DataFrame:
    """
    Aggregate fills sharing the same trade_time into meta-orders.

    Each group represents fills from a single market order walking through
    the order book.  We record the first and last fill price, total notional
    (in both quote and USDT), and net side (buy vs sell initiated).
    """
    if trades.empty:
        return _empty_impact_samples()

    required = {TRADE_TIME, PRICE, QUANTITY, IS_BUYER_MAKER}
    if not required.issubset(trades.columns):
        return _empty_impact_samples()

    df = trades.copy()
    df[PRICE] = pd.to_numeric(df[PRICE], errors="coerce")
    df[QUANTITY] = pd.to_numeric(df[QUANTITY], errors="coerce")
    df = df.dropna(subset=[TRADE_TIME, PRICE, QUANTITY])
    if df.empty:
        return _empty_impact_samples()

    df["quote_notional"] = df[PRICE] * df[QUANTITY]
    # is_buyer_maker=True means taker sell, False means taker buy
    df["signed_notional"] = np.where(df[IS_BUYER_MAKER], -df["quote_notional"], df["quote_notional"])

    grouped = df.groupby(TRADE_TIME, sort=True).agg(
        price_first=(PRICE, "first"),
        price_last=(PRICE, "last"),
        total_quote_notional=("quote_notional", "sum"),
        net_signed_notional=("signed_notional", "sum"),
    )
    grouped = grouped[grouped["total_quote_notional"] > 0].copy()
    if grouped.empty:
        return _empty_impact_samples()

    grouped["side"] = np.where(grouped["net_signed_notional"] >= 0, 1, -1)
    grouped["notional_quote"] = grouped["total_quote_notional"]
    grouped["notional_usdt"] = grouped["notional_quote"] * quote_to_usdt
    grouped["impact_bps"] = grouped["side"] * (grouped["price_last"] / grouped["price_first"] - 1.0) * 1e4
    grouped["impact_bps"] = grouped["impact_bps"].clip(lower=0.0)

    result = grouped.reset_index().rename(columns={TRADE_TIME: "trade_time"})
    return result[_IMPACT_SAMPLE_COLUMNS].reset_index(drop=True)


def _fit_side(
    df_side: pd.DataFrame,
    liquidity_quantile: float,
    fallback_capacity_usdt: float,
) -> tuple[float, float, float, dict]:
    """
    Fit sqrt regression for one side in USDT space: I(Q) = max(0, a + beta * sqrt(Q_usdt)).

    Returns (beta0, beta1, capacity_usdt, diagnostics).
    """
    side_name = "buy" if not df_side.empty and df_side["side"].iloc[0] >= 0 else "sell"
    if df_side.empty or df_side["notional_usdt"].max() <= 0:
        return 0.0, 0.0, fallback_capacity_usdt, _side_diagnostics(side_name, 0, 0.0, 0.0)

    notionals_usdt = df_side["notional_usdt"].to_numpy(dtype=float)
    impacts = df_side["impact_bps"].to_numpy(dtype=float)

    valid = notionals_usdt > 0
    notionals_usdt = notionals_usdt[valid]
    impacts = impacts[valid]

    if len(notionals_usdt) == 0:
        return 0.0, 0.0, fallback_capacity_usdt, _side_diagnostics(side_name, 0, 0.0, 0.0)

    beta0, beta1 = _fit_sqrt_regression(notionals_usdt, impacts)

    capacity_usdt = float(np.quantile(notionals_usdt, liquidity_quantile))
    if not np.isfinite(capacity_usdt) or capacity_usdt <= 0:
        capacity_usdt = fallback_capacity_usdt

    median_impact = float(np.median(impacts))
    max_impact = float(np.max(impacts))
    diag = _side_diagnostics(side_name, len(notionals_usdt), median_impact, max_impact)
    diag["beta0"] = beta0
    diag["beta1"] = beta1
    return beta0, beta1, capacity_usdt, diag


def _side_diagnostics(
    side_name: str,
    num_trades: int,
    median_impact_bps: float,
    max_impact_bps: float,
) -> dict:
    return {
        "side": side_name,
        "num_trades": num_trades,
        "median_impact_bps": median_impact_bps,
        "max_impact_bps": max_impact_bps,
    }


def fit_price_impact_model(
    trades: pd.DataFrame,
    liquidity_quantile: float = 0.9,
    quote_to_usdt: float = 1.0,
) -> PriceImpactModel:
    """
    Fit a side-aware sqrt market-impact model from trade-level data.

    Trades sharing the same execution timestamp are aggregated into meta-orders.
    Notionals are converted to USDT so the impact curve is stable across
    BTC price regimes. Impact is modelled as I(Q) = max(0, a + beta * sqrt(Q_usdt)).
    """
    return fit_price_impact_model_with_diagnostics(
        trades=trades,
        liquidity_quantile=liquidity_quantile,
        quote_to_usdt=quote_to_usdt,
    ).model


def fit_price_impact_model_with_diagnostics(
    trades: pd.DataFrame,
    liquidity_quantile: float = 0.9,
    quote_to_usdt: float = 1.0,
) -> PriceImpactFitResult:
    samples = aggregate_trades_to_orders(trades=trades, quote_to_usdt=quote_to_usdt)
    if samples.empty:
        empty = _empty_price_impact_model(quote_to_usdt=quote_to_usdt)
        diagnostics = pd.DataFrame([
            _side_diagnostics("buy", 0, 0.0, 0.0),
            _side_diagnostics("sell", 0, 0.0, 0.0),
        ])
        return PriceImpactFitResult(model=empty, samples=samples, diagnostics=diagnostics)

    notionals_usdt = samples["notional_usdt"].to_numpy(dtype=float)
    fallback_capacity_usdt = float(np.quantile(notionals_usdt[notionals_usdt > 0], liquidity_quantile))
    if not np.isfinite(fallback_capacity_usdt) or fallback_capacity_usdt <= 0:
        fallback_capacity_usdt = np.inf

    buy_df = samples[samples["side"] == 1]
    sell_df = samples[samples["side"] == -1]

    buy_b0, buy_b1, buy_cap, buy_diag = _fit_side(
        df_side=buy_df,
        liquidity_quantile=liquidity_quantile,
        fallback_capacity_usdt=fallback_capacity_usdt,
    )
    sell_b0, sell_b1, sell_cap, sell_diag = _fit_side(
        df_side=sell_df,
        liquidity_quantile=liquidity_quantile,
        fallback_capacity_usdt=fallback_capacity_usdt,
    )

    model = PriceImpactModel(
        buy_beta0=buy_b0,
        buy_beta1=buy_b1,
        sell_beta0=sell_b0,
        sell_beta1=sell_b1,
        buy_capacity_usdt=buy_cap,
        sell_capacity_usdt=sell_cap,
        quote_to_usdt=quote_to_usdt,
        num_trades=int(samples.shape[0]),
    )
    diagnostics = pd.DataFrame([buy_diag, sell_diag])
    return PriceImpactFitResult(model=model, samples=samples, diagnostics=diagnostics)
