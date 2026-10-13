from collections.abc import Sequence

import numpy as np

from pumps_and_dumps.backtest.portfolio.base import PortfolioStats


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize_quantity(
    stats: Sequence[PortfolioStats], quantity_usdt: float
) -> dict[str, float]:
    """Summarize an order-size sweep without owning execution or portfolio settings."""
    pnls = [item.pnl for item in stats]
    roes = [pnl / quantity_usdt for pnl in pnls] if quantity_usdt > 0 else []
    overall_pnl = float(np.sum(pnls))
    result = {
        "quantity_usdt": quantity_usdt,
        "num_pumps": float(len(stats)),
        "overall_pnl": overall_pnl,
        "cumulative_roe": overall_pnl / quantity_usdt if quantity_usdt > 0 else 0.0,
        "mean_pnl": _mean(pnls),
        "median_pnl": _median(pnls),
        "mean_roe": _mean(roes),
        "median_roe": _median(roes),
        "mean_entry_impact_bps": _mean([item.mean_entry_impact_bps for item in stats]),
        "mean_exit_impact_bps": _mean([item.mean_exit_impact_bps for item in stats]),
        "mean_entry_impact_num_bars": _mean(
            [item.mean_entry_impact_num_bars for item in stats]
        ),
        "mean_exit_impact_num_bars": _mean(
            [item.mean_exit_impact_num_bars for item in stats]
        ),
        "mean_executed_notional_usdt": _mean(
            [item.executed_notional_usdt for item in stats]
        ),
    }
    return result
