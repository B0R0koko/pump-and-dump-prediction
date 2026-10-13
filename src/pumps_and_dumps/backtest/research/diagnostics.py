"""Dataset and impact-fit diagnostics used in the research notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from pumps_and_dumps.backtest.research import graphs
from pumps_and_dumps.backtest.experiment import Experiment
from pumps_and_dumps.backtest.pipelines.inspection import feature_importances
from pumps_and_dumps.backtest.research.impact_example import (
    ExitImpactExample,
    ImpactExample,
)
from pumps_and_dumps.core.feature_type import FeatureType
from pumps_and_dumps.features.schema import REGRESSOR_OFFSETS
from pumps_and_dumps.features.schema import regressor_columns


def plot_cross_section_sizes(df: pd.DataFrame, *, save_path: Path) -> None:
    sizes = df.groupby("pump_hash").agg(
        pump_time=("pump_time", "first"), cross_section_size=("pump_time", "count")
    )
    ax = sizes.sort_values("pump_time").plot(
        x="pump_time", y="cross_section_size", title="Cross-section size over time"
    )
    ax.figure.tight_layout()
    ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def describe_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a diagnostic copy with clipped power-law values and feature statistics."""
    feature_cols = regressor_columns()
    powerlaw_cols = FeatureType.POWERLAW_ALPHA.col_names(offsets=REGRESSOR_OFFSETS)
    clipped = df.copy()
    clipped[powerlaw_cols] = clipped[powerlaw_cols].clip(1, 2)
    print(f"Dataset shape: {clipped.shape}")
    print(f"NaN rows remaining: {clipped[feature_cols].isna().any(axis=1).sum()}")
    return clipped, clipped[feature_cols].describe().T[["mean", "std", "min", "max"]]


def plot_feature_importances(
    experiment: Experiment, *, save_path: Path, top_n: int = 20
) -> None:
    importance = feature_importances(experiment.get_model())
    ax = (
        importance.sort_values()
        .tail(top_n)
        .rename_axis("feature")
        .plot.barh(figsize=(10, 6))
    )
    ax.set_title(experiment.get_experiment_name())
    ax.figure.tight_layout()
    ax.figure.savefig(save_path)
    plt.show()


def impact_fit_metrics(samples: pd.DataFrame, beta: float) -> pd.Series:
    """Compute the same fit diagnostics for entry and exit square-root models."""
    observed = samples["impact_bps"].to_numpy()
    predicted = beta * np.sqrt(samples["notional_usdt"].to_numpy())
    residuals = observed - predicted
    ss_total = np.sum((observed - np.mean(observed)) ** 2)
    return pd.Series(
        {
            "n": len(samples),
            "R²": 1 - np.sum(residuals**2) / ss_total if ss_total > 0 else 0.0,
            "MAE (bps)": np.mean(np.abs(residuals)),
            "RMSE (bps)": np.sqrt(np.mean(residuals**2)),
        }
    )


def show_impact_example(
    example: ImpactExample | ExitImpactExample, *, save_path: Path
) -> None:
    """Plot an illustrative fit and report per-side errors; never used for model selection."""
    is_exit = isinstance(example, ExitImpactExample)
    samples = example.fit_result.samples.copy()
    if not is_exit:
        samples = samples.loc[samples["notional_usdt"] > 0]
    plot = (
        graphs.plot_exit_impact_regression if is_exit else graphs.plot_impact_regression
    )
    plot(
        model=example.fit_result.model,
        samples=samples,
        currency_pair_name=example.currency_pair.name,
        pump_time=example.pump.time,
        save_path=str(save_path),
    )
    groups = (
        {"Exit (sell-only, 5s candles)": samples}
        if is_exit
        else {
            label: samples.loc[samples["side"] == side]
            for label, side in (("Buy", 1), ("Sell", -1))
        }
    )
    metrics = {
        label: impact_fit_metrics(group, example.fit_result.model.beta)
        for label, group in groups.items()
        if not group.empty
    }
    display(pd.DataFrame(metrics).T)
    print(f"β = {example.fit_result.model.beta:.6f}")
    print(f"Asset: {example.currency_pair.name}, Pump: {example.pump.time}")
