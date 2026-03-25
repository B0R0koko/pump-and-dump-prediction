from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc

from backtest.utils.sample import Dataset
from core.feature_type import FeatureType
from core.time_utils import NamedTimeDelta


def plot_precision_recall(
    binary_true: pd.Series,
    model_probas: Dict[str, np.array],
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    :param binary_true: - actual values of the target variable. Indicates if the asset was manipulated or not
    :param model_probas: - predicted probabilities of the target variable. Keys are models' names that produced probabilities
    :param figsize: - size of the figure in inches
    :return: - plt.Figure with Precision Recall curve showing the performance of the model for different thresholds
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for model_name, y_proba in model_probas.items():
        precision, recall, _ = precision_recall_curve(y_true=binary_true, y_score=y_proba)
        PrecisionRecallDisplay(precision=precision, recall=recall).plot(
            ax=ax,
            label=f"PRAUC: {model_name} - {auc(x=recall, y=precision):.4f}",  # type: ignore
        )

    # add isoquants for f1-score
    f_scores = np.linspace(0.1, 0.8, num=10)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="blue", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))  # type: ignore

    plt.legend(loc="upper right")
    plt.title("Precision Recall curves")

    return fig


# ---------------------------------------------------------------------------
# Feature distribution plots
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    df_raw: pd.DataFrame,
    df_scaled: pd.DataFrame,
    feature_types: List[FeatureType],
    offset: NamedTimeDelta = NamedTimeDelta.ONE_DAY,
    n_pumps: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feature distributions before/after cross-sectional standardisation."""
    selected_pump_hashes = np.random.choice(df_raw["pump_hash"].unique(), n_pumps, replace=False)
    df_small = df_raw[df_raw["pump_hash"].isin(selected_pump_hashes)]
    df_scaled_small = df_scaled[df_scaled["pump_hash"].isin(selected_pump_hashes)]

    fig, axs = plt.subplots(len(feature_types), 2, figsize=(16, 8))
    for i, (row_axes, feature) in enumerate(zip(axs, feature_types)):
        ax1, ax2 = row_axes
        col_name = feature.col_name(offset=offset)

        sns.histplot(data=df_small, x=col_name, hue="pump_hash", ax=ax1, legend=False, alpha=0.05, bins=50, kde=True, stat="probability")
        if i == 0:
            ax1.set_title(f"No standardisation: {col_name}")
        for ph in selected_pump_hashes:
            ax1.axvline(x=df_small.loc[df_small["is_pumped"] & (df_small["pump_hash"] == ph), col_name].iloc[0], color="red", linestyle="--")

        sns.histplot(data=df_scaled_small, x=col_name, hue="pump_hash", ax=ax2, legend=False, alpha=0.05, bins=50, kde=True, stat="probability")
        if i == 0:
            ax2.set_title(f"Cross-section standardisation: {col_name}")
        for ph in selected_pump_hashes:
            ax2.axvline(x=df_scaled_small.loc[df_scaled_small["is_pumped"] & (df_scaled_small["pump_hash"] == ph), col_name].iloc[0], color="red", linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Top@K / Top@K% plots
# ---------------------------------------------------------------------------


def plot_topk_accuracy(
    df_topk: pd.DataFrame,
    cols: List[str],
    random_baseline: Optional[List[float]] = None,
    bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Top@K accuracy curves with optional random baseline."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    df_topk[cols].plot(ax=ax, marker="o")
    if random_baseline is not None:
        plt.plot(list(bins), random_baseline, color="grey", marker="o", label="RandomModel")
    plt.legend()
    plt.ylabel("TOPK accuracy")
    plt.xlabel("K")
    plt.title("TOPK Accuracy vs K")
    plt.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_topk_percent_curves(
    df_topkp: pd.DataFrame,
    cols: List[str],
    auc_scores: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Top@K% accuracy curves with AUC in legend."""
    X = np.arange(0, 1.01, 0.01)
    legends = [f"{c} - {auc_scores.get(c, 0):.3f}" for c in cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    df_topkp[cols].plot(ax=ax, alpha=0.7)
    ax.plot(X, X, linestyle="--", color="grey", label="RandomClassifier 0.5")
    ax.legend(legends + ["RandomClassifier 0.5"])
    ax.set_xlabel("K%")
    ax.set_ylabel("TOPK% accuracy")
    ax.set_title("TOPK% Accuracy vs K%")
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_classification_metrics(
    df_metrics: pd.DataFrame,
    cols: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of classification metrics for selected models."""
    plot_cols = [c for c in cols if c in df_metrics.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    df_metrics.loc[plot_cols].plot(kind="bar", ax=ax)
    ax.set_title("Classification metrics on TEST")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Portfolio / PnL plots
# ---------------------------------------------------------------------------


def plot_equity_curves(
    df_curves: pd.DataFrame,
    cols: Optional[List[str]] = None,
    title: str = "Cumulative return on the test set for portfolio of size 5",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cumulative equity curve plot."""
    data = df_curves[cols] if cols else df_curves
    fig, ax = plt.subplots(figsize=(10, 5))
    data.cumsum().plot(ax=ax, marker="o")
    ax.set_title(title)
    ax.set_ylabel("Cumulative return")
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_pnl_sensitivity(
    df_pnl: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """PnL sensitivity to intended order size with price impact."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_pnl["quantity_usdt"], df_pnl["mean_roe_pct"], color="tab:blue", linewidth=2, marker="o", label="Mean ROE (%)")
    ax.set_xscale("log")
    ax.set_xlabel("Intended order size (USDT)")
    ax.set_ylabel("Mean ROE (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.title("PnL sensitivity to intended order size with trade-level price impact")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Impact regression visualisation
# ---------------------------------------------------------------------------


def plot_impact_regression(
    model: "PriceImpactModel",  # noqa: F821
    samples: pd.DataFrame,
    currency_pair_name: str,
    pump_time: "datetime",  # noqa: F821
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter + fitted sqrt impact curve for buy and sell sides."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, side, label, color in [
        (axes[0], 1, "Buy", "tab:blue"),
        (axes[1], -1, "Sell", "tab:orange"),
    ]:
        a = model.buy_beta0 if side == 1 else model.sell_beta0
        b = model.buy_beta1 if side == 1 else model.sell_beta1
        side_df = samples[samples["side"] == side]

        ax.scatter(side_df["notional_usdt"], side_df["impact_bps"], s=16, alpha=0.25, color=color, edgecolors="none")
        if len(side_df) > 0:
            curve_x = np.linspace(0, side_df["notional_usdt"].quantile(0.99), 200)
            curve_y = np.maximum(0, a + b * np.sqrt(curve_x))
            ax.plot(curve_x, curve_y, color="black", linewidth=2, label=f"a={a:.2f}, $\\beta$={b:.4f}")
            if a < 0 and b > 0:
                q_star = (a / b) ** 2
                ax.axvline(q_star, color="red", linestyle="--", alpha=0.6, label=f"Q*={q_star:.0f} USDT")
        ax.set_title(f"{label} side (n={len(side_df)})")
        ax.set_xlabel("Order size (USDT)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Impact (bps)")
    plt.suptitle(f"Sqrt impact model: {currency_pair_name} before {pump_time:%Y-%m-%d %H:%M}", y=1.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Robustness plots
# ---------------------------------------------------------------------------


def plot_robustness_distribution(
    df_robustness: pd.DataFrame,
    metric_cols: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of TOPK% AUC and boxplot of per-bin metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df_robustness["topk_percent_auc"].hist(ax=axes[0], bins=12, alpha=0.85)
    axes[0].axvline(df_robustness["topk_percent_auc"].mean(), color="red", linestyle="--", label="Mean")
    axes[0].set_title("Distribution of TOPK% AUC")
    axes[0].set_xlabel("TOPK% AUC")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    df_robustness[metric_cols].boxplot(ax=axes[1])
    axes[1].set_title("Distribution of TOPK% metrics")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Bootstrap CI plots
# ---------------------------------------------------------------------------


def plot_bootstrap_ci(
    df_topk_ci: pd.DataFrame,
    df_topkp_ci: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Error-bar plots for Top@K and Top@K% with bootstrap CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(df_topk_ci.index))
    y = df_topk_ci["point_estimate"].to_numpy()
    yerr = np.vstack([y - df_topk_ci["ci_lower"].to_numpy(), df_topk_ci["ci_upper"].to_numpy() - y])
    axes[0].errorbar(x, y, yerr=yerr, fmt="o-", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_topk_ci.index)
    axes[0].set_title("Top@K with 95% CI")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Top@K")
    axes[0].grid(True)

    x2 = np.arange(len(df_topkp_ci.index))
    y2 = df_topkp_ci["point_estimate"].to_numpy()
    yerr2 = np.vstack([y2 - df_topkp_ci["ci_lower"].to_numpy(), df_topkp_ci["ci_upper"].to_numpy() - y2])
    axes[1].errorbar(x2, y2, yerr=yerr2, fmt="o-", capsize=4)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(df_topkp_ci.index)
    axes[1].set_title("Top@K% with 95% CI")
    axes[1].set_xlabel("K%")
    axes[1].set_ylabel("Top@K%")
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
