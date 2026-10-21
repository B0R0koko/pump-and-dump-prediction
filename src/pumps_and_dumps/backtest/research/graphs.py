from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc

from pumps_and_dumps.core.feature_type import FeatureType
from pumps_and_dumps.core.time_utils import NamedTimeDelta


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
        precision, recall, _ = precision_recall_curve(
            y_true=binary_true, y_score=y_proba
        )
        PrecisionRecallDisplay(precision=precision, recall=recall).plot(
            ax=ax,
            label=f"PRAUC: {model_name} - {auc(x=recall, y=precision):.4f}",  # type: ignore
        )

    # add isoquants for f1-score
    f_scores = np.linspace(0.1, 0.8, num=10)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color="blue", alpha=0.2)
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
    random_state: int = 42,
) -> plt.Figure:
    """Plot feature distributions before/after cross-sectional standardisation."""
    selected_pump_hashes = np.random.default_rng(random_state).choice(
        sorted(df_raw["pump_hash"].unique()), n_pumps, replace=False
    )
    df_small = df_raw[df_raw["pump_hash"].isin(selected_pump_hashes)]
    df_scaled_small = df_scaled[df_scaled["pump_hash"].isin(selected_pump_hashes)]

    fig, axs = plt.subplots(len(feature_types), 2, figsize=(16, 8))
    for i, (row_axes, feature) in enumerate(zip(axs, feature_types)):
        ax1, ax2 = row_axes
        col_name = feature.col_name(offset=offset)

        sns.histplot(
            data=df_small,
            x=col_name,
            hue="pump_hash",
            ax=ax1,
            legend=False,
            alpha=0.05,
            bins=50,
            kde=True,
            stat="probability",
        )
        if i == 0:
            ax1.set_title(f"No standardisation: {col_name}")
        for ph in selected_pump_hashes:
            ax1.axvline(
                x=df_small.loc[
                    df_small["is_pumped"] & (df_small["pump_hash"] == ph), col_name
                ].iloc[0],
                color="red",
                linestyle="--",
            )

        sns.histplot(
            data=df_scaled_small,
            x=col_name,
            hue="pump_hash",
            ax=ax2,
            legend=False,
            alpha=0.05,
            bins=50,
            kde=True,
            stat="probability",
        )
        if i == 0:
            ax2.set_title(f"Cross-section standardisation: {col_name}")
        for ph in selected_pump_hashes:
            ax2.axvline(
                x=df_scaled_small.loc[
                    df_scaled_small["is_pumped"] & (df_scaled_small["pump_hash"] == ph),
                    col_name,
                ].iloc[0],
                color="red",
                linestyle="--",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Top@k / Top@k% plots
# ---------------------------------------------------------------------------


def plot_topk_accuracy(
    df_topk: pd.DataFrame,
    cols: List[str],
    random_baseline: Optional[List[float]] = None,
    bins: Sequence[int] = (1, 2, 3, 5, 10, 20, 30),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Top@k accuracy curves with optional random baseline, anchored at (k=0, 0)."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    origin = pd.DataFrame(0.0, index=[0], columns=cols)
    df_plot = pd.concat([origin, df_topk[cols].sort_index()])
    df_plot.plot(ax=ax, marker="o")
    if random_baseline is not None:
        plt.plot(
            [0, *bins],
            [0.0, *random_baseline],
            color="grey",
            marker="o",
            label="Random ranking baseline",
        )
    plt.legend()
    plt.ylabel("Top@k accuracy")
    plt.xlabel("k")
    plt.title("Top@k accuracy vs k")
    plt.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_topk_percent_curves(
    df_topkp: pd.DataFrame,
    cols: List[str],
    auc_scores: Dict[str, float],
    max_k_percent: float = 0.20,
    save_path: Optional[str] = None,
    random_baseline: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """Plot Top@k% accuracy curves with Top@k%AUC in legend over ``k% in [0, max_k_percent]``."""
    X = np.linspace(0, max_k_percent, 101)
    legends = [f"{c} - {auc_scores.get(c, 0):.3f}" for c in cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    df_topkp[cols].plot(ax=ax, alpha=0.7)
    baseline_x = X if random_baseline is None else df_topkp.index
    baseline_y = X if random_baseline is None else random_baseline
    ax.plot(baseline_x, baseline_y, linestyle="--", color="grey", label="Random ranking baseline")
    ax.legend(legends + ["Random ranking baseline"])
    ax.set_xlabel("k%")
    ax.set_ylabel("Top@k% accuracy")
    ax.set_title("Top@k% accuracy vs k%")
    ax.set_xlim(0, max_k_percent)
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_classification_metrics(
    df_metrics: pd.DataFrame,
    cols: List[str],
    title: str = "Classification metrics on TEST",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of classification metrics for selected models."""
    plot_cols = [c for c in cols if c in df_metrics.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    df_metrics.loc[plot_cols].plot(kind="bar", ax=ax)
    ax.set_title(title)
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
    baseline_cols: Optional[List[str]] = None,
) -> plt.Figure:
    """Cumulative equity curve plot with optional benchmark styling."""
    data = df_curves[cols] if cols else df_curves
    fig, ax = plt.subplots(figsize=(10, 5))
    marker = "o" if len(data.index) <= 100 else None
    data.cumsum().plot(ax=ax, marker=marker)
    for line in ax.get_lines():
        if baseline_cols and line.get_label() in baseline_cols:
            line.set_color("black")
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Cumulative return")
    ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.6)
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
    if "cumulative_roe" in df_pnl.columns:
        metric_col = "cumulative_roe"
        metric_label = "Cumulative test-sample ROE"
        title = "Cumulative test-sample ROE by intended order size with price impact"
    elif "cumulative_roe_pct" in df_pnl.columns:
        metric_col = "cumulative_roe_pct"
        metric_label = "Cumulative test-sample ROE (%)"
        title = "Cumulative test-sample ROE by intended order size with price impact"
    elif "mean_roe" in df_pnl.columns:
        metric_col = "mean_roe"
        metric_label = "Mean ROE"
        title = "PnL sensitivity to intended order size with trade-level price impact"
    elif "mean_roe_pct" in df_pnl.columns:
        metric_col = "mean_roe_pct"
        metric_label = "Mean ROE (%)"
        title = "PnL sensitivity to intended order size with trade-level price impact"
    else:
        raise KeyError(
            "df_pnl must contain one of 'cumulative_roe', 'cumulative_roe_pct', 'mean_roe', or 'mean_roe_pct'"
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df_pnl["quantity_usdt"],
        df_pnl[metric_col],
        color="tab:blue",
        linewidth=2,
        marker="o",
        label=metric_label,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Intended order size (USDT)")
    ax.set_ylabel(metric_label)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.title(title)
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
    """Scatter + fitted sqrt impact curve for buy and sell sides (single beta, no intercept)."""
    b = model.beta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sides = [
        (axes[0], 1, "Buy", "#4C72B0"),
        (axes[1], -1, "Sell", "#DD8452"),
    ]

    # First pass: scatter points
    side_data = {}
    for ax, side, label, color in sides:
        side_df = samples[samples["side"] == side]
        side_data[side] = side_df

        ax.scatter(
            side_df["notional_usdt"],
            side_df["impact_bps"],
            s=14,
            alpha=0.25,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(left=0)

    # Second pass: draw regression line spanning the full x-axis
    for ax, side, label, color in sides:
        side_df = side_data[side]
        if len(side_df) == 0:
            ax.set_title(f"{label} side  (n=0)", fontsize=12)
            continue

        x_max = ax.get_xlim()[1]
        curve_x = np.linspace(0, x_max, 500)
        curve_y = b * np.sqrt(curve_x)

        # R² calculation
        x_obs = side_df["notional_usdt"].values
        y_obs = side_df["impact_bps"].values
        y_pred = b * np.sqrt(x_obs)
        ss_res = np.sum((y_obs - y_pred) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.plot(
            curve_x,
            curve_y,
            color="black",
            linewidth=2.0,
            label=f"$\\beta$={b:.4f},  $R^2$={r2:.3f}",
            zorder=5,
        )

        ax.set_title(f"{label} side  (n={len(side_df):,})", fontsize=12)
        ax.set_xlabel("Net volume (USDT)", fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Impact (bps)", fontsize=11)
    fig.suptitle(
        f"Sqrt impact model: {currency_pair_name} before {pump_time:%Y-%m-%d %H:%M}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_exit_impact_regression(
    model: "PriceImpactModel",  # noqa: F821
    samples: pd.DataFrame,
    currency_pair_name: str,
    pump_time: "datetime",  # noqa: F821
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter + fitted sqrt impact curve for sell-only exit during manipulation window."""
    b = model.beta
    fig, ax = plt.subplots(figsize=(8, 5))

    sell_df = samples[samples["side"] == -1] if "side" in samples.columns else samples

    ax.scatter(
        sell_df["notional_usdt"],
        sell_df["impact_bps"],
        s=14,
        alpha=0.35,
        color="#C44E52",
        edgecolors="none",
        rasterized=True,
        label="Sell candles (5s)",
    )
    ax.set_xlim(left=0)

    if len(sell_df) > 0:
        x_max = ax.get_xlim()[1]
        curve_x = np.linspace(0, x_max, 500)
        curve_y = b * np.sqrt(curve_x)

        x_obs = sell_df["notional_usdt"].values
        y_obs = sell_df["impact_bps"].values
        y_pred = b * np.sqrt(x_obs)
        ss_res = np.sum((y_obs - y_pred) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.plot(
            curve_x,
            curve_y,
            color="black",
            linewidth=2.0,
            label=f"$\\beta$={b:.4f},  $R^2$={r2:.3f}",
            zorder=5,
        )

    ax.set_title(
        f"Exit impact (sell-only, 5s candles): {currency_pair_name}\n"
        f"10 min after pump at {pump_time:%Y-%m-%d %H:%M}  (n={len(sell_df):,})",
        fontsize=12,
    )
    ax.set_xlabel("Net sell volume (USDT)", fontsize=11)
    ax.set_ylabel("Absolute impact (bps)", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Robustness plots
# ---------------------------------------------------------------------------


def _metric_label(col: str) -> str:
    """Map an internal metric column onto the paper's notation (``topk_percent@0.05`` -> ``Top@5%``)."""
    if col == "topk_percent_auc":
        return "Top@k%AUC"
    if col.startswith("topk_percent@"):
        return f"Top@{float(col.split('@', 1)[1]) * 100:g}%"
    if col.startswith("topk@"):
        return f"Top@{col.split('@', 1)[1]}"
    return col


def plot_robustness_distribution(
    df_robustness: pd.DataFrame,
    metric_cols: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of Top@k%AUC and boxplot of per-bin Top@k% metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df_robustness["topk_percent_auc"].hist(ax=axes[0], bins=12, alpha=0.85)
    axes[0].axvline(
        df_robustness["topk_percent_auc"].mean(),
        color="red",
        linestyle="--",
        label="Mean",
    )
    axes[0].set_title("Distribution of Top@k%AUC")
    axes[0].set_xlabel("Top@k%AUC")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    df_robustness[metric_cols].rename(
        columns={c: _metric_label(c) for c in metric_cols}
    ).boxplot(ax=axes[1])
    axes[1].set_title("Distribution of Top@k% metrics")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_feature_importance_distribution(
    feature_importances: pd.DataFrame,
    top_n: int = 12,
    n_cols: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of per-feature importance histograms across the bootstrap retraining runs.

    ``feature_importances`` is the long-format frame produced by
    :func:`run_cross_section_subset_robustness` with ``collect_feature_importances=True``
    (columns ``run_idx``, ``seed``, ``feature``, ``feature_importance``). The ``top_n``
    features with the highest median importance are shown, each as a histogram of its
    importance across runs with a dashed line at the median.
    """
    medians: pd.Series = (
        feature_importances.groupby("feature")["feature_importance"]
        .median()
        .sort_values(ascending=False)
    )
    top_features: List[str] = list(medians.head(top_n).index)

    n_rows: int = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows), squeeze=False
    )

    for idx, feature in enumerate(top_features):
        ax = axes[idx // n_cols][idx % n_cols]
        values: pd.Series = feature_importances.loc[
            feature_importances["feature"] == feature, "feature_importance"
        ]
        ax.hist(values, bins=10, alpha=0.85)
        ax.axvline(
            values.median(), color="red", linestyle="--", linewidth=1, label="Median"
        )
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Count")

    for j in range(len(top_features), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Bootstrap CI plots
# ---------------------------------------------------------------------------


def _pairwise_decisions(models: list[str], pairwise: pd.DataFrame) -> np.ndarray:
    """Validate the complete comparison family and orient each row's decision."""
    n_models = len(models)
    positions = {model: i for i, model in enumerate(models)}
    decisions = np.full((n_models, n_models), np.nan)
    for row in pairwise.itertuples(index=False):
        if row.model_a not in positions or row.model_b not in positions or row.model_a == row.model_b:
            raise ValueError("pairwise must compare distinct models present in summary")
        i, j = positions[row.model_a], positions[row.model_b]
        if np.isfinite(decisions[i, j]):
            raise ValueError("pairwise must contain exactly one row per unordered model pair")
        decision = 0
        if row.reject_h0:
            if row.better_model not in (row.model_a, row.model_b):
                raise ValueError("rejected comparisons must identify better_model")
            decision = 1 if row.better_model == row.model_a else -1
        decisions[i, j], decisions[j, i] = decision, -decision
    if np.isnan(decisions[~np.eye(n_models, dtype=bool)]).any():
        raise ValueError("pairwise must contain a comparison for every unordered model pair")

    return decisions


def _plot_model_intervals(ax, ordered: pd.DataFrame, identifiers: list[str], split: str, confidence: float) -> None:
    n_models = len(ordered)
    y = np.arange(n_models)
    # Draw intervals independently: a percentile CI need not contain its point estimate.
    ax.hlines(y, ordered["ci_lower"], ordered["ci_upper"], color="tab:blue", linewidth=2)
    ax.scatter(ordered["point_estimate"], y, color="tab:blue", zorder=3)
    ax.vlines(ordered["ci_lower"], y - 0.12, y + 0.12, color="tab:blue")
    ax.vlines(ordered["ci_upper"], y - 0.12, y + 0.12, color="tab:blue")
    ax.set_yticks(y, [f"{identifier}: {model}" for identifier, model in zip(identifiers, ordered.index)])
    ax.set_ylim(n_models - 0.5, -0.5)
    ax.set_xlabel(f"{split} Top@k%AUC (higher is better)")
    ax.set_title(f"Point estimates and {confidence:g}% pointwise bootstrap CIs")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def _plot_pairwise_decisions(ax, decisions: np.ndarray, identifiers: list[str], alpha: float) -> None:
    n_models = len(identifiers)
    y = np.arange(n_models)
    colors = ["#d55e00", "#ededed", "#009e73"]
    cmap = ListedColormap(colors).with_extremes(bad="white")
    ax.imshow(np.ma.masked_invalid(decisions), cmap=cmap, vmin=-1.5, vmax=1.5)
    ax.set_xticks(y, identifiers)
    ax.set_yticks(y, identifiers)
    ax.set_xlabel("Column model (IDs match left panel)")
    ax.set_ylabel("Row model")
    ax.set_title(f"Paired two-sided tests; Holm-adjusted α = {alpha:g}")
    ax.set_xticks(np.arange(-0.5, n_models), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    labels = {-1: "−", 0: "·", 1: "+"}
    for (i, j), value in np.ndenumerate(decisions):
        ax.text(
            j,
            i,
            "—" if i == j else labels[value],
            ha="center",
            va="center",
            color="white" if abs(value) == 1 else "black",
        )
    ax.legend(
        handles=[
            Patch(facecolor=colors[2], label="+ Row significantly better"),
            Patch(facecolor=colors[0], label="− Row significantly worse"),
            Patch(facecolor=colors[1], label="· Inconclusive"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=9,
    )


def plot_model_bootstrap_comparison(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot deterministic Top@k%AUC intervals and paired decisions.

    ``summary`` is indexed by model and contains ``point_estimate``, ``ci_lower``
    and ``ci_upper``. ``pairwise`` contains one row per unordered model pair,
    with ``model_a``, ``model_b``, ``reject_h0`` and ``better_model`` from
    :func:`compare_models_bootstrap_topk_percent_auc`. The optional ``alpha``
    column sets the confidence/significance labels (default: 0.05).

    Separate model intervals are pointwise; their overlap is never used to
    determine significance. Inconclusive comparisons do not establish equality.
    """
    if summary.empty or not summary.index.is_unique:
        raise ValueError("summary must contain at least one model and have a unique model index")
    ordered = summary.sort_values("point_estimate", ascending=False, kind="stable")
    metric_label = "Top@k%AUC"
    split = str(ordered["dataset_type"].iloc[0]).capitalize() if "dataset_type" in ordered else "Validation"
    models = list(ordered.index)
    alpha = float(ordered["alpha"].iloc[0]) if "alpha" in ordered.columns else 0.05
    confidence = 100 * (1 - alpha)
    n_models = len(models)

    decisions = _pairwise_decisions(models, pairwise)

    fig, (forest_ax, matrix_ax) = plt.subplots(
        1,
        2,
        figsize=(18, max(6, 0.48 * n_models + 2)),
        gridspec_kw={"width_ratios": [1.1, 1]},
        layout="constrained",
    )
    identifiers = [f"M{i + 1}" for i in range(n_models)]
    _plot_model_intervals(forest_ax, ordered, identifiers, split, confidence)
    _plot_pairwise_decisions(matrix_ax, decisions, identifiers, alpha)
    fig.suptitle(f"{split} model comparison: {metric_label}\n" "Paired bootstrap over pump events")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_bootstrap_ci(
    df_topk_ci: pd.DataFrame,
    df_topkp_ci: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Error-bar plots for Top@k and Top@k% with bootstrap CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(df_topk_ci.index))
    y = df_topk_ci["point_estimate"].to_numpy()
    yerr = np.vstack([y - df_topk_ci["ci_lower"].to_numpy(), df_topk_ci["ci_upper"].to_numpy() - y])
    axes[0].errorbar(x, y, yerr=yerr, fmt="o-", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Top@{k:g}" for k in df_topk_ci.index])
    axes[0].set_title("Top@k with 95% CI")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Top@k")
    axes[0].grid(True)

    x2 = np.arange(len(df_topkp_ci.index))
    y2 = df_topkp_ci["point_estimate"].to_numpy()
    yerr2 = np.vstack(
        [
            y2 - df_topkp_ci["ci_lower"].to_numpy(),
            df_topkp_ci["ci_upper"].to_numpy() - y2,
        ]
    )
    axes[1].errorbar(x2, y2, yerr=yerr2, fmt="o-", capsize=4)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"Top@{k * 100:g}%" for k in df_topkp_ci.index])
    axes[1].set_title("Top@k% with 95% CI")
    axes[1].set_xlabel("k%")
    axes[1].set_ylabel("Top@k%")
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
