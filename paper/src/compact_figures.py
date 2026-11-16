"""Redraw audited manuscript plots at their printed size without changing data."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "notebooks/analysis_outputs"
DATA = ARTIFACTS / "seed_sweep_paper"
OUTPUT = ARTIFACTS / "seed_sweep_paper_compact"
SINGLE_COLUMN = {
    "validation_seed_distributions.png", "test_seed_distributions.png",
    "topk_accuracy_validation.png", "topk_accuracy_test.png",
    "topkp_auc_validation.png", "topkp_auc_test.png",
    "classification_metrics_validation.png", "classification_metrics_test.png",
    "feature_importances.png", "auc_distributions.png",
}
FILENAMES = SINGLE_COLUMN | {
    "significance_bootstrap_ci.png", "robustness_metrics_distribution.png",
    "portfolios_vs_btc_test.png", "portfolios_1000usdt_with_slippage.png",
}
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, **kwargs)


def save(fig, filename):
    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.grid(alpha=.2, linewidth=.4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "images" / filename, dpi=300, facecolor="white")
    plt.close(fig)


def prepare_compact_figures(names, model_names, *, check=False):
    manifest_path = OUTPUT / "provenance.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["code_sha256"] == sha(Path(__file__))
                and manifest["model_names"] == names
                and all((ROOT / p).exists() and sha(ROOT / p) == h
                        for section in ("input_sha256", "output_sha256")
                        for p, h in manifest[section].items())):
            return manifest
    assert not check, "Compact figures are stale; run update_results.py first"
    (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    INPUTS.clear()
    colors = dict(zip(names, ["#009e73", "#0072b2", "#cc79a7", "#9467bd", "#56b4e9", "#a6a600"]))
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "lines.linewidth": 1,
                         "axes.linewidth": .5, "figure.constrained_layout.use": True,
                         "figure.constrained_layout.h_pad": .035,
                         "figure.constrained_layout.w_pad": .035}):
        for split in ("validation", "test"):
            seed_root = ARTIFACTS / f"{split}_seed_sweep_100x25"
            scores = read(seed_root / f"{split}_auc_by_seed.csv", index_col=0)
            order = scores.mean().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(3.45, 2.4))
            values = [scores[k].to_numpy() for k in order]
            ax.boxplot(values, orientation="horizontal", showmeans=True,
                       tick_labels=[names[model_names[k]] for k in order],
                       flierprops={"markersize": 2},
                       meanprops={"markersize": 4}, medianprops={"linewidth": 1})
            for i, (key, value) in enumerate(zip(order, values), 1):
                ax.scatter(value, i + np.linspace(-.15, .15, len(value)), s=3,
                           alpha=.4, color=colors[model_names[key]], zorder=3)
            ax.invert_yaxis()
            ax.set_xlabel(f"{split.title()} Top@k%AUC (k = 0–20%)")
            save(fig, f"{split}_seed_distributions.png")

            predictions = read(ARTIFACTS / "test_seed_sweep_100x25/fits/catboost/seed_042" / f"{split}_predictions.parquet")
            sizes = predictions.groupby("pump_hash").size().to_numpy()
            for kind in ("topk", "topkp"):
                frame = read(DATA / "tables" / f"{split}_{'topk' if kind == 'topk' else 'topkp_curves'}.csv", index_col=0)
                if kind == "topk":
                    frame = pd.concat([pd.DataFrame(0., index=[0], columns=frame.columns), frame])
                    x = frame.index.to_numpy()
                    random = (np.minimum(x[None, :], sizes[:, None]) / sizes[:, None]).mean(axis=0)
                else:
                    x = frame.index.to_numpy() * 100
                    random = (np.ceil(sizes[:, None] * frame.index.to_numpy()) / sizes[:, None]).mean(axis=0)
                fig, ax = plt.subplots(figsize=(3.45, 2.65))
                for name in names:
                    label = names[name]
                    if kind == "topkp":
                        area = np.trapezoid(frame[name], frame.index) / .2
                        label += f" ({area:.3f})"
                    ax.plot(x, frame[name], label=label, color=colors[name],
                            marker="o" if kind == "topk" else None, markersize=2)
                ax.plot(x, random, "--", color=".45", label="Random ranking")
                ax.set(xlabel="Selected assets, k" if kind == "topk" else "Selected assets, k (%)",
                       ylabel="Top@k" if kind == "topk" else "Top@k%", xlim=(0, x[-1]), ylim=(0, 1))
                ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2,
                          frameon=False, columnspacing=.7, handlelength=1.4, fontsize=6.5)
                save(fig, f"{'topk_accuracy' if kind == 'topk' else 'topkp_auc'}_{split}.png")

            metrics = read(DATA / "tables" / f"{split}_classification.csv", index_col=0).loc[list(names)]
            fig, ax = plt.subplots(figsize=(3.45, 2.6))
            y = np.arange(len(metrics))
            for i, (column, label) in enumerate(zip(metrics, ["PR-AUC", "F1 (Top@1)", "Balanced accuracy (Top@1)"])):
                ax.barh(y + (i - 1) * .24, metrics[column], height=.23, label=label)
            ax.set(yticks=y, yticklabels=[names[n] for n in metrics.index], xlabel="Score", xlim=(0, 1))
            ax.invert_yaxis()
            ax.legend(loc="upper center", bbox_to_anchor=(.4, -.2), ncol=2, frameon=False,
                      columnspacing=.8, handlelength=1.1)
            save(fig, f"classification_metrics_{split}.png")

        importance = read(DATA / "tables/selected_model_feature_importances.csv", index_col=0).iloc[:, 0].sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(3.45, 3.45))
        ax.barh(importance.index, importance, color=colors[model_names["catboost"]])
        ax.set_xlabel("PredictionValuesChange importance")
        save(fig, "feature_importances.png")

        samples = read(DATA / "test_auc_bootstrap/auc_distributions.csv", index_col=0)
        summary = read(DATA / "tables/auc_summary.csv", index_col=0).sort_values("point_estimate", ascending=False)
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        for i, name in enumerate(summary.index):
            q = samples[name].quantile([.025, .25, .5, .75, .975])
            ax.hlines(i, q.iloc[0], q.iloc[4], color=colors[name], linewidth=.9)
            ax.hlines(i, q.iloc[1], q.iloc[3], color=colors[name], linewidth=4)
            ax.plot(q.iloc[2], i, "|", color="black", markersize=5, markeredgewidth=.7)
            ax.plot(summary.loc[name, "point_estimate"], i, "o", color="black", markerfacecolor="white", markersize=3, markeredgewidth=.6)
        ax.set(yticks=np.arange(len(summary)), yticklabels=[names[n] for n in summary.index],
               xlabel="Test Top@k%AUC (k = 0–20%)")
        ax.invert_yaxis()
        ax.legend(handles=[Line2D([], [], marker="o", markerfacecolor="white", color="black", linestyle="", markersize=3, label="Point estimate"),
                           Line2D([], [], marker="|", color="black", linestyle="", label="Median")],
                  loc="upper center", bbox_to_anchor=(.3, -.21), ncol=2, frameon=False,
                  columnspacing=.7, handlelength=.8)
        save(fig, "auc_distributions.png")

        fig, ax = plt.subplots(figsize=(6.95, 2.1))
        frame = read(DATA / "tables/selected_topk_ci.csv", index_col=0)
        y = frame.point_estimate.to_numpy()
        ax.errorbar(np.arange(len(y)), y,
                    yerr=np.vstack([y - frame.ci_lower, frame.ci_upper - y]), fmt="o-", markersize=3, capsize=3)
        ax.set(xticks=np.arange(len(y)), xticklabels=[f"{v:g}" for v in frame.index],
               xlabel="Selected assets, k", ylabel="Top@k", ylim=(0, 1))
        save(fig, "significance_bootstrap_ci.png")

        metrics = read(DATA / "robustness/seed_metrics.csv", index_col=0)
        fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.15))
        axes[0].hist(metrics.topk_percent_auc, bins=12, alpha=.8)
        axes[0].axvline(metrics.topk_percent_auc.mean(), color="black", linestyle="--", label="Mean")
        axes[0].set(xlabel="Test Top@k%AUC", ylabel="Number of seeds")
        axes[0].legend(frameon=False)
        fractions = [.01, .02, .05, .1, .2]
        axes[1].boxplot([metrics[f"topk_percent@{p}"] for p in fractions], tick_labels=[f"{p * 100:g}" for p in fractions], showmeans=True,
                        meanprops={"markersize": 4}, flierprops={"markersize": 2})
        axes[1].set(xlabel="Selected assets, k (%)", ylabel="Test Top@k%")
        save(fig, "robustness_metrics_distribution.png")

        for scenario, old, filename in [("", "paper_run", "portfolios_vs_btc_test.png"),
                                        ("portfolio_1000usdt_slippage", "portfolio_1000usdt_slippage", "portfolios_1000usdt_with_slippage.png")]:
            daily = read(DATA / scenario / "daily_returns.csv", index_col=0, parse_dates=True)
            paths = read(ARTIFACTS / "paper_cohort_473_l2" / old / "random_portfolio/cumulative_paths.csv")
            band = paths.loc[paths.k == 5].copy()
            band.index = pd.to_datetime(band.time).dt.normalize()
            band = band.groupby(level=0)[["lower", "upper"]].last().reindex(daily.index).ffill().fillna(0)
            fig, ax = plt.subplots(figsize=(6.95, 2.5))
            for col in daily.columns:
                if col.startswith("K="):
                    ax.plot(daily.index, daily[col].cumsum(), label=col.replace("K", "k"))
                else:
                    random = col.startswith("Random")
                    ax.plot(daily.index, daily[col].cumsum(), label="Random k=5: mean" if random else "BTC/USDT",
                            color=".4" if random else "black", linestyle="--" if random else "-", linewidth=1)
            ax.fill_between(daily.index, band.lower, band.upper, color=".5", alpha=.15, label="Random k=5: 95% range")
            ax.axhline(0, color=".4", linewidth=.5)
            ax.set_ylabel("Cumulative return")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=5, frameon=False,
                      handlelength=1.6, columnspacing=1)
            save(fig, filename)

    outputs = {str((OUTPUT / "images" / f).relative_to(ROOT)): sha(OUTPUT / "images" / f) for f in sorted(FILENAMES)}
    manifest = {"code_sha256": sha(Path(__file__)), "model_names": names,
                "input_sha256": INPUTS.copy(), "output_sha256": outputs,
                "single_column_inches": 3.45, "double_column_inches": 6.95,
                "note": "Layout derivatives of audited numeric exports; no model fitting or resampling."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
