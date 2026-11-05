"""Replay reported test metrics, bootstrap draws and portfolios from saved outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, balanced_accuracy_score, f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/analysis_outputs/paper_cohort_473_l2"


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def statistics(event_returns, times):
    values = np.asarray(event_returns)
    if values.ndim == 1:
        values = values[:, None]
    dates = pd.DatetimeIndex(times).normalize()
    day_index = (dates - dates.min()).days.to_numpy()
    daily = np.zeros((int(day_index.max()) + 1, values.shape[1]))
    np.add.at(daily, day_index, values)
    annual_return = daily.mean(axis=0) * 365
    volatility = daily.std(axis=0, ddof=1) * np.sqrt(365)
    return {
        "average event return": values.mean(axis=0),
        "cumulative return": values.sum(axis=0),
        "annualized return": annual_return,
        "annualized volatility": volatility,
        "Sharpe ratio": annual_return / volatility,
    }


def main():
    from audit_fitted_models import audit_models

    model_verification = audit_models()
    boot, run, impact = OUTPUT / "test_auc_bootstrap", OUTPUT / "paper_run", OUTPUT / "portfolio_1000usdt_slippage"
    metadata = json.loads((boot / "metadata.json").read_text())
    predictions = pd.read_parquet(boot / "test_predictions.parquet")
    topk = pd.read_csv(run / "tables/test_topk.csv", index_col=0)
    topkp = pd.read_csv(run / "tables/test_topkp.csv", index_col=0)
    classification = pd.read_csv(run / "tables/test_classification.csv", index_col=0)
    summary = pd.read_csv(boot / "auc_summary.csv", index_col=0)
    contributions = pd.read_csv(boot / "event_aucs.csv", index_col=0)
    distributions = pd.read_csv(boot / "auc_distributions.csv", index_col=0)
    with np.load(boot / "resampled_events.npz") as saved:
        event_ids, draws = saved["event_ids"], saved["sampled_indices"]
    assert draws.shape == (10000, 80)
    np.testing.assert_array_equal(draws, np.random.default_rng(42).integers(0, 80, size=draws.shape))
    assert list(contributions.index) == list(event_ids)
    bins = np.array(metadata["auc_bins"])
    reconstructed = {}
    for model, scores in predictions.groupby("model", sort=False):
        assert not scores.duplicated(["pump_hash", "currency_pair"]).any()
        ordered = scores.sort_values("probas_pred", ascending=False, kind="stable")
        groups = dict(tuple(ordered.groupby("pump_hash", sort=False)))
        ranks, counts, event_aucs = [], [], []
        predicted_targets = ordered.groupby("pump_hash", sort=False).head(1).index
        decision = scores.index.isin(predicted_targets)
        precision, recall, _ = precision_recall_curve(scores.is_pumped, scores.probas_pred)
        close(classification.loc[model, "PR-AUC"], auc(recall, precision))
        close(classification.loc[model, "F1 (Top@1 per cross-section)"], f1_score(scores.is_pumped, decision))
        close(
            classification.loc[model, "Balanced Accuracy (Top@1 per cross-section)"],
            balanced_accuracy_score(scores.is_pumped, decision),
        )
        for event in event_ids:
            group = groups[event]
            assert group.is_pumped.sum() == 1
            rank = np.flatnonzero(group.is_pumped.to_numpy())[0] + 1
            ranks.append(rank)
            counts.append(len(group))
            curve = rank <= np.ceil(bins * len(group))
            event_aucs.append(np.trapezoid(curve.astype(float), bins) / 0.2)
        ranks, counts, event_aucs = np.array(ranks), np.array(counts), np.array(event_aucs)
        for k in topk.index:
            close(topk.loc[k, model], (ranks <= k).mean())
        for fraction in topkp.index:
            close(topkp.loc[fraction, model], (ranks <= np.ceil(counts * fraction)).mean())
        close(contributions[model], event_aucs)
        bootstrap = event_aucs[draws].mean(axis=1)
        close(distributions[model], bootstrap)
        close(summary.loc[model, "point_estimate"], event_aucs.mean())
        close(summary.loc[model, ["ci_lower", "ci_upper"]].astype(float), np.quantile(bootstrap, [0.025, 0.975]))
        reconstructed[model] = event_aucs
    pairs = pd.read_csv(boot / "auc_pairwise.csv")
    for row in pairs.itertuples():
        difference = reconstructed[row.model_a] - reconstructed[row.model_b]
        estimate = difference.mean()
        boot_difference = distributions[row.model_a] - distributions[row.model_b]
        close(row.observed_diff, estimate)
        close([row.ci_lower, row.ci_upper], np.quantile(boot_difference, [0.025, 0.975]))
        p = (1 + (np.abs(boot_difference - estimate) >= abs(estimate) - 32 * np.finfo(float).eps).sum()) / 10001
        close(row.p_value, p)
    ordered_p = pairs.p_value.sort_values(kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(ordered_p.to_numpy() * np.arange(len(pairs), 0, -1)))
    close(pairs.loc[ordered_p.index, "p_value_holm"], adjusted)

    winner = metadata["provenance"]["primary_model"]
    membership = pd.read_csv(run / "test_events.csv", index_col=0, parse_dates=["pump_time"])
    early = membership.index[membership.pump_time < pd.Timestamp("2022-06-01")]
    temporal = pd.read_csv(run / "tables/subperiod_metrics.csv", index_col=0)
    close(temporal.loc["early", "topk_percent_auc"], contributions.loc[early, winner].mean())
    assert len(early) == int(temporal.loc["early", "n_pumps"]) == 76
    for filename, table in [
        ("catboost_topkauc_subset_runs.csv", "robustness_summary.csv"),
        ("logistic_subset_runs.csv", "logistic_robustness_summary.csv"),
    ]:
        trials = pd.read_csv(run / "robustness" / filename)
        recorded = pd.read_csv(run / "tables" / table, index_col=0)
        assert len(trials) == 25 and trials.train_cross_sections.eq(203).all()
        expected = trials[recorded.index].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
        close(recorded, expected[recorded.columns])
    importance = pd.read_csv(run / "robustness/catboost_topkauc_feature_importances.csv")
    quantiles = pd.read_csv(run / "tables/feature_importance_quantiles.csv", index_col=0)
    groups = importance.groupby("feature").feature_importance
    for column, q in [("p10", 0.1), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9)]:
        close(quantiles[column], groups.quantile(q).reindex(quantiles.index))
    selection = json.loads((run / "tables/logistic_l2_selection.json").read_text())
    grid = pd.read_csv(run / "tables/logistic_l2_validation_path.csv")
    best = grid.sort_values(["validation_topk_percent_auc", "C"], ascending=[False, True]).iloc[0]
    close(selection["selected_parameters"]["C"], best.C)
    selected = predictions.loc[predictions.model == winner]
    main_metrics = pd.read_csv(run / "tables/portfolio_statistics.csv", index_col=0)
    impact_metrics = pd.read_csv(impact / "model_statistics.csv", index_col=0)
    replayed = {}
    for label, asset_path, random_directory in [
        ("no_impact", run / "random_portfolio/asset_returns.parquet", run / "random_portfolio"),
        ("impact_1000", impact / "asset_returns.parquet", impact / "random_portfolio"),
    ]:
        assets = pd.read_parquet(asset_path)
        joined = selected.merge(
            assets, on=["pump_hash", "currency_pair", "is_pumped"], validate="one_to_one", how="outer", indicator=True
        )
        assert joined._merge.eq("both").all()
        joined = joined.sort_values(
            ["pump_hash", "probas_pred", "currency_pair"], ascending=[True, False, True], kind="stable"
        )
        joined["rank"] = joined.groupby("pump_hash").cumcount() + 1
        replayed[label] = {}
        for k in [1, 2, 5, 10, 20, 30]:
            chosen = joined.loc[joined["rank"] <= k]
            total = chosen["return"].sum() / k
            hit_rate = chosen.groupby("pump_hash").is_pumped.any().mean()
            expected = (
                main_metrics.loc[
                    f"{winner.replace('CatBoost classifier + Top@k%AUC early stopping', 'CatBoost + ES')} K={k}",
                    "cumulative return",
                ]
                if label == "no_impact"
                else impact_metrics.loc[k, "cumulative return"]
            )
            close(total, expected)
            close(hit_rate, topk.loc[k, winner])
            events = chosen.groupby("event_index")["return"].sum().reindex(range(80), fill_value=0) / k
            times = assets.groupby("event_index", sort=True).pump_time.first()
            recorded = (
                main_metrics.loc[
                    f"{winner.replace('CatBoost classifier + Top@k%AUC early stopping', 'CatBoost + ES')} K={k}"
                ]
                if label == "no_impact"
                else impact_metrics.loc[k]
            )
            for metric, replay in statistics(events, times).items():
                close(recorded[metric], replay[0])
            replayed[label][k] = {"cumulative_return": total, "target_hits": int(round(hit_rate * 80))}
        with np.load(random_directory / "resampled_assets.npz") as saved:
            selections, event_returns, sizes = saved["asset_indices"], saved["event_returns"], saved["portfolio_sizes"]
        for index, event in assets.groupby("event_index", sort=True):
            event = event.sort_values("asset_index")
            orders = selections[:, index]
            assert np.all(np.diff(np.sort(orders, axis=1), axis=1) > 0)
            replay = np.cumsum(event["return"].to_numpy()[orders], axis=1)[:, sizes - 1] / sizes
            close(event_returns[:, index], replay)
        random_metrics = pd.read_csv(random_directory / "metrics.csv").set_index(["k", "replicate"])
        for position, k in enumerate(sizes):
            for metric, replay in statistics(event_returns[:, :, position].T, times).items():
                close(random_metrics.loc[k, metric], replay)
    with (
        np.load(run / "random_portfolio/resampled_assets.npz") as a,
        np.load(impact / "random_portfolio/resampled_assets.npz") as b,
    ):
        np.testing.assert_array_equal(a["asset_indices"], b["asset_indices"])
    report = {
        "status": "PASS",
        "models": len(reconstructed),
        "model_checkpoint_verification": model_verification,
        "events": 80,
        "shared_bootstrap_draws": 10000,
        "paired_comparisons": len(pairs),
        "point_metrics_recomputed_from_predictions": True,
        "bootstrap_replayed": True,
        "portfolio_returns_and_target_hits_replayed": replayed,
        "random_selections_replayed": True,
        "same_predictions_used_throughout": True,
        "annualized_portfolio_statistics_replayed": True,
        "training_subset_summaries_and_feature_quantiles_recomputed": True,
        "temporal_metric_and_l2_grid_choice_verified": True,
        "prediction_sha256": hashlib.sha256((boot / "test_predictions.parquet").read_bytes()).hexdigest(),
    }
    (ROOT / "docs/notebook_rerun_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all seven models, 10,000 shared bootstrap draws, 21 paired comparisons and all portfolios replayed.")


if __name__ == "__main__":
    main()
