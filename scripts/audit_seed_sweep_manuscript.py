"""Independently replay the seed-sweep manuscript's point results and portfolios."""

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, balanced_accuracy_score, f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper/src"))
from prepare_seed_results import OUTPUT, OLD, TEST, VALIDATION, MODEL_NAMES, SIZES, sha
from audit_notebook_results import close, statistics


def main():
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    for path, checksum in provenance["inputs"].items():
        assert sha(ROOT / path) == checksum
    for path, checksum in provenance["outputs"].items():
        assert sha(OUTPUT / path) == checksum
    boot = OUTPUT / "test_auc_bootstrap"
    with np.load(boot / "resampled_events.npz") as saved:
        event_ids, draws = saved["event_ids"], saved["sampled_indices"]
    assert len(event_ids) == 80 and draws.shape == (10000, 80)
    np.testing.assert_array_equal(draws, np.random.default_rng(42).integers(0, 80, size=(10000, 80)))
    bins = np.arange(0, .2001, .005)
    contributions = pd.read_csv(boot / "event_aucs.csv", index_col=0)
    distributions = pd.read_csv(boot / "auc_distributions.csv", index_col=0)
    summary = pd.read_csv(boot / "auc_summary.csv", index_col=0)
    source_test_predictions = pd.read_parquet(boot / "test_predictions.parquet")
    reconstructed = {}
    for split in ["validation", "test"]:
        topk = pd.read_csv(OUTPUT / "tables" / f"{split}_topk.csv", index_col=0)
        topkp = pd.read_csv(OUTPUT / "tables" / f"{split}_topkp.csv", index_col=0)
        classification = pd.read_csv(OUTPUT / "tables" / f"{split}_classification.csv", index_col=0)
        dense = pd.read_csv(OUTPUT / "tables" / f"{split}_topkp_curves.csv", index_col=0)
        for key, name in MODEL_NAMES.items():
            source = TEST / "fits" / key / "seed_042"
            scores = pd.read_parquet(source / f"{split}_predictions.parquet")
            ordered = scores.sort_values("score", ascending=False, kind="stable")
            ranks, counts, values = [], [], []
            groups = dict(tuple(ordered.groupby("pump_hash", sort=True)))
            ids = event_ids if split == "test" else sorted(groups)
            for event in ids:
                group = groups[event]
                rank = np.flatnonzero(group.is_pumped.to_numpy())[0] + 1
                ranks.append(rank)
                counts.append(len(group))
                values.append(np.trapezoid((rank <= np.ceil(bins * len(group))).astype(float), bins) / .2)
            ranks, counts, values = np.array(ranks), np.array(counts), np.array(values)
            for k in topk.index:
                close(topk.loc[k, name], (ranks <= k).mean())
            for fraction in topkp.index:
                close(topkp.loc[fraction, name], (ranks <= np.ceil(counts * fraction)).mean())
            close(dense[name], (ranks[:, None] <= np.ceil(counts[:, None] * dense.index.to_numpy())).mean(axis=0))
            precision, recall, _ = precision_recall_curve(scores.is_pumped, scores.score)
            chosen = ordered.groupby("pump_hash").head(1).index
            decision = scores.index.isin(chosen)
            close(classification.loc[name, "PR-AUC"], auc(recall, precision))
            close(classification.loc[name, "F1 (Top@1 per cross-section)"], f1_score(scores.is_pumped, decision))
            close(classification.loc[name, "Balanced Accuracy (Top@1 per cross-section)"], balanced_accuracy_score(scores.is_pumped, decision))
            record = json.loads((source / "result.json").read_text())
            close(values.mean(), record[f"{split}_auc"])
            if split == "test":
                close(contributions[name], values)
                samples = values[draws].mean(axis=1)
                close(distributions[name], samples)
                close(summary.loc[name, "point_estimate"], values.mean())
                close(summary.loc[name, ["ci_lower", "ci_upper"]].astype(float), np.quantile(samples, [.025, .975]))
                exported = source_test_predictions.loc[source_test_predictions.model == name].set_index(["pump_hash", "currency_pair"])
                expected = scores.set_index(["pump_hash", "currency_pair"])
                close(exported.probas_pred.reindex(expected.index), expected.score)
                reconstructed[name] = values
    pairs = pd.read_csv(boot / "auc_pairwise.csv")
    assert len(pairs) == 15
    for row in pairs.itertuples():
        difference = reconstructed[row.model_a] - reconstructed[row.model_b]
        estimate = difference.mean()
        samples = distributions[row.model_a] - distributions[row.model_b]
        close(row.observed_diff, estimate)
        close([row.ci_lower, row.ci_upper], np.quantile(samples, [.025, .975]))
        p = (1 + (np.abs(samples - estimate) >= abs(estimate) - 32 * np.finfo(float).eps).sum()) / 10001
        close(row.p_value, p)
    ordered_p = pairs.p_value.sort_values(kind="stable")
    close(pairs.loc[ordered_p.index, "p_value_holm"], np.minimum(1, np.maximum.accumulate(ordered_p.to_numpy() * np.arange(15, 0, -1))))
    for split, source in [("validation", VALIDATION), ("test", TEST)]:
        recorded = pd.read_csv(OUTPUT / "tables" / f"{split}_seed_distributions.csv", index_col=0)
        expected = pd.read_csv(source / "model_distributions.csv", index_col=0)
        pd.testing.assert_frame_equal(recorded, expected)
        pd.testing.assert_frame_equal(pd.read_csv(OUTPUT / "tables" / f"{split}_paired_seed_differences.csv"),
                                      pd.read_csv(source / "paired_seed_differences.csv"))
    selected_key = json.loads((VALIDATION / "selection.json").read_text())["model"]
    assert selected_key == "catboost"
    selected = MODEL_NAMES[selected_key]
    scores = pd.read_parquet(TEST / "fits/catboost/seed_042/test_predictions.parquet")
    replayed = {}
    for scenario, outcomes, random_dir, destination in [
        ("no_impact", OLD / "paper_run/random_portfolio/asset_returns.parquet", OLD / "paper_run/random_portfolio", OUTPUT),
        ("impact_1000", OLD / "portfolio_1000usdt_slippage/asset_returns.parquet", OLD / "portfolio_1000usdt_slippage/random_portfolio", OUTPUT / "portfolio_1000usdt_slippage"),
    ]:
        assets = pd.read_parquet(outcomes)
        joined = scores[["pump_hash", "currency_pair", "is_pumped", "score"]].merge(assets, on=["pump_hash", "currency_pair", "is_pumped"], how="outer", validate="one_to_one", indicator=True)
        assert joined._merge.eq("both").all()
        joined = joined.sort_values(["pump_hash", "score", "currency_pair"], ascending=[True, False, True], kind="stable")
        joined["rank"] = joined.groupby("pump_hash").cumcount() + 1
        times = assets.groupby("event_index").pump_time.first()
        metrics = pd.read_csv(destination / "model_statistics.csv", index_col=0)
        daily = pd.read_csv(destination / "daily_returns.csv", index_col=0, parse_dates=True)
        replayed[scenario] = {}
        for k in SIZES:
            chosen = joined.loc[joined["rank"] <= k]
            events = chosen.groupby("event_index")["return"].sum().reindex(times.index, fill_value=0) / k
            hits = chosen.groupby("pump_hash").is_pumped.any().mean()
            topk = pd.read_csv(OUTPUT / "tables/test_topk.csv", index_col=0)
            close(hits, topk.loc[k, selected])
            for metric, value in statistics(events, times).items():
                close(metrics.loc[k, metric], value[0])
            path = pd.Series(events.to_numpy(), index=pd.DatetimeIndex(times).normalize()).groupby(level=0).sum().reindex(daily.index, fill_value=0)
            close(path, daily[f"K={k}"])
            replayed[scenario][k] = {"return": float(events.sum()), "target_hits": round(hits * 80)}
        with np.load(random_dir / "resampled_assets.npz") as stored:
            choices, event_returns, sizes = stored["asset_indices"], stored["event_returns"], stored["portfolio_sizes"]
        for i, group in assets.groupby("event_index", sort=True):
            vector = group.sort_values("asset_index")["return"].to_numpy()
            replay = np.cumsum(vector[choices[:, i]], axis=1)[:, sizes - 1] / sizes
            close(replay, event_returns[:, i])
        means = event_returns[:, :, list(sizes).index(5)].mean(axis=0)
        path = pd.Series(means, index=pd.DatetimeIndex(times).normalize()).groupby(level=0).sum().reindex(daily.index, fill_value=0)
        close(path, daily["Random K=5 (mean of 10,000 draws)"])
    importance = pd.read_csv(OUTPUT / "robustness/seed_feature_importances.csv")
    assert importance.seed.nunique() == 100
    quantiles = pd.read_csv(OUTPUT / "tables/feature_importance_quantiles.csv", index_col=0)
    for col, q in [("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9)]:
        close(quantiles[col], importance.groupby("feature").feature_importance.quantile(q).reindex(quantiles.index))
    for seed in range(1, 101):
        with (TEST / "fits/catboost" / f"seed_{seed:03d}/model.pickle").open("rb") as stream:
            model = pickle.load(stream)
        values = importance.loc[importance.seed == seed].set_index("feature").feature_importance
        close(values.reindex(model._model.feature_names_), model._model.feature_importances_)
    metrics = pd.read_csv(OUTPUT / "robustness/seed_metrics.csv", index_col=0)
    summary = pd.read_csv(OUTPUT / "tables/robustness_summary.csv", index_col=0)
    close(summary, metrics.describe(percentiles=[.1, .25, .5, .75, .9]).T)
    assert replayed["no_impact"][5]["target_hits"] == replayed["impact_1000"][5]["target_hits"] == 34
    report = {"status": "PASS", "families": 6, "seeds_per_family": 100, "point_estimate_seed": 42,
        "event_bootstrap_replicates": 10000, "event_bootstrap_comparisons": 15,
        "validation_and_test_point_metrics_replayed": True, "event_bootstrap_replayed": True,
        "seed_tables_match_audited_experiments": True, "feature_importance_models_verified": 100,
        "random_portfolio_selections_replayed": True, "portfolio_results": replayed,
        "impact_scaling": provenance["execution"], "artifact_provenance_sha256": sha(OUTPUT / "provenance.json")}
    (ROOT / "docs/seed_sweep_manuscript_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: seed tables, six point models, 10,000 event draws, 15 pairs, 100 importance models and all portfolios verified.")


if __name__ == "__main__":
    main()
