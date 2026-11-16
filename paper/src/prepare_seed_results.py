"""Build manuscript artifacts from the completed 100-seed experiment.

Seed distributions select the family. The preselected seed 42 supplies point
metrics, event bootstrap, feature importance and every portfolio selection.
Execution outcomes are model-independent and are reused only after matching
the complete event/asset universe and replaying their price-impact arithmetic.
"""

import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
from types import ModuleType, SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, balanced_accuracy_score, f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_snapshots.seed_sweep import DEFAULT_OUTPUT as VALIDATION, MODELS
from research_snapshots.test_seed_sweep import DEFAULT_OUTPUT as TEST

OUTPUT = ROOT / "notebooks/analysis_outputs/seed_sweep_paper"
RUN = COMPARISON = OUTPUT
IMPACT = OUTPUT / "portfolio_1000usdt_slippage"
OLD = ROOT / "notebooks/analysis_outputs/paper_cohort_473_l2"
MODEL_NAMES = {
    "logistic": "Logistic regression + Tuned", "forest": "Random forest + Tuned",
    "catboost": "CatBoost classifier + Tuned", "smote": "CatBoost classifier + SMOTE + Tuned",
    "ranker": "CatBoost ranker + Tuned", "catboost_es": "CatBoost classifier + Top@k%AUC early stopping",
}
SIZES = [1, 2, 5, 10, 20, 30]
FRACTIONS = [.01, .02, .05, .10, .20, .30]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers():
    helper = ModuleType("_seed_paper_plotting")
    sys.modules[helper.__name__] = helper
    notebook = json.loads((ROOT / "notebooks/analysis_helpers.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "analysis_helpers.ipynb:" + cell["id"], "exec"), helper.__dict__)
    helper.silence_missing_data_logs()
    return helper


def ranked_events(predictions):
    ordered = predictions.sort_values("score", ascending=False, kind="stable").copy()
    ordered["rank"] = ordered.groupby("pump_hash", sort=False).cumcount() + 1
    assert ordered.groupby("pump_hash").is_pumped.sum().eq(1).all()
    target = ordered.loc[ordered.is_pumped].set_index("pump_hash")
    target["size"] = ordered.groupby("pump_hash").size().reindex(target.index)
    return ordered, target.sort_index()


def score_tables(predictions):
    ordered, target = ranked_events(predictions)
    ranks, sizes = target["rank"].to_numpy()[:, None], target["size"].to_numpy()[:, None]
    bins = np.arange(0, .2001, .005)
    curve = (ranks <= np.ceil(sizes * bins)).mean(axis=0)
    point = float(np.trapezoid(curve, bins) / .2)
    topk = pd.Series((ranks <= np.array(SIZES)).mean(axis=0), index=SIZES)
    topkp = pd.Series((ranks <= np.ceil(sizes * FRACTIONS)).mean(axis=0), index=FRACTIONS)
    precision, recall, _ = precision_recall_curve(predictions.is_pumped, predictions.score)
    labels = ordered["rank"].eq(1).reindex(predictions.index)
    classification = pd.Series({"PR-AUC": auc(recall, precision),
        "F1 (Top@1 per cross-section)": f1_score(predictions.is_pumped, labels),
        "Balanced Accuracy (Top@1 per cross-section)": balanced_accuracy_score(predictions.is_pumped, labels)})
    return point, topk, topkp, classification, pd.Series(curve, index=bins), target


class SavedScores:
    def rank(self, dataset):
        return dataset.all_data().score.to_numpy()


def saved_experiment(name, predictions):
    from backtest.utils.sample import Dataset, DatasetType
    from backtest.utils.feature_set import FeatureSet
    dataset = Dataset(predictions, FeatureSet(["score"], "is_pumped"), DatasetType.TEST)
    return SimpleNamespace(get_experiment_name=lambda: name, get_model=lambda: SavedScores(),
                           get_sample=lambda: SimpleNamespace(get_dataset=lambda kind: dataset))


def impact_returns(no_impact, impact_1000, quantity):
    """Exact square-root scaling of the saved entry and inventory-sized exit."""
    assert quantity > 0
    a0 = impact_1000.entry_impact_bps.to_numpy() / 1e4
    b0 = impact_1000.exit_impact_bps.to_numpy() / 1e4
    a = a0 * np.sqrt(quantity / 1000)
    b = b0 * np.sqrt((quantity / 1000) * (1 + a0) / (1 + a))
    assert np.all(b < 1), "Re-execute trades if the exit-price floor becomes active"
    gross = no_impact["return"].to_numpy() + 1 + .0025
    values = gross * (1 - b) / (1 + a) - 1 - .0025
    return np.where(no_impact.executed, values, 0)


def portfolio_outputs(predictions, helper, inputs):
    from pumps_and_dumps.backtest.evaluation import compute_portfolio_statistics

    base = OLD / "paper_run"
    old_impact = OLD / "portfolio_1000usdt_slippage"
    keys = ["pump_hash", "currency_pair", "is_pumped"]
    assets = pd.read_parquet(base / "random_portfolio/asset_returns.parquet")
    impacted = pd.read_parquet(old_impact / "asset_returns.parquet")
    pd.testing.assert_frame_equal(assets[keys + ["executed"]], impacted[keys + ["executed"]])
    np.testing.assert_allclose(impact_returns(assets, impacted, 1000), impacted["return"], rtol=1e-11, atol=1e-12)
    joined = predictions[keys + ["score"]].merge(assets, on=keys, validate="one_to_one", how="outer", indicator=True)
    assert joined._merge.eq("both").all()
    joined = joined.sort_values(["pump_hash", "score", "currency_pair"], ascending=[True, False, True], kind="stable")
    joined["rank"] = joined.groupby("pump_hash").cumcount() + 1
    joined[keys + ["event_index", "asset_index", "score", "rank"]].to_parquet(OUTPUT / "portfolio_rankings.parquet", index=False)
    times = assets.groupby("event_index").pump_time.first()
    btc = pd.read_csv(base / "tables/portfolio_returns.csv", index_col=0, parse_dates=True)["BTCUSDT buy-and-hold"]
    old_statistics = pd.read_csv(base / "tables/portfolio_statistics.csv", index_col=0)
    statistics_by_scenario = {}
    for scenario, outcomes, random_dir, destination, image in [
        ("no_impact", assets, base / "random_portfolio", OUTPUT, "portfolios_vs_btc_test.png"),
        ("impact_1000", impacted, old_impact / "random_portfolio", IMPACT, "portfolios_1000usdt_with_slippage.png"),
    ]:
        destination.mkdir(parents=True, exist_ok=True)
        returns = joined.drop(columns="return").merge(outcomes[keys + ["return"]], on=keys, validate="one_to_one")
        events = pd.DataFrame({k: returns.loc[returns["rank"] <= k].groupby("event_index")["return"].sum().reindex(times.index, fill_value=0) / k for k in SIZES})
        events.index = pd.DatetimeIndex(times)
        stats = compute_portfolio_statistics(events)
        stats.insert(1, "cumulative return", events.sum())
        stats["Top@k hit rate"] = [returns.loc[returns["rank"] <= k].groupby("pump_hash").is_pumped.any().mean() for k in SIZES]
        statistics_by_scenario[scenario] = stats
        random_summary = pd.read_csv(random_dir / "summary.csv").set_index(["k", "metric"])
        means = pd.read_csv(random_dir / "mean_event_returns.csv", index_col=0, parse_dates=True)
        paths = pd.read_csv(random_dir / "cumulative_paths.csv")
        daily = events.groupby(events.index.normalize()).sum().reindex(btc.index, fill_value=0)
        daily.columns = [f"K={k}" for k in SIZES]
        random_name = "Random K=5 (mean of 10,000 draws)"
        daily[random_name] = means["5"].groupby(means.index.normalize()).sum().reindex(btc.index, fill_value=0)
        daily[btc.name] = btc
        band = paths.loc[paths.k == 5].copy()
        band.index = pd.to_datetime(band.time).dt.normalize()
        band = band.groupby(level=0)[["lower", "upper"]].last().reindex(btc.index).ffill().fillna(0)
        figure = helper.plot_equity_curves(daily, title="CatBoost portfolios on the test events" + (" with price impact" if scenario == "impact_1000" else ""),
                                          baseline_cols=[btc.name], random_col=random_name, random_band=band)
        figure.savefig(destination / image if scenario == "impact_1000" else OUTPUT / "images" / image, dpi=160, bbox_inches="tight")
        plt.close(figure)
        events.to_csv(destination / "model_event_returns.csv")
        daily.to_csv(destination / "daily_returns.csv")
        stats.to_csv(destination / "model_statistics.csv")
        # Random portfolios are unchanged because their complete execution universe is identical.
        comparison = random_summary.rename(columns={"mean": "random mean"}).copy()
        comparison["model"] = [stats.loc[k, metric] for k, metric in comparison.index]
        comparison.to_csv(destination / "model_vs_random.csv")
        (destination / "random_portfolio").mkdir(exist_ok=True)
        shutil.copy2(random_dir / "summary.csv", destination / "random_portfolio/summary.csv")
        if scenario == "no_impact":
            table = stats.drop(columns="Top@k hit rate").copy()
            table.index = [f"{MODEL_NAMES['catboost']} K={k}" for k in table.index]
            unchanged = old_statistics.loc[[f"Random K={k} (mean)" for k in SIZES] + [btc.name]]
            pd.concat([table, unchanged]).to_csv(OUTPUT / "tables/portfolio_statistics.csv")
            daily.to_csv(OUTPUT / "tables/portfolio_returns.csv")
            shutil.copy2(random_dir / "summary.csv", OUTPUT / "tables/random_portfolio_metrics.csv")
        for filename in ["summary.csv", "metrics.csv", "mean_event_returns.csv", "cumulative_paths.csv", "resampled_assets.npz"]:
            inputs[str((random_dir / filename).relative_to(ROOT))] = sha(random_dir / filename)
    # Replay the archived strategy's full quantity sweep to validate the scaling
    # identity independently before applying it to the selected seed-42 ranks.
    previous = pd.read_parquet(OLD / "test_auc_bootstrap/test_predictions.parquet")
    previous = previous.loc[previous.model == MODEL_NAMES["catboost"]]
    previous = previous.merge(assets[keys + ["event_index", "asset_index"]], on=keys, validate="one_to_one")
    previous = previous.sort_values(["pump_hash", "probas_pred", "currency_pair"], ascending=[True, False, True], kind="stable")
    previous = previous.groupby("pump_hash").head(5)
    archived_sensitivity = pd.read_csv(base / "tables/impact_sensitivity.csv").set_index("quantity_usdt")
    sensitivity = []
    for quantity in [100, 200, 500, 1000, 2000, 5000, 10000]:
        outcomes = assets[keys].copy()
        outcomes["return"] = impact_returns(assets, impacted, quantity)
        replay = previous.merge(outcomes, on=keys, validate="one_to_one")["return"].sum() / 5
        np.testing.assert_allclose(replay, archived_sensitivity.loc[quantity, "cumulative_roe"], rtol=1e-10, atol=1e-12)
        selected = joined.loc[joined["rank"] <= 5, keys].merge(outcomes, on=keys, validate="one_to_one")
        returns = selected.groupby("pump_hash")["return"].sum() / 5
        sensitivity.append({"quantity_usdt": quantity, "num_pumps": 80, "cumulative_roe": returns.sum(),
                            "overall_pnl": returns.sum() * quantity, "mean_roe": returns.mean()})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUTPUT / "tables/impact_sensitivity.csv", index=False)
    np.testing.assert_allclose(sensitivity.set_index("quantity_usdt").loc[1000, "cumulative_roe"], statistics_by_scenario["impact_1000"].loc[5, "cumulative return"])
    figure = helper.plot_pnl_sensitivity(sensitivity, save_path=str(OUTPUT / "images/pnl_vs_quantity_with_impact.png"))
    plt.close(figure)
    return {"execution_universe": len(assets), "impact_scaling_replayed_at_quantities": len(sensitivity),
            "same_model_for_ranking_and_portfolios": True,
            "no_impact_k5": statistics_by_scenario["no_impact"].loc[5].to_dict(),
            "impact_k5": statistics_by_scenario["impact_1000"].loc[5].to_dict()}


def prepare(names, *, check=False):
    from pumps_and_dumps.backtest.robustness.auc_bootstrap import evaluate_test_auc_bootstrap
    from pumps_and_dumps.backtest.robustness.significance import bootstrap_topk_ci, bootstrap_topk_percent_ci
    from pumps_and_dumps.backtest.pipelines.inspection import feature_importances
    from backtest.utils.sample import DatasetType

    selection = json.loads((VALIDATION / "selection.json").read_text())
    assert selection["model"] == "catboost" and selection["final_seed"] == 42
    assert json.loads((TEST / "audit.json").read_text())["status"] == "PASS"
    files = [Path(__file__), ROOT / "notebooks/analysis_helpers.ipynb", VALIDATION / "selection.json",
             VALIDATION / "model_distributions.csv", VALIDATION / "paired_seed_differences.csv", VALIDATION / "validation_seed_distributions.png",
             TEST / "model_distributions.csv", TEST / "paired_seed_differences.csv", TEST / "test_seed_distributions.png",
             TEST / "test_event_metrics.parquet", TEST / "audit.json", VALIDATION / "development.parquet", TEST / "test.parquet",
             OLD / "paper_run/run_manifest.json", OLD / "paper_run/random_portfolio/asset_returns.parquet",
             OLD / "portfolio_1000usdt_slippage/asset_returns.parquet", OLD / "paper_run/tables/impact_sensitivity.csv",
             OLD / "paper_run/tables/portfolio_statistics.csv", OLD / "paper_run/tables/portfolio_returns.csv",
             OLD / "test_auc_bootstrap/test_predictions.parquet"]
    for key in MODELS:
        directory = TEST / "fits" / key / "seed_042"
        files.extend(directory / filename for filename in ["result.json", "model.pickle", "validation_predictions.parquet", "test_predictions.parquet"])
    files.extend(sorted((TEST / "fits/catboost").glob("seed_*/model.pickle")))
    for directory in [OLD / "paper_run/random_portfolio", OLD / "portfolio_1000usdt_slippage/random_portfolio"]:
        files.extend(directory / filename for filename in ["summary.csv", "metrics.csv", "mean_event_returns.csv", "cumulative_paths.csv", "resampled_assets.npz"])
    descriptive = ["cs_size.jpeg", "cs_standardization_impact.png", "impact_regression_example.png", "exit_impact_regression_example.png"]
    files.extend(OLD / "paper_run/images" / filename for filename in descriptive)
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in files}
    provenance_path = OUTPUT / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["inputs"] == inputs:
            assert all(sha(OUTPUT / path) == checksum for path, checksum in previous["outputs"].items())
            return previous
    assert not check, "Prepare the seed-sweep manuscript artifacts before checking"
    for directory in [OUTPUT / "tables", OUTPUT / "images", OUTPUT / "robustness"]:
        directory.mkdir(parents=True, exist_ok=True)
    helper = load_helpers()
    predictions = {}
    parameters = {}
    for split in ["validation", "test"]:
        results = {}
        for key, name in MODEL_NAMES.items():
            directory = TEST / "fits" / key / "seed_042"
            record = json.loads((directory / "result.json").read_text())
            data = pd.read_parquet(directory / f"{split}_predictions.parquet")
            results[name] = score_tables(data)
            np.testing.assert_allclose(results[name][0], record[f"{split}_auc"], rtol=1e-12, atol=1e-12)
            if split == "test":
                predictions[name] = data
                parameters[name] = record["parameters"]
        topk = pd.DataFrame({name: values[1] for name, values in results.items()})
        topkp = pd.DataFrame({name: values[2] for name, values in results.items()})
        classification = pd.DataFrame({name: values[3] for name, values in results.items()}).T
        curves = pd.DataFrame({name: values[4] for name, values in results.items()})
        points = pd.Series({name: values[0] for name, values in results.items()})
        for title, frame in [("topk", topk), ("topkp", topkp), ("classification", classification), ("topkp_curves", curves)]:
            frame.to_csv(OUTPUT / "tables" / f"{split}_{title}.csv")
        if split == "validation":
            pd.DataFrame({"Top@k%AUC": points}).to_csv(OUTPUT / "tables/validation_metrics.csv")
        sizes = next(iter(results.values()))[5]["size"].to_numpy()
        random = [(np.minimum(k, sizes) / sizes).mean() for k in SIZES]
        fig = helper.plot_topk_accuracy(topk.rename(columns=names), list(names.values()), random_baseline=random, bins=SIZES,
                                       save_path=str(OUTPUT / "images" / f"topk_accuracy_{split}.png"))
        plt.close(fig)
        random_fraction = np.minimum(np.ceil(sizes[:, None] * curves.index.to_numpy()), sizes[:, None]) / sizes[:, None]
        fig = helper.plot_topk_percent_curves(curves.rename(columns=names), list(names.values()),
                    {names[name]: value for name, value in points.items()}, random_baseline=random_fraction.mean(axis=0),
                    save_path=str(OUTPUT / "images" / f"topkp_auc_{split}.png"))
        plt.close(fig)
        fig = helper.plot_classification_metrics(classification.rename(index=names), list(names.values()), title=f"{split.title()} classification metrics (seed 42)",
                    save_path=str(OUTPUT / "images" / f"classification_metrics_{split}.png"))
        plt.close(fig)
    bootstrap = evaluate_test_auc_bootstrap([saved_experiment(name, data) for name, data in predictions.items()], dataset_type=DatasetType.TEST)
    helper.export_test_auc_bootstrap(bootstrap, OUTPUT / "test_auc_bootstrap", provenance={"experiment": str(TEST), "fixed_seed": 42})
    for filename in ["auc_summary.csv", "auc_pairwise.csv"]:
        shutil.copy2(OUTPUT / "test_auc_bootstrap" / filename, OUTPUT / "tables" / filename)
    display = SimpleNamespace(summary=bootstrap.summary.rename(index=names), alpha=.05,
        bootstrap_aucs=bootstrap.bootstrap_aucs.rename(columns=names), event_aucs=bootstrap.event_aucs.rename(columns=names))
    fig = helper.plot_test_auc_distributions(display)
    fig.savefig(OUTPUT / "images/auc_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    selected = MODEL_NAMES["catboost"]
    scored = bootstrap.scored_datasets[selected]
    topk_ci = bootstrap_topk_ci(scored, bins=SIZES, n_bootstrap=10000, random_state=42)
    topkp_ci = bootstrap_topk_percent_ci(scored, bins=FRACTIONS[:-1], n_bootstrap=10000, random_state=42)
    topk_ci.to_csv(OUTPUT / "tables/selected_topk_ci.csv")
    topkp_ci.to_csv(OUTPUT / "tables/selected_topkp_ci.csv")
    fig = helper.plot_bootstrap_ci(topk_ci, topkp_ci, save_path=str(OUTPUT / "images/significance_bootstrap_ci.png"))
    plt.close(fig)
    importance = []
    for seed in range(1, 101):
        with (TEST / "fits/catboost" / f"seed_{seed:03d}/model.pickle").open("rb") as stream:
            model = pickle.load(stream)
        values = feature_importances(model)
        importance.append(pd.DataFrame({"seed": seed, "feature": values.index, "feature_importance": values.to_numpy()}))
        if seed == 42:
            values.to_csv(OUTPUT / "tables/selected_model_feature_importances.csv")
            fig = helper.plot_feature_importances(SimpleNamespace(get_model=lambda: model, get_experiment_name=lambda: "CatBoost (seed 42)"),
                                                 save_path=OUTPUT / "images/feature_importances.png")
            plt.close(fig)
    importance = pd.concat(importance, ignore_index=True)
    importance.to_csv(OUTPUT / "robustness/seed_feature_importances.csv", index=False)
    importance.groupby("feature").feature_importance.quantile([.1, .25, .5, .75, .9]).unstack().rename(
        columns={.1: "p10", .25: "p25", .5: "p50", .75: "p75", .9: "p90"}).to_csv(OUTPUT / "tables/feature_importance_quantiles.csv")
    event_scores = pd.read_parquet(TEST / "test_event_metrics.parquet").query("model == 'catboost'")
    rows = []
    for seed, data in event_scores.groupby("seed"):
        row = {"seed": seed, "topk_percent_auc": data.topkp_auc.mean()}
        row.update({f"topk_percent@{p}": float((data.target_rank <= np.ceil(data.eligible_assets * p)).mean()) for p in FRACTIONS[:-1]})
        rows.append(row)
    robust = pd.DataFrame(rows).set_index("seed")
    robust.to_csv(OUTPUT / "robustness/seed_metrics.csv")
    robust.describe(percentiles=[.1, .25, .5, .75, .9]).T.to_csv(OUTPUT / "tables/robustness_summary.csv")
    fig = helper.plot_robustness_distribution(robust, [f"topk_percent@{p}" for p in FRACTIONS[:-1]], save_path=str(OUTPUT / "images/robustness_metrics_distribution.png"))
    plt.close(fig)
    _, targets = ranked_events(predictions[selected])
    early = predictions[selected].loc[predictions[selected].pump_time < pd.Timestamp("2022-06-01")]
    early_score = score_tables(early)[0]
    pd.DataFrame([{"subperiod": "early", "n_pumps": early.pump_hash.nunique(), "topk_percent_auc": early_score}]).set_index("subperiod").to_csv(OUTPUT / "tables/subperiod_metrics.csv")
    for root, split in [(VALIDATION, "validation"), (TEST, "test")]:
        shutil.copy2(root / "model_distributions.csv", OUTPUT / "tables" / f"{split}_seed_distributions.csv")
        shutil.copy2(root / "paired_seed_differences.csv", OUTPUT / "tables" / f"{split}_paired_seed_differences.csv")
        shutil.copy2(root / f"{split}_seed_distributions.png", OUTPUT / "images" / f"{split}_seed_distributions.png")
    execution = portfolio_outputs(predictions[selected], helper, inputs)
    manifest = json.loads((OLD / "paper_run/run_manifest.json").read_text())
    manifest = {"split_counts": manifest["split_counts"], "model_parameters": parameters,
        "selection": selection, "reproducibility": {"random_seed": 42, "model_threads": 1, "optuna_n_jobs": 1,
            "tuning_and_training_seeds": list(range(1, 101)), "trials_per_seed": 25,
            "point_estimate_seed": 42, "resampling_seed": 42},
        "validation_experiment": str(VALIDATION), "test_experiment": str(TEST)}
    for split, frame in [("TRAIN", pd.read_parquet(VALIDATION / "development.parquet").query("pump_time < '2020-09-01'")),
                         ("VALIDATION", pd.read_parquet(VALIDATION / "development.parquet").query("pump_time >= '2020-09-01'")),
                         ("TEST", pd.read_parquet(TEST / "test.parquet"))]:
        assert len(frame) == manifest["split_counts"][split]["rows"]
        assert frame.pump_hash.nunique() == manifest["split_counts"][split]["events"]
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "tables/selected_model_analysis.json").write_text(json.dumps({"model": selected, "parameters": parameters[selected], "seed": 42}, indent=2) + "\n")
    for filename in descriptive:
        shutil.copy2(OLD / "paper_run/images" / filename, OUTPUT / "images" / filename)
    outputs = {str(path.relative_to(OUTPUT)): sha(path) for path in OUTPUT.rglob("*") if path.is_file() and path != provenance_path}
    provenance = {"inputs": inputs, "outputs": outputs, "models": 6, "seeds": 100, "paired_comparisons": 15,
                  "selection": "Highest mean validation AUC; fixed seed 42 for point, event-bootstrap and portfolio results",
                  "same_fitted_models_and_predictions": True, "execution": execution}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print("Prepared all seed-sweep manuscript tables, bootstrap results, feature importances and portfolios.", flush=True)
    return provenance


if __name__ == "__main__":
    prepare({name: MODELS[key][1] for key, name in MODEL_NAMES.items()})
