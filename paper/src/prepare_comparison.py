"""Prepare the six-model manuscript comparison from the saved notebook run."""

import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "notebooks/analysis_outputs/paper_cohort_473_l2"
RUN = OUTPUT / "paper_run"
BOOT = OUTPUT / "test_auc_bootstrap"
COMPARISON = OUTPUT / "manuscript_comparison"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filtered_pairs(frame, names):
    """Keep the reported model family and apply Holm to its 15 raw p-values."""
    frame = frame.loc[frame.model_a.isin(names) & frame.model_b.isin(names)].copy()
    assert len(frame) == len(names) * (len(names) - 1) // 2 == 15
    ordered = frame.p_value.sort_values(kind="stable")
    corrected = np.minimum(1, np.maximum.accumulate(ordered.to_numpy() * np.arange(len(frame), 0, -1)))
    frame.loc[ordered.index, "p_value_holm"] = corrected
    frame["reject_h0"] = frame.p_value_holm < frame.alpha
    frame["better_model"] = np.where(frame.reject_h0, np.where(frame.observed_diff > 0, frame.model_a, frame.model_b), "")
    return frame.reset_index(drop=True)


def prepare(names, *, check=False):
    models = list(names)
    files = [RUN / "run_manifest.json", RUN / "models/manifest.json", ROOT / "notebooks/analysis_helpers.ipynb", Path(__file__),
             BOOT / "test_predictions.parquet", BOOT / "auc_summary.csv", BOOT / "auc_pairwise.csv",
             BOOT / "event_aucs.csv", BOOT / "auc_distributions.csv"]
    tables = ["validation_metrics.csv", "validation_topk.csv", "validation_topkp.csv", "validation_classification.csv",
              "test_topk.csv", "test_topkp.csv", "test_classification.csv"]
    files += [RUN / "tables" / name for name in tables]
    model_records = json.loads((RUN / "models/manifest.json").read_text())
    files += [RUN / "models" / model_records[name]["file"] for name in models]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in files}
    provenance_path = COMPARISON / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["inputs"] == inputs and previous["model_labels"] == names:
            assert all(sha(COMPARISON / file) == digest for file, digest in previous["outputs"].items())
            return previous
    assert not check, "Regenerate the manuscript comparison with update_results.py"
    (COMPARISON / "tables").mkdir(parents=True, exist_ok=True)
    (COMPARISON / "images").mkdir(exist_ok=True)
    for filename in tables:
        data = pd.read_csv(RUN / "tables" / filename, index_col=0)
        filtered = data.loc[data.index.isin(models)] if "classification" in filename or filename == "validation_metrics.csv" else data[models]
        filtered.to_csv(COMPARISON / "tables" / filename)
    summary = pd.read_csv(BOOT / "auc_summary.csv", index_col=0).loc[lambda df: df.index.isin(models)]
    summary.to_csv(COMPARISON / "tables/auc_summary.csv")
    pairs = filtered_pairs(pd.read_csv(BOOT / "auc_pairwise.csv"), models)
    pairs.to_csv(COMPARISON / "tables/auc_pairwise.csv", index=False)

    # Reuse the notebook's plotting functions with the same saved fitted models.
    helper = ModuleType("_manuscript_plotting")
    sys.modules[helper.__name__] = helper
    notebook = json.loads((ROOT / "notebooks/analysis_helpers.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "analysis_helpers.ipynb:" + cell["id"], "exec"), helper.__dict__)
    helper.silence_missing_data_logs()
    from backtest.utils.sample import DatasetType
    pipelines = {"Logistic regression + Tuned": "LogisticRegression", "Random forest + Tuned": "RandomForest",
                 "CatBoost classifier + Tuned": "CatboostClassifier", "CatBoost classifier + SMOTE + Tuned": "CatboostClassifierSMOTE",
                 "CatBoost ranker + Tuned": "CatboostRanker", "CatBoost classifier + Top@k%AUC early stopping": "CatboostClassifierTOPKAUC"}
    saved_predictions = pd.read_parquet(BOOT / "test_predictions.parquet")
    validation = pd.read_csv(RUN / "tables/validation_metrics.csv", index_col=0)
    curves = {"validation": {}, "test": {}}
    bins = np.arange(0, .21, .01)
    auc_bins = np.arange(0, .2001, .005)
    counts_by_split = {}
    for name in models:
        record = model_records[name]
        with (RUN / "models" / record["file"]).open("rb") as stream:
            model = pickle.load(stream)
        module = importlib.import_module(f"backtest.pipelines.{pipelines[name]}.pipe")
        sample = getattr(module, pipelines[name] + "Pipeline")().create_sample()
        for split, kind in [("validation", DatasetType.VALIDATION), ("test", DatasetType.TEST)]:
            dataset = sample.get_dataset(kind)
            data = dataset.all_data().copy()
            data["score"] = model.rank(dataset)
            if split == "test":
                saved = saved_predictions.loc[saved_predictions.model == name]
                assert data.pump_hash.tolist() == saved.pump_hash.tolist()
                assert data.currency_pair.tolist() == saved.currency_pair.tolist()
                np.testing.assert_array_equal(data.score, saved.probas_pred)
            ranks, counts = [], []
            for _, group in data.sort_values("score", ascending=False, kind="stable").groupby("pump_hash", sort=True):
                assert group.is_pumped.sum() == 1
                ranks.append(np.flatnonzero(group.is_pumped.to_numpy())[0] + 1)
                counts.append(len(group))
            ranks, counts = np.array(ranks)[:, None], np.array(counts)[:, None]
            curves[split][name] = (ranks <= np.ceil(counts * bins)).mean(axis=0)
            auc = np.trapezoid((ranks <= np.ceil(counts * auc_bins)).mean(axis=0), auc_bins) / .2
            expected_auc = validation.loc[name, "Top@k%AUC"] if split == "validation" else summary.loc[name, "point_estimate"]
            np.testing.assert_allclose(auc, expected_auc, atol=1e-12, rtol=1e-12)
            expected = pd.read_csv(RUN / "tables" / f"{split}_topkp.csv", index_col=0)[name]
            for fraction, value in expected.items():
                np.testing.assert_allclose((ranks <= np.ceil(counts * fraction)).mean(), value, atol=1e-12, rtol=1e-12)
            counts_by_split[split] = counts.ravel()
    for split in curves:
        dense = pd.DataFrame(curves[split], index=bins)
        dense.to_csv(COMPARISON / "tables" / f"{split}_topkp_curves.csv")
        topk = pd.read_csv(COMPARISON / "tables" / f"{split}_topk.csv", index_col=0).rename(columns=names)
        random = [(np.minimum(k, counts_by_split[split]) / counts_by_split[split]).mean() for k in topk.index]
        fig = helper.plot_topk_accuracy(topk, list(names.values()), random_baseline=random, bins=topk.index,
                                       save_path=str(COMPARISON / "images" / f"topk_accuracy_{split}.png"))
        plt.close(fig)
        auc_values = validation["Top@k%AUC"] if split == "validation" else summary.point_estimate
        fig = helper.plot_topk_percent_curves(dense.rename(columns=names), list(names.values()),
                                             {names[name]: float(auc_values[name]) for name in models},
                                             save_path=str(COMPARISON / "images" / f"topkp_auc_{split}.png"))
        plt.close(fig)
        classification = pd.read_csv(COMPARISON / "tables" / f"{split}_classification.csv", index_col=0).rename(index=names)
        fig = helper.plot_classification_metrics(classification, list(names.values()), title=f"Classification metrics on {split}",
                                                 save_path=str(COMPARISON / "images" / f"classification_metrics_{split}.png"))
        plt.close(fig)
    result = SimpleNamespace(summary=summary.rename(index=names), alpha=.05,
                             bootstrap_aucs=pd.read_csv(BOOT / "auc_distributions.csv", index_col=0)[models].rename(columns=names),
                             event_aucs=pd.read_csv(BOOT / "event_aucs.csv", index_col=0)[models].rename(columns=names))
    fig = helper.plot_test_auc_distributions(result)
    fig.savefig(COMPARISON / "images/auc_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    output_hashes = {str(p.relative_to(COMPARISON)): sha(p) for folder in ["tables", "images"] for p in sorted((COMPARISON / folder).iterdir()) if p.is_file()}
    provenance = {"model_labels": names, "models": len(models), "paired_comparisons": len(pairs),
                  "same_fitted_models_and_predictions": True, "inputs": inputs, "outputs": output_hashes}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
