"""Verify saved model weights against test predictions and validation metrics."""

from pathlib import Path
import hashlib
import importlib
import json
import pickle
import sys
import numpy as np
import pandas as pd

# The historical notebook namespaces are kept beside the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.utils.sample import DatasetType
from sklearn.metrics import auc, precision_recall_curve, f1_score, balanced_accuracy_score


def audit_models():
    root = Path(__file__).resolve().parents[1]
    run = root / "notebooks/analysis_outputs/paper_cohort_473_l2/paper_run"
    record = json.loads((run / "models/manifest.json").read_text())
    predictions = pd.read_parquet(run.parent / "test_auc_bootstrap/test_predictions.parquet")
    validation = pd.read_csv(run / "tables/validation_metrics.csv", index_col=0)
    modules = {
        "Logistic regression + Tuned": "LogisticRegression",
        "Logistic regression L2 + Tuned": "LogisticRegression",
        "Random forest + Tuned": "RandomForest",
        "CatBoost classifier + Tuned": "CatboostClassifier",
        "CatBoost classifier + SMOTE + Tuned": "CatboostClassifierSMOTE",
        "CatBoost ranker + Tuned": "CatboostRanker",
        "CatBoost classifier + Top@k%AUC early stopping": "CatboostClassifierTOPKAUC",
    }
    assert set(record) == set(modules) == set(validation.index) == set(predictions.model)
    for name, rec in record.items():
        path = run / "models" / rec["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"]
        with path.open("rb") as f:
            model = pickle.load(f)
        module = importlib.import_module("backtest.pipelines." + modules[name] + ".pipe")
        pipe = getattr(module, modules[name] + "Pipeline")()
        sample = pipe.create_sample()
        ds = sample.get_dataset(DatasetType.TEST)
        assert ds.feature_set.regressors == rec["feature_order"]
        expected = predictions.loc[predictions.model.eq(name)]
        data = ds.all_data()
        assert data.pump_hash.tolist() == expected.pump_hash.tolist()
        assert data.currency_pair.astype(str).tolist() == expected.currency_pair.astype(str).tolist()
        np.testing.assert_allclose(model.rank(ds), expected.probas_pred, rtol=0, atol=0)
        validation_ds = sample.get_dataset(DatasetType.VALIDATION)
        v = validation_ds.all_data().copy()
        v["score"] = model.rank(validation_ds)
        ranked = v.sort_values("score", ascending=False, kind="stable")
        target_indices = ranked.groupby("pump_hash", sort=False).head(1).index
        ranks = []
        counts = []
        for _, group in ranked.groupby("pump_hash"):
            ranks.append(np.flatnonzero(group.is_pumped.to_numpy())[0] + 1)
            counts.append(len(group))
        ranks = np.array(ranks)
        counts = np.array(counts)
        bins = np.arange(0, 0.2001, 0.005)
        curve = (ranks[:, None] <= np.ceil(counts[:, None] * bins)).mean(axis=0)
        metrics = {"Top@k%AUC": np.trapezoid(curve, bins) / 0.2}
        for k in [1, 5, 10]:
            metrics[f"Top@{k}"] = (ranks <= k).mean()
        for k in [0.01, 0.05, 0.1]:
            metrics[f"Top@{k*100:g}%"] = (ranks <= np.ceil(counts * k)).mean()
        precision, recall, _ = precision_recall_curve(v.is_pumped, v.score)
        metrics.update(
            {
                "PR-AUC": auc(recall, precision),
                "F1": f1_score(v.is_pumped, v.index.isin(target_indices)),
                "Balanced accuracy": balanced_accuracy_score(v.is_pumped, v.index.isin(target_indices)),
            }
        )
        for key, value in metrics.items():
            np.testing.assert_allclose(value, validation.loc[name, key], rtol=1e-12, atol=1e-12)
        print(name, "matches test predictions and all validation metrics", flush=True)
    (run / "models/verification.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "models": 7,
                "all_test_predictions_exactly_reproduced": True,
                "all_validation_metrics_recomputed": True,
            },
            indent=2,
        )
        + "\n"
    )
    return json.loads((run / "models/verification.json").read_text())


if __name__ == "__main__":
    audit_models()
