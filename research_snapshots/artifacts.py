"""Keep fitted model artifacts alongside predictions and input provenance."""

import hashlib
import json
from pathlib import Path
import pickle
import re


def save_models(experiments, directory):
    directory = Path(directory) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for experiment in experiments:
        name = experiment.get_experiment_name()
        filename = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".pickle"
        path = directory / filename
        with path.open("wb") as stream:
            pickle.dump(experiment.get_model(), stream, protocol=pickle.HIGHEST_PROTOCOL)
        records[name] = {
            "file": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_order": next(experiment.get_sample().iter_datasets())[1].feature_set.regressors,
        }
    (directory / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
