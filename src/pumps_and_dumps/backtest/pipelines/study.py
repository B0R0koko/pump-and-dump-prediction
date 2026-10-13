import logging
from typing import Literal

import optuna
from optuna import Study
from optuna.samplers import TPESampler

from pumps_and_dumps.core.paths import SQLITE_URL
from pumps_and_dumps.backtest.datasets.splitting import MODEL_SELECTION_SPLIT_POLICY, SPLIT_FRACTIONS


def validate_study_split(study: Study) -> None:
    """Reject hyperparameters selected using another partition of the events."""
    if study.user_attrs.get("split_policy") != MODEL_SELECTION_SPLIT_POLICY:
        raise ValueError(
            f"Study {study.study_name!r} does not use the 60/10/20/10 split. "
            "Run a fresh hyperparameter search for the current split."
        )


def delete_study_if_exists(study_name: str, *, storage: str | None = None) -> bool:
    """Delete a study if it exists. Returns True if deleted, False if not found."""
    storage = SQLITE_URL if storage is None else storage
    for study in optuna.get_all_study_summaries(storage=storage):
        if study.study_name == study_name:
            logging.info("Deleting study %s", study.study_name)
            optuna.delete_study(study_name=study_name, storage=storage)
            return True
    return False


def create_study(
    study_name: str,
    start_new: bool = True,
    *,
    storage: str | None = None,
    direction: Literal["maximize", "minimize"] = "maximize",
) -> Study:
    storage = SQLITE_URL if storage is None else storage
    if start_new:
        delete_study_if_exists(study_name=study_name, storage=storage)
    study: Study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=42),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    if study.trials:
        validate_study_split(study)
    study.set_user_attr("split_policy", MODEL_SELECTION_SPLIT_POLICY)
    study.set_user_attr("split_fractions", {kind.name: fraction for kind, fraction in SPLIT_FRACTIONS.items()})
    return study
