import logging

import optuna
from optuna import Study

from core.paths import SQLITE_URL


def delete_study_if_exists(study_name: str) -> bool:
    """Delete a study if it exists. Returns True if deleted, False if not found."""
    for study in optuna.get_all_study_summaries(storage=SQLITE_URL):
        if study.study_name == study_name:
            logging.info("Deleting study %s", study.study_name)
            optuna.delete_study(study_name=study_name, storage=SQLITE_URL)
            return True
    return False


def create_study(study_name: str) -> Study:
    delete_study_if_exists(study_name=study_name)
    study: Study = optuna.create_study(direction="maximize", storage=SQLITE_URL, study_name=study_name)
    return study
