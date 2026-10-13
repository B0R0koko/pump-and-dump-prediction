import optuna
import pytest

from pumps_and_dumps.backtest.datasets.splitting import MODEL_SELECTION_SPLIT_POLICY
from pumps_and_dumps.backtest.pipelines.study import create_study, validate_study_split


def test_current_study_can_resume_and_retains_split_metadata(tmp_path):
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    study = create_study("current", storage=storage)
    study.optimize(lambda trial: 1.0, n_trials=1)

    resumed = create_study("current", start_new=False, storage=storage)

    validate_study_split(resumed)
    assert resumed.user_attrs["split_policy"] == MODEL_SELECTION_SPLIT_POLICY
    assert resumed.user_attrs["split_fractions"] == {
        "TRAIN": 0.6, "TUNING": 0.1, "VALIDATION": 0.2, "TEST": 0.1,
    }
    assert len(resumed.trials) == 1


def test_old_study_cannot_be_loaded_or_resumed_as_new_split(tmp_path):
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    study = optuna.create_study(study_name="previous", storage=storage)
    study.set_user_attr("split_policy", "frozen_chronological_event_half_v2")
    study.optimize(lambda trial: 1.0, n_trials=1)

    with pytest.raises(ValueError, match="fresh hyperparameter search"):
        validate_study_split(study)
    with pytest.raises(ValueError, match="fresh hyperparameter search"):
        create_study("previous", start_new=False, storage=storage)

    unchanged = optuna.load_study(study_name="previous", storage=storage)
    assert unchanged.user_attrs["split_policy"] == "frozen_chronological_event_half_v2"
    assert len(unchanged.trials) == 1
