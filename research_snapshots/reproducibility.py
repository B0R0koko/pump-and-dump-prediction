"""Shared randomness and execution settings for the primary research notebook."""

import os
import platform
import random
from importlib.metadata import version

import numpy as np
from threadpoolctl import threadpool_limits

RANDOM_SEED = 42
OPTUNA_N_JOBS = 1
MODEL_THREADS = int(os.environ.get("PUMPS_AND_DUMPS_MODEL_THREADS", "1"))
if MODEL_THREADS < 1:
    raise ValueError("PUMPS_AND_DUMPS_MODEL_THREADS must be positive")

_thread_limiter = None


def reproducibility_settings() -> dict:
    """Record the settings needed to repeat fresh searches and fixed-parameter fits."""
    return {
        "random_seed": RANDOM_SEED,
        "optuna_sampler": "TPESampler",
        "optuna_n_jobs": OPTUNA_N_JOBS,
        "model_threads": MODEL_THREADS,
        "catboost_task_type": "CPU",
        "python": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not set; feature order is explicit"),
        "versions": {name: version(name) for name in (
            "numpy", "pandas", "polars", "scipy", "scikit-learn", "catboost",
            "imbalanced-learn", "optuna", "numba", "statsmodels", "threadpoolctl",
        )},
        "search_replay": "Start a fresh study with identical inputs, trial count, search space and software versions.",
    }


def initialize_reproducibility() -> dict:
    """Seed global generators and limit native numerical libraries in this kernel.

    Estimators, SMOTE and Optuna also receive explicit integer seeds. Setting
    PYTHONHASHSEED here would not change the running interpreter; the batch
    launcher sets it before starting the kernel instead.
    """
    global _thread_limiter
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    _thread_limiter = threadpool_limits(limits=MODEL_THREADS)
    return reproducibility_settings()
