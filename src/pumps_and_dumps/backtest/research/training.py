"""Construct named experiments using the existing pipeline training methods."""

from pumps_and_dumps.backtest.pipelines.base_model import TrainingPipeline
from pumps_and_dumps.backtest.experiment import Experiment


def train_baseline_and_tuned(
    pipeline: TrainingPipeline, name: str
) -> tuple[Experiment, Experiment]:
    """Train both variants on the same sample, loading saved studies for tuning."""
    sample = pipeline.create_sample()
    baseline = Experiment(name, pipeline.train(sample=sample, tuned=False), sample)
    tuned = Experiment(
        f"{name} + Tuned", pipeline.train(sample=sample, tuned=True), sample
    )
    return baseline, tuned


def train_experiment(
    pipeline: TrainingPipeline, name: str, *, tuned: bool = True
) -> Experiment:
    """Train one named experiment without changing pipeline settings or splits."""
    sample = pipeline.create_sample()
    return Experiment(name, pipeline.train(sample=sample, tuned=tuned), sample)
