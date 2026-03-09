from dataclasses import dataclass

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.utils.sample import Sample


@dataclass
class Experiment:
    experiment_name: str
    model: ImplementsRank
    sample: Sample

    def get_experiment_name(self) -> str:
        return self.experiment_name

    def get_model(self) -> ImplementsRank:
        """Returns our wrapper around underlying model"""
        return self.model

    def get_sample(self) -> Sample:
        return self.sample

    def __str__(self) -> str:
        return self.experiment_name

    def __repr__(self) -> str:
        return f"Experiment<{self.experiment_name}>"
