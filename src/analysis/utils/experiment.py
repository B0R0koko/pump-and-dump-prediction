from dataclasses import dataclass

from analysis.pipelines.BaseModel import ImplementsRank
from analysis.utils.sample import Sample


@dataclass
class Experiment:
    experiment_name: str
    model: ImplementsRank
    sample: Sample
