"""The interface every model plugs in behind."""

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class PropertyInfo:
    id: str
    name: str
    category: str
    units: str
    task_type: Literal["regression", "classification", "descriptor"]
    description: str = ""
    # The model's own reported held-out performance on this property, e.g. {"AUROC": 0.87}.
    benchmark: dict[str, float] = field(default_factory=dict)
    dataset_url: str = ""
    # Physically meaningful bounds, where they exist. A regression model can predict outside
    # these (e.g. a negative half-life), which the UI flags rather than hides.
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    description: str
    citation: str
    paper_url: str
    code_url: str
    license: str


@dataclass(frozen=True)
class Prediction:
    property_id: str
    value: float
    # Percentile of this value among approved drugs, where the model provides a reference set.
    drugbank_percentile: float | None = None
    # Model's own caveat about this molecule, e.g. an element it was not trained on.
    warning: str | None = None


class Predictor(Protocol):
    info: ModelInfo
    properties: list[PropertyInfo]

    def predict(self, smiles: str) -> list[Prediction]: ...
