"""ADMET-AI: Chemprop-RDKit ensembles trained on the TDC ADMET benchmark group."""

import csv
import logging
import math
import threading
from importlib import resources

from .base import ModelInfo, Prediction, PropertyInfo

_METRICS = ("AUROC", "AUPRC", "R^2", "MAE")


def _bound(raw: str) -> float | None:
    """admet.csv writes unbounded limits as ±inf; treat those as "no bound"."""
    if not raw:
        return None
    value = float(raw)
    return None if math.isinf(value) else value


def _load_properties() -> list[PropertyInfo]:
    path = resources.files("admet_ai") / "resources" / "data" / "admet.csv"
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["category"] != "Physicochemical"]
    return [
        PropertyInfo(
            id=r["id"],
            name=r["name"],
            category=r["category"],
            units=r["units"],
            task_type=r["task_type"],
            benchmark={m: float(r[m]) for m in _METRICS if r[m]},
            dataset_url=r["url"],
            minimum=_bound(r["minimum"]),
            maximum=_bound(r["maximum"]),
        )
        for r in rows
    ]


class ADMETAI:
    info = ModelInfo(
        id="admet_ai",
        name="ADMET-AI",
        description=(
            "Ensembles of Chemprop graph neural networks augmented with RDKit features, "
            "trained on 41 ADMET datasets from the Therapeutics Data Commons."
        ),
        citation=(
            "Swanson K, Walther P, Leitz J, et al. ADMET-AI: a machine learning ADMET platform "
            "for evaluation of large-scale chemical libraries. Bioinformatics 40(7), 2024."
        ),
        paper_url="https://doi.org/10.1093/bioinformatics/btae416",
        code_url="https://github.com/swansonk14/admet_ai",
        license="MIT",
    )

    def __init__(self) -> None:
        for name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
            logging.getLogger(name).setLevel(logging.ERROR)
        from admet_ai import ADMETModel

        self.properties = _load_properties()
        self._model = ADMETModel(include_physchem=False, num_workers=0)
        # Torch inference shares model state; serialize calls from FastAPI's thread pool.
        self._lock = threading.Lock()

    def predict(self, smiles: str) -> list[Prediction]:
        with self._lock:
            raw = self._model.predict(smiles)
        out = []
        for p in self.properties:
            value = raw.get(p.id)
            if value is None or math.isnan(value):
                continue
            pct = raw.get(f"{p.id}_drugbank_approved_percentile")
            out.append(Prediction(p.id, float(value), None if pct is None else float(pct)))
        return out
