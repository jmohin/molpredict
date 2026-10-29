"""MolCLR fine-tuned on EPA/NIEHS OPERA data.

Unlike the other models here, these weights are ours: MolCLR's self-supervised GIN encoder
(pretrained on 10M unlabelled molecules) fine-tuned on OPERA's curated experimental data, one
model per endpoint. `training/` holds the code that produced them.

Each checkpoint carries the held-out score it earned on a scaffold split, and that is what the
UI shows — so the number beside a prediction is the one this model actually achieved, not a
figure copied from a paper about a different model.
"""

import threading
from dataclasses import replace
from importlib import resources

import torch

from ..vendor.molclr.featurize import Unfeaturizable, featurize_smiles
from ..vendor.molclr.model import GINet
from .base import ModelInfo, Prediction, PropertyInfo

# Scores the training run reported, keyed by the metric the UI should label them with.
_METRIC_FOR_TASK = {"regression": "R^2", "classification": "AUROC"}

_PROPERTIES = {
    "boiling_point": PropertyInfo(
        id="boiling_point",
        name="Boiling Point",
        category="Physicochemical",
        units="°C",
        task_type="regression",
        description="At 760 mmHg.",
        dataset_url="https://doi.org/10.1186/s13321-018-0263-1",
        minimum=-273.15,
    ),
    "melting_point": PropertyInfo(
        id="melting_point",
        name="Melting Point",
        category="Physicochemical",
        units="°C",
        task_type="regression",
        description=(
            "The weakest of these models. Melting point depends on how molecules pack in a "
            "crystal, which a 2D graph cannot see — symmetric molecules like naphthalene are "
            "badly under-predicted. Read it alongside its error, not on its own."
        ),
        dataset_url="https://doi.org/10.1186/s13321-018-0263-1",
        minimum=-273.15,
    ),
    "vapor_pressure": PropertyInfo(
        id="vapor_pressure",
        name="Vapor Pressure",
        category="Physicochemical",
        units="log10(mmHg)",
        task_type="regression",
        description="Experimental values measured at 15–30 °C.",
        dataset_url="https://doi.org/10.1186/s13321-018-0263-1",
    ),
    "ready_biodegradability": PropertyInfo(
        id="ready_biodegradability",
        name="Ready Biodegradability",
        category="Environmental",
        units="probability",
        task_type="classification",
        description=(
            "Probability a substance passes a ready-biodegradability test. OPERA's label is "
            "binary pass/fail pooled across the OECD 301 variants, not percent degradation at "
            "28 days under 301F specifically."
        ),
        dataset_url="https://doi.org/10.1186/s13321-018-0263-1",
        minimum=0.0,
        maximum=1.0,
    ),
    "rat_oral_ld50": PropertyInfo(
        id="rat_oral_ld50",
        name="Rat Oral LD50",
        category="Toxicity",
        units="log10(mg/kg)",
        task_type="regression",
        description="Higher means less acutely toxic. Trained on CATMoS measured values only.",
        dataset_url="https://doi.org/10.1186/s13321-020-00404-z",
    ),
}

# Kept apart from the endpoints above because the data behind them is licensed differently.
_PKA_PROPERTIES = {
    "pka_acidic": PropertyInfo(
        id="pka_acidic",
        name="Most Acidic pKa",
        category="Physicochemical",
        units="pKa",
        task_type="regression",
        description=(
            "First acid dissociation (pK1) — the molecule giving up a proton. Separate from the "
            "basic pKa because they are different quantities; a single number per molecule is "
            "not well defined. The model answers for any structure, so for a molecule with no "
            "acidic proton the number is meaningless rather than large."
        ),
        dataset_url="https://doi.org/10.5281/zenodo.7236453",
    ),
    "pka_basic": PropertyInfo(
        id="pka_basic",
        name="Most Basic pKa",
        category="Physicochemical",
        units="pKa",
        task_type="regression",
        description=(
            "First protonation of a base (pKAH1), given as the pKa of its conjugate acid. As "
            "with the acidic value, a molecule with no basic site still gets a number."
        ),
        dataset_url="https://doi.org/10.5281/zenodo.7236453",
    ),
}


class _FineTuned:
    """Shared loading for the models trained in `training/`."""

    _catalogue: dict[str, PropertyInfo] = {}

    def __init__(self) -> None:
        self._models: dict[str, tuple[GINet, float, float, str]] = {}
        properties = []
        files = resources.files("molpredict_backend.models")
        for name, prop in self._catalogue.items():
            with resources.as_file(files / f"{name}.pt") as path:
                blob = torch.load(path, map_location="cpu", weights_only=True)
            model = GINet(task=blob["task"], drop_ratio=0.0)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            self._models[name] = (model, blob["target_mean"], blob["target_std"], blob["task"])

            metric = _METRIC_FOR_TASK[blob["task"]]
            properties.append(replace(prop, benchmark={metric: round(blob["metrics"][metric], 3)}))
        self.properties = properties
        # torch inference shares module state across FastAPI's thread pool.
        self._lock = threading.Lock()

    def predict(self, smiles: str) -> list[Prediction]:
        try:
            x, edge_index, edge_attr = featurize_smiles(smiles)
        except Unfeaturizable:
            return []

        nodes = torch.from_numpy(x)
        edges = torch.from_numpy(edge_index)
        attrs = torch.from_numpy(edge_attr)
        batch = torch.zeros(len(x), dtype=torch.long)

        predictions = []
        with self._lock, torch.no_grad():
            for name, (model, mean, std, task) in self._models.items():
                _, out = model(nodes, edges, attrs, batch)
                if task == "classification":
                    value = torch.softmax(out, dim=1)[0, 1].item()
                else:
                    value = out.squeeze().item() * std + mean
                predictions.append(Prediction(name, float(value)))
        return predictions


class MolCLR(_FineTuned):
    _catalogue = _PROPERTIES
    info = ModelInfo(
        id="molclr",
        name="MolCLR (fine-tuned here)",
        description=(
            "Self-supervised graph neural network pretrained on 10M molecules, fine-tuned by "
            "this project on EPA/NIEHS OPERA experimental data — one model per property."
        ),
        citation=(
            "Wang, Y.; Wang, J.; Cao, Z.; Barati Farimani, A. Molecular contrastive learning of "
            "representations via graph neural networks. Nat. Mach. Intell. 2022, 4, 279–287. "
            "Fine-tuned on data from Mansouri, K. et al. OPERA models. J. Cheminform. 2018, 10, 10."
        ),
        paper_url="https://doi.org/10.1038/s42256-022-00447-x",
        code_url="https://github.com/yuyangw/MolCLR",
        license="MIT (encoder); weights fine-tuned here",
    )


class MolCLRpKa(_FineTuned):
    """Separate from MolCLR above so the licence on these two is visible, not buried.

    Their training data is CC BY-NC: these weights may not be used commercially, while
    everything else in molpredict may.
    """

    _catalogue = _PKA_PROPERTIES
    info = ModelInfo(
        id="molclr_pka",
        name="MolCLR pKa (fine-tuned here)",
        description=(
            "The same pretrained encoder, fine-tuned on IUPAC's digitised pKa compilation. "
            "Acidic and basic pKa are modelled separately, since one number per molecule is "
            "not a well-defined quantity."
        ),
        citation=(
            "Zheng, J. W. IUPAC Digitized pKa Dataset, v1.0, 2022. Copyright © 2022 "
            "International Union of Pure and Applied Chemistry. Reproduced by permission of "
            "IUPAC. Encoder: Wang, Y. et al. Nat. Mach. Intell. 2022, 4, 279–287."
        ),
        paper_url="https://doi.org/10.5281/zenodo.7236453",
        code_url="https://github.com/jonwzheng/Dissociation-Constants-2",
        license="CC BY-NC 4.0 (data) — non-commercial use only",
    )
