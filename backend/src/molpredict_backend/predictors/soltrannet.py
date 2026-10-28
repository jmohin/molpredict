"""SolTranNet: a molecule attention transformer for aqueous solubility.

The published package imports `pkg_resources`, removed in Python 3.12, so the model code and
its (60 kB) weights are vendored under `molpredict_backend.vendor.soltrannet` instead.
"""

import threading
from importlib import resources

import torch

from ..vendor.soltrannet.data_utils import featurize_mol
from ..vendor.soltrannet.transformer import make_model
from .base import ModelInfo, Prediction, PropertyInfo

# Column 11 of SolTranNet's one-hot atom features is its catch-all "other element" flag; the
# model saw few such atoms in training, so its own CLI warns when one is present.
_OTHER_ATOM_COLUMN = 11

SOLUBILITY = PropertyInfo(
    id="Solubility_AqSolDB",
    name="Aqueous Solubility",
    category="Absorption",
    units="log(mol/L)",
    task_type="regression",
    dataset_url="https://doi.org/10.1038/s41597-019-0151-1",
)


class SolTranNet:
    info = ModelInfo(
        id="soltrannet",
        name="SolTranNet",
        description=(
            "A molecule attention transformer that predicts aqueous solubility (logS) directly "
            "from the 2D structure. Trained on AqSolDB, the same dataset behind ADMET-AI's "
            "solubility endpoint, so the two are directly comparable."
        ),
        citation=(
            "Francoeur PG, Koes DR. SolTranNet — a machine learning tool for fast aqueous "
            "solubility prediction. Journal of Chemical Information and Modeling 61(6), 2021."
        ),
        paper_url="https://doi.org/10.1021/acs.jcim.1c00331",
        code_url="https://github.com/gnina/SolTranNet",
        license="Apache-2.0",
    )
    properties = [SOLUBILITY]

    def __init__(self) -> None:
        weights = resources.files("molpredict_backend.vendor.soltrannet") / "soltrannet_aqsol_trained.weights"
        self._model = make_model()
        with resources.as_file(weights) as path:
            self._model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self._model.eval()
        self._lock = threading.Lock()

    def predict(self, smiles: str) -> list[Prediction]:
        from rdkit import Chem

        node_features, adjacency = featurize_mol(Chem.MolFromSmiles(smiles), add_dummy_node=True)
        nodes = torch.as_tensor(node_features, dtype=torch.float32).unsqueeze(0)
        adj = torch.as_tensor(adjacency, dtype=torch.float32).unsqueeze(0)
        mask = torch.sum(torch.abs(nodes), dim=-1) != 0

        with self._lock, torch.no_grad():
            value = self._model(nodes, mask, adj, None).flatten().item()

        warning = None
        if nodes[0][:, _OTHER_ATOM_COLUMN].sum():
            warning = "Contains an element SolTranNet saw little of in training."
        return [Prediction(SOLUBILITY.id, value, warning=warning)]
