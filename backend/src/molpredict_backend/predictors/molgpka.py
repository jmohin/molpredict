"""MolGpKa: pKa for each ionisable site, which is what a charge-vs-pH curve needs.

Every other pKa model here returns one number per molecule. MolGpKa finds the ionisable centres
by substructure match and predicts a pKa for each one, so a molecule with an amine and a
carboxyl gets both — the per-site values the charge curve is built from.

It reports under the same property ids as this project's own pKa models, so the two appear
side by side and the UI's disagreement flag applies. That comparison does not flatter MolGpKa,
and the benchmark figures below say so: they were measured here, on the same held-out molecules
the MolCLR pKa models were scored on, because the paper is paywalled and its own figure could
not be read. Its median error is small (0.47 for acids) but its errors are heavy-tailed, so the
mean is much worse than the median.
"""

import threading
from importlib import resources

import torch
from rdkit import Chem

from .. import charge
from ..vendor.molgpka.featurize import Unfeaturizable, featurize, find_sites, prepare
from ..vendor.molgpka.model import GCNNet
from .base import ModelInfo, Prediction, PropertyInfo

_SHARED_DESCRIPTION = (
    "Predicted for each ionisable site separately and reduced to one number here — the lowest "
    "acidic site, the highest basic one. The per-site values drive the charge-vs-pH curve. "
    "Measured against this project's own pKa models it is the weaker of the two; it is here "
    "because it is the only open model that resolves individual sites."
)

_PROPERTIES = [
    PropertyInfo(
        id="pka_acidic",
        name="Most Acidic pKa",
        category="Physicochemical",
        units="pKa",
        task_type="regression",
        description=_SHARED_DESCRIPTION,
        # Measured here on the IUPAC scaffold-split test molecules, not taken from the paper.
        benchmark={"R^2": 0.352, "MAE": 1.489},
        dataset_url="https://github.com/Xundrug/MolGpKa",
    ),
    PropertyInfo(
        id="pka_basic",
        name="Most Basic pKa",
        category="Physicochemical",
        units="pKa",
        task_type="regression",
        description=_SHARED_DESCRIPTION,
        benchmark={"R^2": 0.384, "MAE": 2.096},
        dataset_url="https://github.com/Xundrug/MolGpKa",
    ),
]


class MolGpKa:
    """Per-site pKa. Also the source of the sites behind `/api/charge`."""

    info = ModelInfo(
        id="molgpka",
        name="MolGpKa",
        description=(
            "Graph-convolutional network predicting the pKa of each ionisable centre, matched "
            "by a set of 143 SMARTS patterns. Ported here to plain PyTorch so the weights run "
            "without torch-geometric."
        ),
        citation=(
            "Pan, X.; Wang, H.; Li, C.; Zhang, J. Z. H.; Ji, C. MolGpKa: A Web Server for Small "
            "Molecule pKa Prediction Using a Graph-Convolutional Neural Network. J. Chem. Inf. "
            "Model. 2021, 61, 3159–3165."
        ),
        paper_url="https://doi.org/10.1021/acs.jcim.1c00075",
        code_url="https://github.com/Xundrug/MolGpKa",
        # The repository's LICENSE.md is the MIT text but its copyright line still reads
        # "OpenNMT", apparently copied unedited; the licence is MIT either way.
        license="MIT",
    )
    properties = _PROPERTIES

    def __init__(self) -> None:
        self._models: dict[str, GCNNet] = {}
        files = resources.files("molpredict_backend.vendor.molgpka")
        for kind in ("acid", "base"):
            with resources.as_file(files / f"weight_{kind}.pth") as path:
                state = torch.load(path, map_location="cpu", weights_only=True)
            model = GCNNet()
            # strict, so a checkpoint that no longer matches this port fails loudly instead of
            # silently leaving layers at their random initialisation.
            model.load_state_dict(state, strict=True)
            model.eval()
            self._models[kind] = model
        self._lock = threading.Lock()

    def sites(self, smiles: str) -> tuple[list[charge.Site], float]:
        """Every ionisable site with its predicted pKa, plus the neutral form's formal charge.

        The formal charge is read after neutralisation, so it is zero for an ordinary molecule
        and non-zero only where the charge cannot be removed — a quaternary ammonium stays +1
        at every pH, and the curve should show that rather than starting from zero.
        """
        try:
            mol = prepare(smiles)
        except Unfeaturizable:
            return [], 0.0
        formal = float(Chem.GetFormalCharge(mol))

        found = []
        with self._lock, torch.no_grad():
            for site in find_sites(mol):
                x, edge_index = featurize(mol, site.atom_index)
                value = self._models[site.kind](
                    torch.from_numpy(x), torch.from_numpy(edge_index)
                ).item()
                found.append(charge.Site(site.atom_index, site.kind, float(value)))
        return found, formal

    def predict(self, smiles: str) -> list[Prediction]:
        sites, _ = self.sites(smiles)
        acids = [s.pka for s in sites if s.kind == "acid"]
        bases = [s.pka for s in sites if s.kind == "base"]
        out = []
        if acids:
            out.append(Prediction("pka_acidic", min(acids)))
        if bases:
            out.append(Prediction("pka_basic", max(bases)))
        return out
