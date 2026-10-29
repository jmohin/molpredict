"""SCScore: synthetic complexity learned from a reaction corpus.

Trained on 12M Reaxys reactions on the premise that a reaction's product is usually more
complex than its reactants, so the score needs no hand-assigned complexity labels.

Only the published weights are vendored, under `molpredict_backend.vendor.scscore`. The
reference implementation is Python 2-era (it imports `six`, defaults to unpickling its weights,
and uses `np.bool`, removed in NumPy 1.24), so the forward pass below is written out instead —
it is five ReLU layers and a sigmoid. The weights were converted from the published
`.as_numpy.json.gz` to a compressed `.npz`; they are float32 in the checkpoint, so that
conversion is bit-exact, and it drops 6.2 MB to 2.4 MB.
"""

from importlib import resources

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .base import ModelInfo, Prediction, PropertyInfo

# The published model is the 1024-bit binary Morgan variant, radius 2, chirality included.
_FP_BITS = 1024
_FP_RADIUS = 2
# The network's output is squashed into 1..5: 1 + (scale - 1) * sigmoid(x).
_SCORE_SCALE = 5.0

COMPLEXITY = PropertyInfo(
    id="synthetic_complexity_scscore",
    name="Synthetic Complexity (SCScore)",
    category="Synthesis",
    units="1 (simple) – 5 (complex)",
    task_type="regression",
    description=(
        "Learned from 12M Reaxys reactions, assuming products are more complex than their "
        "reactants. Higher means more synthetic effort."
    ),
    dataset_url="https://doi.org/10.1021/acs.jcim.7b00622",
    minimum=1.0,
    maximum=5.0,
)


class SCScore:
    info = ModelInfo(
        id="scscore",
        name="SCScore",
        description="Neural net scoring how hard a molecule looks to synthesise.",
        citation=(
            "Coley, C. W.; Rogers, L.; Green, W. H.; Jensen, K. F. SCScore: Synthetic "
            "Complexity Learned from a Reaction Corpus. J. Chem. Inf. Model. 2018, 58, 252–261."
        ),
        paper_url="https://doi.org/10.1021/acs.jcim.7b00622",
        code_url="https://github.com/connorcoley/scscore",
        license="MIT",
    )
    properties = [COMPLEXITY]

    def __init__(self) -> None:
        weights = resources.files("molpredict_backend.vendor.scscore") / "scscore_1024bool.npz"
        with resources.as_file(weights) as path, np.load(path) as data:
            self._weights = [data[f"w{i}"] for i in range(len(data.files))]
        self._fingerprint = rdFingerprintGenerator.GetMorganGenerator(
            radius=_FP_RADIUS, fpSize=_FP_BITS, includeChirality=True
        )

    def predict(self, smiles: str) -> list[Prediction]:
        mol = Chem.MolFromSmiles(smiles)
        fp = np.asarray(self._fingerprint.GetFingerprintAsNumPy(mol), dtype=np.float32)

        x = fp
        # Every pair of arrays is a weight matrix and its bias; all but the last are ReLU.
        for i in range(0, len(self._weights) - 2, 2):
            x = np.maximum(0.0, x @ self._weights[i] + self._weights[i + 1])
        x = x @ self._weights[-2] + self._weights[-1]

        value = 1.0 + (_SCORE_SCALE - 1.0) / (1.0 + np.exp(-x.item()))
        return [Prediction(COMPLEXITY.id, float(value))]
