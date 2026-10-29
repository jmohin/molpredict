"""SA_Score: Ertl & Schuffenhauer's synthetic accessibility estimate.

Not a trained model — it combines fragment frequencies in PubChem with penalties for size,
stereocentres, rings and macrocycles. It is here as a published second opinion on how hard a
molecule is to make, next to SCScore, which learned that notion from reactions instead.

RDKit ships it under Contrib rather than in the importable package, so it has to be reached by
path. That is the upstream-documented way to use it.
"""

import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))

import sascorer  # noqa: E402  (only importable once the path above is set)

from .base import ModelInfo, Prediction, PropertyInfo

ACCESSIBILITY = PropertyInfo(
    id="synthetic_accessibility_sa",
    name="Synthetic Accessibility (SA)",
    category="Synthesis",
    units="1 (easy) – 10 (hard)",
    task_type="descriptor",
    description=(
        "Fragment frequencies in PubChem, penalised for size, stereocentres and unusual rings. "
        "Rule-based, not learned."
    ),
    dataset_url="https://doi.org/10.1186/1758-2946-1-8",
    minimum=1.0,
    maximum=10.0,
)


class SAScore:
    info = ModelInfo(
        id="sa_score",
        name="SA_Score",
        description="Fragment-contribution heuristic for synthetic accessibility.",
        citation=(
            "Ertl, P.; Schuffenhauer, A. Estimation of synthetic accessibility score of "
            "drug-like molecules based on molecular complexity and fragment contributions. "
            "J. Cheminform. 2009, 1, 8."
        ),
        paper_url="https://doi.org/10.1186/1758-2946-1-8",
        code_url="https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score",
        license="BSD-3-Clause",
    )
    properties = [ACCESSIBILITY]

    def predict(self, smiles: str) -> list[Prediction]:
        value = sascorer.calculateScore(Chem.MolFromSmiles(smiles))
        return [Prediction(ACCESSIBILITY.id, value)]
