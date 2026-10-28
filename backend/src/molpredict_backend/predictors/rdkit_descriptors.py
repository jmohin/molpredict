"""Classic RDKit physicochemical descriptors: instant, and a sanity anchor for the AI models."""

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

from .base import ModelInfo, Prediction, PropertyInfo

_DESCRIPTORS = [
    ("molecular_weight", "Molecular weight", "Da", Descriptors.MolWt),
    ("logP", "cLogP (Crippen)", "log-ratio", Crippen.MolLogP),
    ("tpsa", "Topological polar surface area", "Å²", rdMolDescriptors.CalcTPSA),
    ("hydrogen_bond_donors", "H-bond donors", "count", Lipinski.NumHDonors),
    ("hydrogen_bond_acceptors", "H-bond acceptors", "count", Lipinski.NumHAcceptors),
    ("rotatable_bonds", "Rotatable bonds", "count", rdMolDescriptors.CalcNumRotatableBonds),
    ("heavy_atoms", "Heavy atoms", "count", lambda m: m.GetNumHeavyAtoms()),
    ("fraction_csp3", "Fraction sp³ carbons", "fraction", rdMolDescriptors.CalcFractionCSP3),
    ("QED", "QED drug-likeness", "0–1", QED.qed),
    (
        "lipinski_violations",
        "Lipinski rule-of-5 violations",
        "count",
        lambda m: sum(
            [
                Descriptors.MolWt(m) > 500,
                Crippen.MolLogP(m) > 5,
                Lipinski.NumHDonors(m) > 5,
                Lipinski.NumHAcceptors(m) > 10,
            ]
        ),
    ),
]


class RDKitDescriptors:
    info = ModelInfo(
        id="rdkit",
        name="RDKit descriptors",
        description="Rule-based physicochemical descriptors. Not machine learning; shown as a reference.",
        citation="RDKit: Open-source cheminformatics. https://www.rdkit.org",
        paper_url="https://www.rdkit.org",
        code_url="https://github.com/rdkit/rdkit",
        license="BSD-3-Clause",
    )
    properties = [
        PropertyInfo(id=pid, name=name, category="Physicochemical", units=units, task_type="descriptor")
        for pid, name, units, _ in _DESCRIPTORS
    ]

    def predict(self, smiles: str) -> list[Prediction]:
        mol = Chem.MolFromSmiles(smiles)
        return [Prediction(pid, float(fn(mol))) for pid, _, _, fn in _DESCRIPTORS]
