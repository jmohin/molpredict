"""SMILES to the graph MolCLR was pretrained on.

This must match MolCLR's `dataset/dataset_test.py` exactly. The pretrained embeddings are
indexed by these very numbers, so any drift here silently turns the pretrained weights into
noise — the model would still train and still report a score.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

# Atom index is (atomic number - 1): MolCLR's ATOM_LIST is list(range(1, 119)).
_NUM_ATOM_TYPE = 119

_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
# MolCLR lists four chirality tags but sizes the embedding at three, so a CHI_OTHER atom would
# index out of bounds upstream. Clamping keeps the rare case working and matches every tag the
# pretraining data could actually have reached.
_MAX_CHIRALITY_INDEX = 2

_BOND = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
_BOND_DIR = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

# Bond type 4 marks the self-loop edge the convolution adds; the embedding has 5 rows.
SELF_LOOP_BOND_TYPE = 4


class Unfeaturizable(ValueError):
    """The molecule cannot be turned into the graph MolCLR expects."""


def featurize(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (atom_features[N,2], edge_index[2,E], edge_features[E,2]).

    Hydrogens are made explicit, as they were during pretraining: MolCLR calls `Chem.AddHs`,
    so a graph built without them would not look like anything the model has seen.
    """
    mol = Chem.AddHs(mol)

    atoms = []
    for atom in mol.GetAtoms():
        number = atom.GetAtomicNum()
        if not 1 <= number < _NUM_ATOM_TYPE:
            raise Unfeaturizable(f"Atomic number {number} is outside MolCLR's embedding.")
        chirality = min(_CHIRALITY.index(atom.GetChiralTag()), _MAX_CHIRALITY_INDEX)
        atoms.append((number - 1, chirality))

    sources, targets, bonds = [], [], []
    for bond in mol.GetBonds():
        kind = bond.GetBondType()
        if kind not in _BOND:
            raise Unfeaturizable(f"Bond type {kind} is outside MolCLR's embedding.")
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Every bond becomes two directed edges, as in the reference implementation.
        sources += [start, end]
        targets += [end, start]
        feature = (_BOND.index(kind), _BOND_DIR.index(bond.GetBondDir()))
        bonds += [feature, feature]

    atom_features = np.array(atoms, dtype=np.int64).reshape(-1, 2)
    edge_index = np.array([sources, targets], dtype=np.int64).reshape(2, -1)
    edge_features = np.array(bonds, dtype=np.int64).reshape(-1, 2)
    return atom_features, edge_index, edge_features


def featurize_smiles(smiles: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Unfeaturizable(f"RDKit could not parse {smiles!r}.")
    return featurize(mol)
