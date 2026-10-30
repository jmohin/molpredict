"""MolGpKa's atom featurisation and ionisable-site detection.

Both must match the reference exactly, since the weights were trained against them. Two
departures, neither of which changes a number:

  * The reference calls `ComputeGasteigerCharges` and `AssignStereochemistry` and then uses
    neither in its 29 features. They are dropped here.
  * It resolves `smarts_pattern.tsv` against `os.path.abspath("")`, the process's working
    directory, so importing it from anywhere but MolGpKa's own `src/` raises FileNotFoundError.
    This resolves the file against the module instead.

The featurised graph is one molecule paired with ONE candidate site: the last two features say
how far each atom is from that site and whether it is that site, so a molecule with five
ionisable centres is run through the network five times.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

SYMBOLS = ["C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
RING_SIZES = [3, 4, 5, 6, 7, 8]

_ACCEPTOR_SMARTS = [
    "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
]
_DONOR_SMARTS = [
    "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
    "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",
]


class Unfeaturizable(Exception):
    """The molecule cannot be represented for this model."""


@dataclass(frozen=True)
class Site:
    """One ionisable centre: an atom index and whether it acts as an acid or a base."""

    atom_index: int
    kind: str  # "acid" or "base"


@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[Chem.Mol, list[int], str], ...]:
    """Load the SMARTS table as (pattern, indices into the match, acid/base)."""
    text = resources.files(__package__).joinpath("smarts_pattern.tsv").read_text()
    out = []
    for row in csv.reader(text.splitlines()[1:], delimiter="\t"):
        if len(row) < 4:
            continue
        _, smarts, index, kind = (cell.strip() for cell in row[:4])
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        out.append((pattern, [int(i) for i in index.split(",")], kind))
    return tuple(out)


def _implicit_valence(atom: Chem.Atom) -> int:
    """Same number as the reference's `GetImplicitValence`, without its deprecation warning.

    After AddHs this is 0 for every atom, so it contributes nothing; it is kept because the
    weights were trained with the column present.
    """
    try:
        return atom.GetValence(Chem.ValenceType.IMPLICIT)
    except AttributeError:  # older RDKit
        return atom.GetImplicitValence()


@lru_cache(maxsize=1)
def _hydrogen_patterns() -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    return (
        [Chem.MolFromSmarts(s) for s in _DONOR_SMARTS],
        [Chem.MolFromSmarts(s) for s in _ACCEPTOR_SMARTS],
    )


def prepare(smiles: str) -> Chem.Mol:
    """Neutralise, re-canonicalise and add explicit hydrogens, as the reference does.

    The weights were trained on the neutral form with hydrogens present, so a drawn molecule
    that arrives already deprotonated has to be put back before it is asked about.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise Unfeaturizable(smiles)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol is None:
        raise Unfeaturizable(smiles)
    return Chem.AddHs(mol)


def find_sites(mol: Chem.Mol) -> list[Site]:
    """Every ionisable centre matched by MolGpKa's 143 SMARTS patterns."""
    acids: set[int] = set()
    bases: set[int] = set()
    for pattern, indices, kind in _patterns():
        for match in mol.GetSubstructMatches(pattern):
            for index in indices:
                if index >= len(match):
                    continue
                (acids if kind == "A" else bases).add(match[index])
    return [Site(i, "acid") for i in sorted(acids)] + [
        Site(i, "base") for i in sorted(bases)
    ]


def featurize(mol: Chem.Mol, atom_index: int) -> tuple[np.ndarray, np.ndarray]:
    """29 features per atom, relative to one candidate site, plus the bond graph."""
    donors, acceptors = _hydrogen_patterns()
    donor_atoms = {a for p in donors for match in mol.GetSubstructMatches(p) for a in match}
    acceptor_atoms = {a for p in acceptors for match in mol.GetSubstructMatches(p) for a in match}
    ring = mol.GetRingInfo()

    rows = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        row: list[float] = []
        row += [atom.GetSymbol() == s for s in SYMBOLS[:-1]]
        # one_hot falls back to the last entry for anything unlisted, so iodine's slot also
        # carries every element MolGpKa never saw.
        row += [atom.GetSymbol() not in SYMBOLS[:-1]]
        row += [atom.GetDegree()]
        row += [atom.GetHybridization() == h for h in HYBRIDIZATIONS[:-1]]
        row += [atom.GetHybridization() not in HYBRIDIZATIONS[:-1]]
        row += [_implicit_valence(atom)]
        row += [atom.GetIsAromatic()]
        row += [ring.IsAtomInRingOfSize(i, size) for size in RING_SIZES]
        row += [i in donor_atoms]
        row += [i in acceptor_atoms]
        row += [atom.GetFormalCharge()]
        # Distance to the site, counted as the number of atoms along the path including both
        # ends — so an adjacent atom scores 2, not 1 — and 0 for the site itself.
        row += [0 if i == atom_index else len(rdmolops.GetShortestPath(mol, i, atom_index))]
        row += [i == atom_index]
        rows.append([float(v) for v in row])

    sources, targets = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sources += [start, end]
        targets += [end, start]

    x = np.asarray(rows, dtype=np.float32)
    if x.shape[1] != 29:
        raise AssertionError(f"expected 29 features per atom, built {x.shape[1]}")
    edge_index = np.asarray([sources, targets], dtype=np.int64).reshape(2, -1)
    return x, edge_index
