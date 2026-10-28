"""Parsing and standardization of user-supplied structures."""

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


class InvalidStructure(ValueError):
    pass


def parse(structure: str) -> Chem.Mol:
    """Parse a SMILES string or a molfile (V2000/V3000) into an RDKit molecule."""
    structure = structure.strip()
    if not structure:
        raise InvalidStructure("Empty structure.")
    if "M  END" in structure:
        mol = Chem.MolFromMolBlock(structure)
    else:
        mol = Chem.MolFromSmiles(structure)
    if mol is None or mol.GetNumAtoms() == 0:
        raise InvalidStructure("Could not parse structure.")
    return mol


def standardize(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest fragment (drops salts/solvents) and neutralize charges."""
    mol = rdMolStandardize.FragmentParent(mol)
    return rdMolStandardize.Uncharger().uncharge(mol)


def canonical_smiles(structure: str) -> str:
    return Chem.MolToSmiles(standardize(parse(structure)))
