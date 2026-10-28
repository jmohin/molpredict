"""Parsing and standardization of user-supplied structures."""

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


class InvalidStructure(ValueError):
    pass


def parse(structure: str) -> Chem.Mol:
    """Parse a SMILES string or a molfile (V2000/V3000) into an RDKit molecule."""
    if not structure.strip():
        raise InvalidStructure("Empty structure.")
    if "M  END" in structure:
        # A molfile's first line is its title and is usually blank, so the leading newline is
        # structural: stripping it shifts every line up and the counts line is misread.
        mol = Chem.MolFromMolBlock(structure)
    else:
        mol = Chem.MolFromSmiles(structure.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise InvalidStructure("Could not parse structure.")
    return mol


def standardize(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest fragment (drops salts/solvents) and neutralize charges."""
    mol = rdMolStandardize.FragmentParent(mol)
    return rdMolStandardize.Uncharger().uncharge(mol)


def canonical_smiles(structure: str) -> str:
    return Chem.MolToSmiles(standardize(parse(structure)))
