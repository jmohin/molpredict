"""Structure parsing is the one place user input reaches RDKit, so it carries the edge cases."""

import pytest
from rdkit import Chem

from molpredict_backend import chem

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_parses_smiles():
    assert chem.parse(ASPIRIN).GetNumAtoms() == 13


def test_parses_molblock():
    molblock = Chem.MolToMolBlock(Chem.MolFromSmiles(ASPIRIN))
    assert chem.canonical_smiles(molblock) == ASPIRIN


@pytest.mark.parametrize("bad", ["", "   ", "C1CC", "not a molecule", "C(C"])
def test_rejects_unparseable_input(bad):
    with pytest.raises(chem.InvalidStructure):
        chem.parse(bad)


def test_canonicalizes_equivalent_smiles():
    """Different spellings of one molecule must collapse, or the prediction cache misses."""
    assert chem.canonical_smiles("OC(=O)c1ccccc1OC(C)=O") == chem.canonical_smiles(ASPIRIN)


def test_strips_counterions():
    assert chem.canonical_smiles(f"{ASPIRIN}.[Na+].[Cl-]") == ASPIRIN


def test_keeps_largest_fragment_regardless_of_order():
    assert chem.canonical_smiles(f"[Na+].[Cl-].{ASPIRIN}") == ASPIRIN


def test_neutralizes_charges():
    """Models are trained on neutral structures; a drawn carboxylate should match its acid."""
    assert chem.canonical_smiles("CC(=O)Oc1ccccc1C(=O)[O-]") == ASPIRIN


def test_leading_and_trailing_whitespace_is_ignored():
    assert chem.canonical_smiles(f"  {ASPIRIN}\n") == ASPIRIN
