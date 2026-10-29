"""The MolCLR port's contract with the published checkpoint.

`model.py` rewrites MolCLR's message passing in plain PyTorch so torch-geometric stays out of
the serving image. That rewrite was checked against the reference torch-geometric
implementation with the pretrained weights loaded: featurization matched atom for atom and the
encoder output matched to 0.0 on every molecule tried, chirality, bond directions and a
bondless ion included.

CI cannot repeat that comparison without installing torch-geometric, so the encoder outputs
from that verified run are pinned here instead. If a change to the port moves them, it has
drifted from the implementation the pretrained weights belong to.
"""

from importlib import resources

import pytest
import torch
from rdkit import Chem

from molpredict_backend.vendor.molclr.featurize import Unfeaturizable, featurize_smiles
from molpredict_backend.vendor.molclr.model import GINet

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# The encoder has 59 tensors; the checkpoint's remaining 4 belong to the pretraining head,
# which fine-tuning replaces.
EXPECTED_PRETRAINED_TENSORS = 59

# Measured with the port verified against torch-geometric. Not arbitrary: change only with a
# fresh comparison against the reference implementation.
GOLDEN_SUMS = {
    "CCO": 82.094627,
    ASPIRIN: 86.797333,
}


@pytest.fixture(scope="module")
def encoder():
    model = GINet(task="regression", drop_ratio=0.0)
    weights = resources.files("molpredict_backend.vendor.molclr") / "pretrained_gin.pth"
    with resources.as_file(weights) as path:
        adopted = model.load_pretrained(torch.load(path, map_location="cpu", weights_only=True))
    assert adopted == EXPECTED_PRETRAINED_TENSORS
    model.eval()
    return model


def embed(model, smiles):
    x, edge_index, edge_attr = featurize_smiles(smiles)
    with torch.no_grad():
        h, _ = model(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_attr),
            torch.zeros(len(x), dtype=torch.long),
        )
    return h


@pytest.mark.parametrize("smiles,expected", GOLDEN_SUMS.items())
def test_encoder_matches_the_verified_reference(encoder, smiles, expected):
    assert embed(encoder, smiles).sum().item() == pytest.approx(expected, abs=1e-4)


def test_featurization_adds_explicit_hydrogens(encoder):
    """MolCLR pretrained on graphs with hydrogens; dropping them changes every molecule."""
    x, _, _ = featurize_smiles("CCO")
    assert len(x) == Chem.AddHs(Chem.MolFromSmiles("CCO")).GetNumAtoms() == 9


def test_atom_index_is_atomic_number_minus_one():
    """The pretrained embedding is indexed this way; an off-by-one silently reads a different
    element's vector."""
    x, _, _ = featurize_smiles("CCO")
    carbon, oxygen, hydrogen = 6 - 1, 8 - 1, 1 - 1
    assert set(x[:, 0].tolist()) == {carbon, oxygen, hydrogen}


def test_every_bond_becomes_two_directed_edges():
    x, edge_index, edge_attr = featurize_smiles("CCO")
    bonds = Chem.AddHs(Chem.MolFromSmiles("CCO")).GetNumBonds()
    assert edge_index.shape == (2, 2 * bonds)
    assert len(edge_attr) == 2 * bonds


def test_a_molecule_without_bonds_still_embeds(encoder):
    """Self-loops carry a lone atom through; an empty edge tensor must not break the shapes."""
    assert torch.isfinite(embed(encoder, "[Na+]")).all()


def test_embedding_is_deterministic(encoder):
    assert torch.equal(embed(encoder, ASPIRIN), embed(encoder, ASPIRIN))


def test_unrepresentable_input_is_rejected():
    with pytest.raises(Unfeaturizable):
        featurize_smiles("not a molecule")
