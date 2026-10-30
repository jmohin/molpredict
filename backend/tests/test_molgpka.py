"""The MolGpKa port: that the weights still fit it, and that its arithmetic is right.

The reference implementation needs torch-geometric and torch-scatter, which this project does
not install, so these tests pin the port two ways: against an independent dense-matrix
implementation of the same convolution, and against experimental pKa values, which no amount of
self-consistency would satisfy if the featurisation were wrong.
"""

from importlib import resources

import pytest
import torch

from molpredict_backend.vendor.molgpka.featurize import (
    Unfeaturizable,
    featurize,
    find_sites,
    prepare,
)
from molpredict_backend.vendor.molgpka.model import GCNConv, GCNNet

# MolGpKa's GCNNet has 43 tensors across five convolutions, their norms, attention and 3 layers.
EXPECTED_TENSORS = 43


@pytest.fixture(scope="module")
def models() -> dict[str, GCNNet]:
    files = resources.files("molpredict_backend.vendor.molgpka")
    out = {}
    for kind in ("acid", "base"):
        with resources.as_file(files / f"weight_{kind}.pth") as path:
            state = torch.load(path, map_location="cpu", weights_only=True)
        assert len(state) == EXPECTED_TENSORS
        model = GCNNet()
        # strict: every published key must land on a parameter of this port, and none may be
        # left at its random initialisation. This is the test the port exists to pass.
        model.load_state_dict(state, strict=True)
        model.eval()
        out[kind] = model
    return out


def predict(models, smiles: str, kind: str) -> float | None:
    mol = prepare(smiles)
    values = []
    for site in find_sites(mol):
        if site.kind != kind:
            continue
        x, edge_index = featurize(mol, site.atom_index)
        with torch.no_grad():
            values.append(models[kind](torch.from_numpy(x), torch.from_numpy(edge_index)).item())
    if not values:
        return None
    return min(values) if kind == "acid" else max(values)


# Pinned from this port. They are not round numbers, so an accidental change to the
# featurisation or the message passing moves them.
GOLDEN = {
    ("CC(=O)O", "acid"): 4.434,
    ("c1ccccc1O", "acid"): 10.139,
    ("c1cnc[nH]1", "base"): 6.760,
}


@pytest.mark.parametrize(("smiles", "kind"), list(GOLDEN))
def test_golden_values(models, smiles, kind):
    assert predict(models, smiles, kind) == pytest.approx(GOLDEN[(smiles, kind)], abs=5e-3)


# Experimental values. The tolerance is wide because the model's own error is wide; the point
# is that a broken featurisation would miss by far more than this.
EXPERIMENTAL = [
    ("CC(=O)O", "acid", 4.76, "acetic acid"),
    ("c1ccccc1O", "acid", 9.99, "phenol"),
    ("OC(=O)c1ccccc1O", "acid", 2.98, "salicylic acid"),
    ("CC(=O)Oc1ccccc1C(=O)O", "acid", 3.49, "aspirin"),
    ("c1cnc[nH]1", "base", 6.95, "imidazole"),
    ("NCCO", "base", 9.50, "ethanolamine"),
]


@pytest.mark.parametrize(("smiles", "kind", "truth", "name"), EXPERIMENTAL)
def test_tracks_experiment(models, smiles, kind, truth, name):
    assert predict(models, smiles, kind) == pytest.approx(truth, abs=1.5)


def test_convolution_matches_a_dense_implementation():
    """Cross-check the message passing against the matrix form of the same equation.

    D^-1/2 (A + I) D^-1/2 X W + b, written out with a dense adjacency. The forward pass builds
    this with index_add_ over edges, so a slip in the direction of aggregation or in which end
    of each edge is normalised would show up here.
    """
    torch.manual_seed(0)
    conv = GCNConv(5, 4)
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.normal_(conv.bias)

    n = 6
    x = torch.randn(n, 5)
    # A path graph, stored in both directions as the featuriser stores bonds.
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    edge_index = torch.tensor(
        [[a for a, _ in pairs] + [b for _, b in pairs], [b for _, b in pairs] + [a for a, _ in pairs]]
    )

    adjacency = torch.eye(n)
    for a, b in pairs:
        adjacency[a, b] = adjacency[b, a] = 1.0
    degree = adjacency.sum(dim=1)
    normalised = adjacency / degree.sqrt().unsqueeze(1) / degree.sqrt().unsqueeze(0)
    expected = normalised @ (x @ conv.weight) + conv.bias

    assert torch.allclose(conv(x, edge_index), expected, atol=1e-5)


def test_featurisation_shape_and_site_marking():
    mol = prepare("CC(=O)O")
    site = find_sites(mol)[0]
    x, edge_index = featurize(mol, site.atom_index)
    assert x.shape == (mol.GetNumAtoms(), 29)
    assert edge_index.shape[0] == 2
    # Last column marks the site, and exactly one atom is the site.
    assert x[:, -1].sum() == 1.0
    assert x[site.atom_index, -1] == 1.0
    # Second to last is the distance to the site, which is zero at the site itself.
    assert x[site.atom_index, -2] == 0.0


def test_hydrogens_are_explicit():
    """The weights were trained with hydrogens in the graph."""
    assert prepare("CC(=O)O").GetNumAtoms() == 8


def test_bonds_are_stored_in_both_directions():
    mol = prepare("CCO")
    _, edge_index = featurize(mol, 0)
    assert edge_index.shape[1] == 2 * mol.GetNumBonds()


def test_glycine_has_an_acid_and_a_base_site():
    kinds = {s.kind for s in find_sites(prepare("NCC(=O)O"))}
    assert kinds == {"acid", "base"}


def test_hexane_has_no_ionisable_site():
    assert find_sites(prepare("CCCCCC")) == []


def test_charged_input_is_neutralised_first():
    """A drawn carboxylate must be put back to its neutral form, which the weights expect."""
    assert prepare("CC(=O)[O-]").GetNumAtoms() == prepare("CC(=O)O").GetNumAtoms()


def test_rejects_an_unparseable_structure():
    with pytest.raises(Unfeaturizable):
        prepare("not a molecule")


def test_deterministic(models):
    assert predict(models, "CC(=O)O", "acid") == predict(models, "CC(=O)O", "acid")
