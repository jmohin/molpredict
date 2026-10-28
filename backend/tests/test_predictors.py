"""Contract every predictor must honour, so a new model can't quietly break the UI."""

import math

import pytest

from molpredict_backend.predictors.base import Prediction

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
HEXADECANE = "CCCCCCCCCCCCCCCC"


@pytest.fixture(scope="session")
def by_id(predictors):
    return {p.info.id: p for p in predictors}


def test_every_predictor_is_registered_once(predictors):
    ids = [p.info.id for p in predictors]
    assert len(ids) == len(set(ids))


def test_every_model_declares_citation_and_license(predictors):
    for p in predictors:
        assert p.info.citation and p.info.license and p.info.code_url, p.info.id


def test_declared_properties_have_unique_ids(predictors):
    for p in predictors:
        ids = [prop.id for prop in p.properties]
        assert len(ids) == len(set(ids)), p.info.id


def test_predictions_only_reference_declared_properties(predictors):
    for p in predictors:
        declared = {prop.id for prop in p.properties}
        returned = {pred.property_id for pred in p.predict(ASPIRIN)}
        assert returned <= declared, p.info.id


def test_predictions_are_finite(predictors):
    for p in predictors:
        for pred in p.predict(ASPIRIN):
            assert math.isfinite(pred.value), f"{p.info.id}/{pred.property_id}"


def test_every_predictor_returns_something(predictors):
    for p in predictors:
        assert p.predict(ASPIRIN), p.info.id


def test_predictions_are_deterministic(predictors):
    """The UI caches by SMILES, so a model that wanders would show stale numbers."""
    for p in predictors:
        first = {x.property_id: x.value for x in p.predict(ASPIRIN)}
        second = {x.property_id: x.value for x in p.predict(ASPIRIN)}
        assert first == second, p.info.id


def test_shared_property_ids_agree_on_units_and_task(predictors):
    """Two models sharing a property id end up on one row, so their metadata must match."""
    seen: dict[str, tuple[str, str]] = {}
    for p in predictors:
        for prop in p.properties:
            signature = (prop.units, prop.task_type)
            assert seen.setdefault(prop.id, signature) == signature, prop.id


def test_solubility_is_predicted_by_two_models(predictors):
    models = [p.info.id for p in predictors for prop in p.properties if prop.id == "Solubility_AqSolDB"]
    assert set(models) == {"admet_ai", "soltrannet"}


def test_classification_values_are_probabilities(by_id):
    admet = by_id["admet_ai"]
    classification = {p.id for p in admet.properties if p.task_type == "classification"}
    for pred in admet.predict(ASPIRIN):
        if pred.property_id in classification:
            assert 0.0 <= pred.value <= 1.0, pred.property_id


def _solubility(predictor, smiles: str) -> Prediction:
    return next(p for p in predictor.predict(smiles) if p.property_id == "Solubility_AqSolDB")


@pytest.mark.parametrize("model_id", ["admet_ai", "soltrannet"])
def test_solubility_ranks_ethanol_above_hexadecane(by_id, model_id):
    """A sanity check on direction: ethanol is miscible, hexadecane is not."""
    model = by_id[model_id]
    assert _solubility(model, ETHANOL).value > _solubility(model, HEXADECANE).value


def test_solubility_models_agree_on_a_common_drug(by_id):
    gap = abs(_solubility(by_id["admet_ai"], ASPIRIN).value - _solubility(by_id["soltrannet"], ASPIRIN).value)
    assert gap < 1.0


def test_soltrannet_warns_about_unfamiliar_elements(by_id):
    stn = by_id["soltrannet"]
    assert _solubility(stn, "CC[Sn](CC)(CC)CC").warning is not None
    assert _solubility(stn, ASPIRIN).warning is None


def test_rdkit_descriptors_match_known_values(by_id):
    values = {p.property_id: p.value for p in by_id["rdkit"].predict(ASPIRIN)}
    assert values["molecular_weight"] == pytest.approx(180.16, abs=0.01)
    assert values["hydrogen_bond_donors"] == 1
    assert values["heavy_atoms"] == 13
    assert values["lipinski_violations"] == 0
