"""The HTTP contract the frontend depends on."""

import pytest
from fastapi.testclient import TestClient

from molpredict_backend.api import app

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_models(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["models"] >= 1


def test_models_endpoint_describes_every_property(client):
    models = client.get("/api/models").json()
    assert {m["id"] for m in models} == {"rdkit", "admet_ai", "soltrannet"}
    for m in models:
        assert m["properties"]
        for prop in m["properties"]:
            assert prop["id"] and prop["name"] and prop["category"]
            assert prop["task_type"] in {"regression", "classification", "descriptor"}


def test_predict_returns_canonical_smiles_and_results(client):
    body = client.post("/api/predict", json={"structure": ASPIRIN}).json()
    assert body["smiles"] == ASPIRIN
    assert {r["model_id"] for r in body["results"]} == {"rdkit", "admet_ai", "soltrannet"}


def test_predict_canonicalizes_input(client):
    body = client.post("/api/predict", json={"structure": "OC(=O)c1ccccc1OC(C)=O"}).json()
    assert body["smiles"] == ASPIRIN


def test_predict_strips_counterions(client):
    body = client.post("/api/predict", json={"structure": f"{ASPIRIN}.[Na+].[Cl-]"}).json()
    assert body["smiles"] == ASPIRIN


@pytest.mark.parametrize("bad", ["", "C1CC", "nonsense"])
def test_predict_rejects_bad_structures(client, bad):
    response = client.post("/api/predict", json={"structure": bad})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_prediction_fields_match_the_frontend_contract(client):
    body = client.post("/api/predict", json={"structure": ASPIRIN}).json()
    for result in body["results"]:
        for pred in result["predictions"]:
            assert set(pred) == {"property_id", "value", "drugbank_percentile", "warning"}
            assert isinstance(pred["value"], float)


def test_every_prediction_maps_to_a_declared_property(client):
    """A prediction whose property the model never declared is dropped silently by the UI."""
    declared = {m["id"]: {p["id"] for p in m["properties"]} for m in client.get("/api/models").json()}
    body = client.post("/api/predict", json={"structure": ASPIRIN}).json()
    for result in body["results"]:
        for pred in result["predictions"]:
            assert pred["property_id"] in declared[result["model_id"]]


def test_repeated_requests_are_stable(client):
    first = client.post("/api/predict", json={"structure": ASPIRIN}).json()
    second = client.post("/api/predict", json={"structure": ASPIRIN}).json()
    assert first == second


def test_static_mount_serves_the_frontend_without_shadowing_the_api(tmp_path, monkeypatch):
    """In the container the built frontend is mounted at "/", which must not swallow /api."""
    (tmp_path / "index.html").write_text("<!doctype html><title>molpredict</title>")
    monkeypatch.setenv("MOLPREDICT_STATIC", str(tmp_path))

    import importlib

    import molpredict_backend.api as api

    module = importlib.reload(api)
    try:
        with TestClient(module.app) as c:
            assert c.get("/").status_code == 200
            assert "molpredict" in c.get("/").text
            assert c.get("/api/health").json()["ok"] is True
            assert c.get("/api/models").status_code == 200
    finally:
        # Leave the module as the other tests found it: no static mount.
        monkeypatch.delenv("MOLPREDICT_STATIC")
        importlib.reload(api)
