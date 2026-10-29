"""The HTTP contract the frontend depends on."""

import pytest
from fastapi.testclient import TestClient

from molpredict_backend.api import app, mount_frontend

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
MODEL_IDS = {"rdkit", "admet_ai", "soltrannet", "scscore", "sa_score"}


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
    assert {m["id"] for m in models} == MODEL_IDS
    for m in models:
        assert m["properties"]
        for prop in m["properties"]:
            assert prop["id"] and prop["name"] and prop["category"]
            assert prop["task_type"] in {"regression", "classification", "descriptor"}


def test_predict_returns_canonical_smiles_and_results(client):
    body = client.post("/api/predict", json={"structure": ASPIRIN}).json()
    assert body["smiles"] == ASPIRIN
    assert {r["model_id"] for r in body["results"]} == MODEL_IDS


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


def test_static_mount_does_not_shadow_the_api(tmp_path, client):
    """In the container the built frontend is mounted at "/", which must not swallow /api."""
    (tmp_path / "index.html").write_text("<!doctype html><title>molpredict</title>")
    mount_frontend(app, str(tmp_path))
    try:
        assert client.get("/").status_code == 200
        assert "molpredict" in client.get("/").text
        assert client.get("/api/health").json()["ok"] is True
        assert len(client.get("/api/models").json()) == len(MODEL_IDS)
    finally:
        # The mount is the last route added; leave the app as the other tests found it.
        app.router.routes.pop()


def test_large_responses_are_compressed(client):
    """The frontend bundle is ~29 MB uncompressed; nothing in front of the app compresses it."""
    body = client.get("/api/models", headers={"accept-encoding": "gzip"})
    assert body.headers.get("content-encoding") == "gzip"
