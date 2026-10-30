"""HTTP API: list models, predict properties for one structure."""

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import charge, chem
from .predictors import Predictor, load_predictors
from .predictors.molgpka import MolGpKa

predictors: list[Predictor] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictors.extend(load_predictors())
    yield
    predictors.clear()


app = FastAPI(title="molpredict", lifespan=lifespan)

# Ketcher inlines its WASM into the JS bundle, which makes it ~29 MB. Neither StaticFiles nor
# Fly's proxy compresses, so without this every first visit downloads all of it.
app.add_middleware(GZipMiddleware, minimum_size=1024)


class PredictRequest(BaseModel):
    # SMILES or molfile.
    structure: str


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models": len(predictors)}


@app.get("/api/models")
def models() -> list[dict]:
    return [
        {**asdict(p.info), "properties": [asdict(prop) for prop in p.properties]}
        for p in predictors
    ]


@lru_cache(maxsize=2048)
def _predict(smiles: str) -> tuple:
    return tuple(
        {"model_id": p.info.id, "predictions": [asdict(x) for x in p.predict(smiles)]}
        for p in predictors
    )


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    try:
        smiles = chem.canonical_smiles(req.structure)
    except chem.InvalidStructure as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"smiles": smiles, "results": list(_predict(smiles))}


@lru_cache(maxsize=1024)
def _charge(smiles: str) -> dict:
    """Net charge across pH 0-14, from MolGpKa's per-site pKa values.

    A separate route from /api/predict because this is a curve rather than a number, and the
    Predictor protocol every model implements returns scalars.
    """
    model = next((p for p in predictors if isinstance(p, MolGpKa)), None)
    if model is None:
        raise HTTPException(status_code=503, detail="The pKa site model is not loaded.")

    sites, formal = model.sites(smiles)
    usable = charge.relevant(sites)
    phs, charges = charge.curve(usable, formal)
    return {
        "smiles": smiles,
        "ph": [round(p, 2) for p in phs],
        "charge": [round(c, 4) for c in charges],
        "sites": [
            {"atom_index": s.atom_index, "kind": s.kind, "pka": round(s.pka, 2)}
            for s in sorted(usable, key=lambda s: s.pka)
        ],
        # Sites whose pKa falls far outside 0-14 are dropped from the curve; saying how many
        # keeps the plot honest about what it left out.
        "sites_outside_window": len(sites) - len(usable),
        "formal_charge": formal,
        "charge_at_physiological_ph": round(
            charge.net_charge(usable, charge.PHYSIOLOGICAL_PH, formal), 3
        ),
        "isoelectric_point": (
            round(point, 2)
            if (point := charge.isoelectric_point(usable, formal)) is not None
            else None
        ),
    }


@app.post("/api/charge")
def charge_curve(req: PredictRequest) -> dict:
    try:
        smiles = chem.canonical_smiles(req.structure)
    except chem.InvalidStructure as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _charge(smiles)


# In the container the built frontend is served from this same app, so the browser sees one
# origin. In development the variable is unset and Vite serves the frontend instead, proxying
# /api here. The mount is last because it claims "/": the routes above are matched first.
def mount_frontend(app: FastAPI, directory: str) -> None:
    """Serve the built frontend at "/".

    Mounting "/" claims every path, so this must run after the API routes are declared: routes
    are matched in the order they were added.
    """
    app.mount("/", StaticFiles(directory=directory, html=True), name="static")


if static_dir := os.environ.get("MOLPREDICT_STATIC"):
    mount_frontend(app, static_dir)
