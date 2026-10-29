"""HTTP API: list models, predict properties for one structure."""

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import chem
from .predictors import Predictor, load_predictors

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
