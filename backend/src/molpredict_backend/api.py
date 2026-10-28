"""HTTP API: list models, predict properties for one structure."""

from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
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
