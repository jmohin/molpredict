# molpredict

Draw a molecule, see what recently published AI models predict about it.

- **Frontend** (`frontend/`): React + Vite with the [Ketcher](https://github.com/epam/ketcher)
  structure editor running fully in the browser (standalone/WASM mode).
- **Backend** (`backend/`): FastAPI + RDKit. Structures are parsed (SMILES or molfile),
  stripped to the largest fragment and neutralized, then sent to every registered model.

## Models

| Model | What it predicts | License |
|---|---|---|
| RDKit descriptors | MW, cLogP, TPSA, HBD/HBA, QED, Lipinski… (rule-based reference, not ML) | BSD-3 |
| [ADMET-AI](https://github.com/swansonk14/admet_ai) (Swanson et al., *Bioinformatics* 2024) | 41 TDC ADMET endpoints, with DrugBank-approved percentiles and per-endpoint benchmark scores | MIT |

Add a model by implementing the `Predictor` protocol in
`backend/src/molpredict_backend/predictors/base.py` and listing it in `predictors/__init__.py`.
Use the same `property_id` as an existing model when predicting the same property; the UI then
shows the two models side by side.

## Run locally

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
cd backend && uv run uvicorn molpredict_backend.api:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend.

## API

- `GET /api/models`: models, citations, and the properties each predicts.
- `POST /api/predict` `{"structure": "<SMILES or molfile>"}`: canonical SMILES plus each model's predictions.

Predictions are for research use only.
