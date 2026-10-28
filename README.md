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
| [SolTranNet](https://github.com/gnina/SolTranNet) (Francoeur & Koes, *JCIM* 2021) | Aqueous solubility (logS), as a second opinion on ADMET-AI's AqSolDB endpoint | Apache-2.0 |

SolTranNet is vendored **unmodified** under `backend/src/molpredict_backend/vendor/soltrannet/`
(model code, 60 kB of weights, and its Apache-2.0 license) because the published package imports
`pkg_resources`, which Python 3.12 removed.

A model only belongs here if it takes a SMILES string and answers fast enough to feel live as you
draw. Models needing 3D coordinates, a generated conformer, or a training step of your own do not
fit this UI.

Add a model by implementing the `Predictor` protocol in
`backend/src/molpredict_backend/predictors/base.py` and listing it in `predictors/__init__.py`.
Use the same `property_id` as an existing model when predicting the same property; the UI then
shows the two models side by side and flags it when they disagree.

## Reading the results

- **vs. approved drugs** places the value among DrugBank approved drugs, as scored by that model.
- **models disagree by N** appears when two models differ by more than 0.2 (probabilities) or
  1.0 (regression units, mostly log scale).
- **out of range** means the model extrapolated past a physically possible bound — a negative
  half-life, say. The raw value is still shown.
- **limited training data** is the model's own caveat about this molecule.
- The benchmark score next to each model is that model's reported held-out performance for that
  property. Some are poor: ADMET-AI reports R² of −2.39 for half-life.

## Run locally

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
cd backend && uv run uvicorn molpredict_backend.api:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend.

## Run the container

One image serves both halves: the built frontend as static files, the API under `/api`. Same
origin, so there is no CORS or proxy config to get wrong.

```bash
docker compose up --build
```

Open http://localhost:8080. The build takes a few minutes the first time (the CPU torch wheel
alone is 200 MB) and produces a ~2.4 GB image; afterwards uv's cache is reused via a build cache
mount, so only the layers you changed rebuild.

The container reads `MOLPREDICT_STATIC`, the directory of built frontend files to serve at `/`.
It is unset in development, where Vite serves the frontend instead, and the mount is declared
after the API routes so it cannot shadow them.

Startup is dominated by loading the ADMET-AI ensembles — about 8 seconds before the first
request is answered, then ~0.1 s per prediction. Anywhere this is deployed needs roughly 2 GB of
memory, and a host that stops idle instances pays that 8 seconds again on the first request
after a wake-up.

## Tests

```bash
cd backend && uv run pytest
```

```bash
cd frontend && npm test
```

The backend tests cover structure parsing (salts, charges, molfiles, junk input), a contract every
predictor must honour, and the HTTP API. The contract tests are what catch a new model
misbehaving: predicting an undeclared property, returning a non-finite value, disagreeing with
another model about a shared property's units, or drifting between identical calls.

The frontend tests cover the flagging rules in `src/flags.ts` — disagreement thresholds, physical
bounds, value formatting and benchmark labels.

CI (`.github/workflows/ci.yml`) runs both suites on every push and pull request, plus a
type-check and a production build of the frontend, and builds the container and smoke-tests it —
health, the served `index.html`, and a real prediction through the API. It needs a GitHub remote
to run.

On Linux — CI and any deployment container — torch resolves to the CPU-only build from
PyTorch's own index, which keeps several GB of CUDA packages out of the image. macOS keeps the
default PyPI wheel. A CI step asserts the pin still holds, since a transitive dependency could
quietly pull the CUDA build back in.

## API

- `GET /api/models`: models, citations, and the properties each predicts.
- `POST /api/predict` `{"structure": "<SMILES or molfile>"}`: canonical SMILES plus each model's predictions.

Predictions are for research use only.
