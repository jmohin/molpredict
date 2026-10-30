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
| [SCScore](https://github.com/connorcoley/scscore) (Coley et al., *JCIM* 2018) | Synthetic complexity (1–5), learned from 12M Reaxys reactions | MIT |
| [SA_Score](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score) (Ertl & Schuffenhauer, *J. Cheminform.* 2009) | Synthetic accessibility (1–10), fragment-based heuristic | BSD-3 |
| [MolCLR](https://github.com/yuyangw/MolCLR) (Wang et al., *Nat. Mach. Intell.* 2022), fine-tuned here | Boiling point, melting point, vapor pressure, ready biodegradability, fathead minnow LC50, rat oral LD50 | MIT (encoder) |
| MolCLR fine-tuned on the [IUPAC pKa dataset](https://doi.org/10.5281/zenodo.7236453) (Zheng 2022) | Acidic and basic pKa | **CC BY-NC 4.0** |

SolTranNet is vendored **unmodified** under `backend/src/molpredict_backend/vendor/soltrannet/`
(model code, 60 kB of weights, and its Apache-2.0 license) because the published package imports
`pkg_resources`, which Python 3.12 removed.

SCScore's published weights are vendored too, under `vendor/scscore/`, with its MIT license. They
are converted from the released `.as_numpy.json.gz` to a compressed `.npz` — the checkpoint is
float32, so the conversion is bit-exact, and it takes 6.2 MB down to 2.4 MB. The forward pass
(five ReLU layers and a sigmoid) is rewritten rather than vendored, because the reference
implementation is Python 2-era: it imports `six`, defaults to unpickling its own weights, and
uses `np.bool`, removed in NumPy 1.24.

The two synthesis models disagree in an interesting way, and deliberately do not share a row.
SCScore learned complexity from reaction data, so it scores large drug-like molecules highly
(imatinib ≈ 5.0). SA_Score penalises stereocentres, fused rings and macrocycles, so it scores
natural products highly instead (erythromycin ≈ 6.1, where SCScore says 2.9). They answer
different questions on different scales, so comparing the numbers directly would be a mistake.

A model only belongs here if it takes a SMILES string and answers fast enough to feel live as you
draw. Models needing 3D coordinates, a generated conformer, or a training step of your own do not
fit this UI.

Add a model by implementing the `Predictor` protocol in
`backend/src/molpredict_backend/predictors/base.py` and listing it in `predictors/__init__.py`.
Use the same `property_id` as an existing model when predicting the same property; the UI then
shows the two models side by side and flags it when they disagree.

## Models trained here

Every other model above is someone else's, used as published. The MolCLR row is different: it is
MolCLR's self-supervised GIN encoder — pretrained on 10M unlabelled molecules — fine-tuned by
this project on EPA/NIEHS [OPERA](https://github.com/NIEHS/OPERA) experimental data, one model
per endpoint. `training/` holds the code.

```bash
cd training && uv run python prepare_data.py && uv run python prepare_ecotox.py && uv run python train.py
```

`prepare_data.py` pulls OPERA's archive and `prepare_ecotox.py` the EPA ECOTOX release (143 MB,
resumable — the EPA's host dropped the connection partway through every attempt made while
writing this, and a truncated archive still looks like a zip until something reads its central
directory). Both write one CSV per endpoint, standardising SMILES exactly as the API does, so
each model is trained on the form it will be asked about. `train.py`
fine-tunes and writes `runs/<endpoint>/model.pt`, which carries its own held-out metrics; those
are the numbers the UI shows beside each prediction, so a score always belongs to the model that
earned it.

Held-out scores on that split, which are the numbers the UI shows:

| Endpoint | Molecules | Score | Error |
|---|---|---|---|
| Boiling point | 7,801 | R² 0.770 | RMSE 53.7 °C |
| Melting point | 22,287 | R² 0.764 | RMSE 56.3 °C |
| Vapor pressure | 6,643 | R² 0.691 | RMSE 2.20 log units |
| Ready biodegradability | 1,600 | AUROC 0.899 | accuracy 0.800 |
| Fathead minnow LC50 | 661 | R² 0.618 | RMSE 1.02 log units |
| Rat oral LD50 | 11,168 | R² 0.505 | RMSE 0.60 log units |
| Acidic pKa | 3,154 | R² 0.687 | RMSE 1.98 |
| Basic pKa | 5,103 | R² 0.810 | RMSE 2.05 |

The first three scores are **lower than the ones they replaced** while the models behind them are
better, which is worth spelling out because the opposite reading is the natural one. OPERA ships
"2.9 update" tables that are almost entirely molecules its SDFs do not carry, and merging them
grew melting point from 8,641 molecules to 22,287, vapor pressure from 2,815 to 6,643 and
boiling point from 5,433 to 7,801. That changes the scaffold split, so the held-out set is new
chemistry and a harder exam. The column above is each model's score on its own test set, and
across a change of dataset those are not comparable.

Scored instead on molecules that are honest for both — the new test set minus everything the old
model had in training or validation — the extra data wins every endpoint:

| Endpoint | Common test molecules | Old R² | New R² | Old RMSE | New RMSE |
|---|---|---|---|---|---|
| Boiling point | 445 | 0.655 | **0.712** | 70.2 °C | **64.2 °C** |
| Melting point | 1,451 | 0.713 | **0.731** | 62.0 °C | **60.0 °C** |
| Vapor pressure | 459 | 0.561 | **0.627** | 2.64 | **2.43** |

Both models score well below their headline numbers there, which is the point of that subset: it
is the hardest slice available, excluding every scaffold family either model was trained on.

The other four endpoints kept their earlier weights. They gained no data — OPERA has no more,
and CATMoS's remaining 39,000 structures are its own predictions rather than measurements — so
the only lever left was the training recipe, and a cosine learning-rate decay made three of the
four slightly worse (acidic pKa R² 0.687 → 0.656, basic pKa 0.810 → 0.792, biodegradability
AUROC 0.899 → 0.895, LD50 0.5045 → 0.509). With identical data, splits and seeds that is a clean
negative result, so it was not shipped for them.

### Aquatic toxicity, and one model that was not shipped

**Fathead minnow LC50** comes from EPA's [ECOTOX Knowledgebase](https://cfpub.epa.gov/ecotox/),
whose ASCII release is 143 MB of pipe-delimited tables abstracted from thousands of separate
published studies. It carries no structures at all — only CAS numbers and DSSTox ids — so SMILES
come from OPERA's archive, three of whose files carry `CASRN` beside QSAR-ready structures. That
keeps one structure convention across every endpoint here.

Nearly all the work is in the filters, because ECOTOX holds several kinds of row that look
numeric and are not comparable:

- **Censored values.** A result of `>100 mg/L` means the LC50 was never reached. The operator
  lives in its own column, `conc1_mean_op`, while `conc1_mean` holds a bare `100` either way —
  so a filter that inspects the value cannot see the censoring and keeps every one of these.
  They cluster at round numbers, so the damage would be systematic rather than noise. 3.7% of
  all results are censored; 920 were dropped here.
- **Mixed durations.** A 24-hour and a 96-hour LC50 are different quantities. Fathead minnow
  results run 3,337 rows at the standard 96 h against 932 at 24 h and 849 at 48 h.
- **Formulated products.** `conc1_type = F` is the concentration of a commercial formulation,
  not of the molecule the SMILES describes.
- **Mixed units**, from ng/L to µM, plus units of a dosed carrier.

A measured 661-chemical dataset survives that. MolCLR earns its place on it, which is worth
checking rather than assuming at this size:

| Fathead minnow LC50, same scaffold split (test n=66) | MAE | R² |
|---|---|---|
| Predict the training mean | 1.367 | −0.046 |
| Morgan fingerprint + ridge | 1.070 | 0.256 |
| **MolCLR fine-tuned** | **0.797** | **0.618** |

**Daphnia magna LC50 was built and deliberately not shipped.** The same pipeline yields it
(`prepare_ecotox.py` writes both CSVs, and `train.py` trains both), but only 256 chemicals
survive filtering — a third of its rows are formulated products — which leaves a 25-molecule
test set and R² 0.123. That is better than a fingerprint ridge (−0.517) and than predicting the
mean (−0.102), and still not a model worth putting a number beside: 25 molecules cannot measure
one. The code is kept so it can be retrained if the dataset grows.

Unlike the pKa data, ECOTOX carries no non-commercial restriction: EPA's standard position is
that its data is in the public domain under 17 U.S.C. § 105.

Two endpoints deserve a caveat the number alone does not give. **Melting point** carries the
largest absolute error: it depends on how molecules pack in a crystal, which a 2D graph cannot
see, so symmetric molecules are badly under-predicted — naphthalene comes out near 5 °C against
a real 80 °C.
And the **pKa** models answer for any structure, so a molecule with no acidic proton still gets
an acidic pKa; the number is meaningless there rather than merely large. Both caveats are shown
in the UI next to the property.

The pKa models are worth comparing against the alternatives this project rejected. pKaPredict
managed MAE 2.44 with R² 0.289 — barely better than guessing a constant. These reach MAE 1.27
and 1.32 with R² 0.69 and 0.81, so they are genuinely predictive. They are still roughly twice
the error of specialist published pKa models, which report RMSE under 1.

Scores come from a **Bemis-Murcko scaffold split**: molecules sharing a scaffold stay on the same
side, so the test set measures generalisation to new chemistry rather than to near-copies. A
random split would read much better and mean much less.

Three things had to be fixed before those scores meant anything, and all three were found by
looking at the split rather than at the model:

- `MurckoScaffoldSmiles` returns an empty string for *every* acyclic molecule, so treating that
  as a shared scaffold put methane, ethanol and hexadecane in one group — 49% of the boiling
  point set. It went wholly into training and left a test set that was systematically
  ring-containing: train mean 180 °C against test mean 252 °C, and a reported R² of 0.002 that
  was measuring the population gap, not the model. Each acyclic molecule now stands alone.
- MolCLR's own fill walks the groups against running cutoffs for train, then valid, then test,
  but train and valid fill at the same time, so the test branch can never be reached. On this
  data it returned an **empty test set**. Groups now go to whichever split is furthest below
  quota.
- Placing groups in shuffled order let a late, large group overshoot: one benzene scaffold
  covers a third of the pKa set. Largest groups are placed first.

All seven splits now come out at exactly 80/10/10 with zero scaffold overlap between train and
test.

MolCLR's own `finetune.py` is not used. It catches only `FileNotFoundError` when loading the
checkpoint and otherwise prints "Training from scratch", so a wrong path yields a randomly
initialised model that still reports a score; it z-scores targets only for two named QM
datasets, which boiling point in °C against an MSE loss needs just as much; and its split is
unseeded. Here a short checkpoint is a hard error, every regression target is standardised, and
the split is seeded.

The encoder itself is ported to plain PyTorch in `vendor/molclr/model.py` so torch-geometric
stays out of the serving image. That port was checked against the reference torch-geometric
implementation with the pretrained weights loaded — featurization identical atom for atom, and
encoder outputs matching to 0.0 across chirality, bond directions and a bondless ion.
`tests/test_molclr.py` pins those outputs so the port cannot drift.

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

## Deployment

Deployed to [Fly.io](https://fly.io) as a single machine in `iad`, configured in `fly.toml`.

```bash
flyctl deploy --remote-only
```

`--remote-only` builds on Fly's own builder, which is x86 — a local build on an Apple Silicon
Mac produces an arm64 image the machine cannot run.

Two things keep the first paint quick, and both matter more than they look. The app gzips its
own responses, because neither `StaticFiles` nor Fly's proxy does. And the Ketcher editor is
loaded as a separate chunk: it inlines its WASM, so it is ~29 MB, and with it in the initial
bundle the page showed nothing at all until every byte had parsed. The initial JS is now
~230 kB (72 kB gzipped).

The machine has 2 GB of memory and **suspends** when idle rather than stopping. Suspending
snapshots its memory, so a waking machine skips the ~8 second model load; a genuinely cold start
(first boot, or after a deploy) still pays it, which is what the health check's 60 second grace
period is for.

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

## License

Apache-2.0 (`LICENSE`) — chosen to match the only third-party source code redistributed here,
SolTranNet, so the repository is not a mix of licenses.

**One exception.** The two pKa models in `models/pka_*.pt` are fine-tuned on the IUPAC Digitized
pKa Dataset, which is CC BY-NC 4.0: *Reproduced by permission of the International Union of Pure
and Applied Chemistry*, © 2022 IUPAC. Those weights are a derivative work of non-commercial data
and may not be used commercially. Everything else here may. The dataset itself is downloaded at
training time and is not committed.

Everything else is a dependency, installed rather than redistributed, and keeps its own license:
RDKit (BSD-3), ADMET-AI (MIT), Ketcher (Apache-2.0). The vendored SolTranNet files keep their
own `LICENSE.txt` and are unmodified; upstream ships no `NOTICE` file, so there is none to
propagate.

The models' *training data* is a separate question from their code, and some datasets carry
terms of their own. Worth checking before any commercial use.
