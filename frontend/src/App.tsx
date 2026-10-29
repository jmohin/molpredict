import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import type { Ketcher } from 'ketcher-core'
import { fetchModels, predict, type ModelInfo, type PredictResponse } from './api'
import { ModelCards, Results } from './Results'

// Ketcher inlines its WASM, so it is ~29 MB — the whole page used to stay blank until all of it
// had parsed. Loading it separately lets the rest of the app (SMILES box, examples, results)
// work while the editor is still arriving.
const MoleculeEditor = lazy(() =>
  import('./MoleculeEditor').then((m) => ({ default: m.MoleculeEditor })),
)

const EXAMPLES: [string, string][] = [
  ['Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'],
  ['Caffeine', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'],
  ['Imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'],
  ['Atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O'],
]

export default function App() {
  const ketcherRef = useRef<Ketcher | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [smilesInput, setSmilesInput] = useState('')
  const [data, setData] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const debounce = useRef<number | undefined>(undefined)
  const inflight = useRef<AbortController | null>(null)

  useEffect(() => {
    fetchModels().then(setModels, (e) => setError(String(e.message ?? e)))
  }, [])

  const run = useCallback(async (structure: string) => {
    inflight.current?.abort()
    if (!structure.trim()) {
      setData(null)
      setError(null)
      return
    }
    const ctrl = new AbortController()
    inflight.current = ctrl
    setLoading(true)
    try {
      const res = await predict(structure, ctrl.signal)
      setData(res)
      setError(null)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      if (inflight.current === ctrl) setLoading(false)
    }
  }, [])

  const onEditorChange = useCallback(() => {
    window.clearTimeout(debounce.current)
    debounce.current = window.setTimeout(async () => {
      const ketcher = ketcherRef.current
      if (!ketcher) return
      try {
        const smiles = await ketcher.getSmiles()
        setSmilesInput(smiles)
        run(smiles)
      } catch {
        // Partially drawn structures (e.g. a dangling query atom) can't be exported yet.
      }
    }, 400)
  }, [run])

  const loadSmiles = useCallback(async (smiles: string) => {
    setSmilesInput(smiles)
    await ketcherRef.current?.setMolecule(smiles)
    run(smiles)
  }, [run])

  return (
    <div className="page">
      <header>
        <h1>molpredict</h1>
        <p className="muted">Draw a molecule. See what published AI models predict about it.</p>
      </header>

      <div className="layout">
        <div className="left">
          <div className="editor">
            <Suspense fallback={<div className="editor-loading muted">Loading the editor…</div>}>
              <MoleculeEditor onReady={(k) => (ketcherRef.current = k)} onChange={onEditorChange} />
            </Suspense>
          </div>
          <form
            className="smiles"
            onSubmit={(e) => {
              e.preventDefault()
              loadSmiles(smilesInput)
            }}
          >
            <input
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              placeholder="…or paste a SMILES string"
              spellCheck={false}
              aria-label="SMILES"
            />
            <button type="submit">Load</button>
          </form>
          <div className="examples">
            <span className="muted">Try:</span>
            {EXAMPLES.map(([name, smi]) => (
              <button key={name} className="chip" onClick={() => loadSmiles(smi)}>
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="right">
          <div className="status">
            {error ? (
              <span className="error">{error}</span>
            ) : data ? (
              <code>{data.smiles}</code>
            ) : (
              <span className="muted">Draw a structure or pick an example to see predictions.</span>
            )}
            {loading && <span className="muted"> · predicting…</span>}
          </div>
          {data && models.length > 0 && <Results models={models} data={data} />}
        </div>
      </div>

      <footer>
        <h2>Models</h2>
        <ModelCards models={models} />
        <p className="muted disclaimer">
          Predictions are for research use only. Classification values are model probabilities, not
          measured outcomes. Percentiles compare against DrugBank approved drugs as scored by the same model.
        </p>
      </footer>
    </div>
  )
}
