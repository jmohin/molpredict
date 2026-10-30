export type PropertyInfo = {
  id: string
  name: string
  category: string
  units: string
  task_type: 'regression' | 'classification' | 'descriptor'
  description: string
  benchmark: Record<string, number>
  dataset_url: string
  minimum: number | null
  maximum: number | null
}

export type ModelInfo = {
  id: string
  name: string
  description: string
  citation: string
  paper_url: string
  code_url: string
  license: string
  properties: PropertyInfo[]
}

export type Prediction = {
  property_id: string
  value: number
  drugbank_percentile: number | null
  warning: string | null
}

export type PredictResponse = {
  smiles: string
  results: { model_id: string; predictions: Prediction[] }[]
}

export type ChargeSite = {
  atom_index: number
  kind: 'acid' | 'base'
  pka: number
}

export type ChargeResponse = {
  smiles: string
  ph: number[]
  charge: number[]
  sites: ChargeSite[]
  /** Sites whose pKa falls so far outside pH 0-14 that they cannot bend the curve. */
  sites_outside_window: number
  /** Charge of the neutral reference species: non-zero only for a permanent ion. */
  formal_charge: number
  charge_at_physiological_ph: number
  isoelectric_point: number | null
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const r = await fetch('/api/models')
  if (!r.ok) throw new Error(`Failed to load models (${r.status})`)
  return r.json()
}

async function post<T>(path: string, structure: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ structure }),
    signal,
  })
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${r.status})`)
  }
  return r.json()
}

export function predict(structure: string, signal?: AbortSignal): Promise<PredictResponse> {
  return post<PredictResponse>('/api/predict', structure, signal)
}

export function fetchCharge(structure: string, signal?: AbortSignal): Promise<ChargeResponse> {
  return post<ChargeResponse>('/api/charge', structure, signal)
}
