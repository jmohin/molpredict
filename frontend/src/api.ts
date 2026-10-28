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

export async function fetchModels(): Promise<ModelInfo[]> {
  const r = await fetch('/api/models')
  if (!r.ok) throw new Error(`Failed to load models (${r.status})`)
  return r.json()
}

export async function predict(structure: string, signal?: AbortSignal): Promise<PredictResponse> {
  const r = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ structure }),
    signal,
  })
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new Error(body?.detail ?? `Prediction failed (${r.status})`)
  }
  return r.json()
}
