/** The rules behind the warnings shown next to a prediction. */

import type { ModelInfo, PropertyInfo } from './api'

export type Value = {
  model: ModelInfo
  /** This model's own metadata for the property, which carries its own benchmark score. */
  prop: PropertyInfo
  value: number
  percentile: number | null
  warning: string | null
}

export type Row = { prop: PropertyInfo; values: Value[] }

/** Reading order for the result sections; anything unlisted sorts to the end, alphabetically. */
export const CATEGORY_ORDER = [
  'Physicochemical',
  'Absorption',
  'Distribution',
  'Metabolism',
  'Excretion',
  'Toxicity',
  // Not part of the ADMET sequence above; it answers a different question entirely.
  'Synthesis',
]

export function compareCategories(a: string, b: string): number {
  const ra = CATEGORY_ORDER.indexOf(a)
  const rb = CATEGORY_ORDER.indexOf(b)
  if (ra === -1 && rb === -1) return a.localeCompare(b)
  if (ra === -1) return 1
  if (rb === -1) return -1
  return ra - rb
}

// How far two models must be apart before we call it a disagreement. Classification values are
// probabilities, so the gap is absolute; regression targets here are mostly log-scale, where one
// unit is an order of magnitude.
export const DISAGREEMENT_THRESHOLD = { classification: 0.2, regression: 1.0 }

/** The spread between the most and least extreme prediction, when it is wide enough to report. */
export function disagreement(row: Row): number | null {
  if (row.values.length < 2) return null
  const values = row.values.map((v) => v.value)
  const spread = Math.max(...values) - Math.min(...values)
  const limit =
    row.prop.task_type === 'classification'
      ? DISAGREEMENT_THRESHOLD.classification
      : DISAGREEMENT_THRESHOLD.regression
  return spread > limit ? spread : null
}

/** Describes how a value breaks a physical bound, or null when it is within one. */
export function outOfRange(prop: PropertyInfo, value: number): string | null {
  if (prop.minimum != null && value < prop.minimum) return `below the possible minimum of ${prop.minimum}`
  if (prop.maximum != null && value > prop.maximum) return `above the possible maximum of ${prop.maximum}`
  return null
}

export function formatValue(prop: PropertyInfo, value: number): string {
  if (prop.task_type === 'classification') return `${Math.round(value * 100)}%`
  if (prop.units === 'count') return String(Math.round(value))
  const abs = Math.abs(value)
  return abs >= 100 ? value.toFixed(0) : abs >= 10 ? value.toFixed(1) : value.toFixed(2)
}

export function benchmarkLabel(prop: PropertyInfo): string | null {
  const b = prop.benchmark
  if (b.AUROC != null) return `AUROC ${b.AUROC.toFixed(2)}`
  if (b['R^2'] != null) return `R² ${b['R^2'].toFixed(2)}`
  if (b.MAE != null) return `MAE ${b.MAE.toFixed(2)}`
  return null
}
