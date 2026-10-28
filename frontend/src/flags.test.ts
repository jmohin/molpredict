import { describe, expect, it } from 'vitest'

import type { ModelInfo, PropertyInfo } from './api'
import {
  benchmarkLabel,
  compareCategories,
  disagreement,
  formatValue,
  outOfRange,
  type Row,
} from './flags'

const model = { id: 'm', name: 'M' } as ModelInfo

function property(over: Partial<PropertyInfo> = {}): PropertyInfo {
  return {
    id: 'p',
    name: 'P',
    category: 'Absorption',
    units: 'log(mol/L)',
    task_type: 'regression',
    description: '',
    benchmark: {},
    dataset_url: '',
    minimum: null,
    maximum: null,
    ...over,
  }
}

function row(prop: PropertyInfo, ...values: number[]): Row {
  return {
    prop,
    values: values.map((value) => ({ model, prop, value, percentile: null, warning: null })),
  }
}

describe('disagreement', () => {
  it('stays silent for a single model', () => {
    expect(disagreement(row(property(), -1.6))).toBeNull()
  })

  it('stays silent when regression models are close', () => {
    expect(disagreement(row(property(), -1.62, -1.86))).toBeNull()
  })

  it('reports the spread when regression models are far apart', () => {
    expect(disagreement(row(property(), -3.81, -1.35))).toBeCloseTo(2.46, 2)
  })

  it('uses a tighter threshold for probabilities', () => {
    const p = property({ task_type: 'classification' })
    expect(disagreement(row(p, 0.5, 0.65))).toBeNull()
    expect(disagreement(row(p, 0.1, 0.9))).toBeCloseTo(0.8, 2)
  })

  it('measures the full spread across more than two models', () => {
    expect(disagreement(row(property(), -1.0, -4.5, -2.0))).toBeCloseTo(3.5, 2)
  })

  it('does not fire exactly at the threshold', () => {
    expect(disagreement(row(property(), 0, 1.0))).toBeNull()
  })
})

describe('outOfRange', () => {
  const halfLife = property({ units: 'hr', minimum: 0, maximum: null })

  it('accepts a value inside the bounds', () => {
    expect(outOfRange(halfLife, 4.2)).toBeNull()
  })

  it('flags a negative half-life', () => {
    expect(outOfRange(halfLife, -19.7)).toContain('below')
  })

  it('flags a percentage over 100', () => {
    expect(outOfRange(property({ minimum: 0, maximum: 100 }), 101)).toContain('above')
  })

  it('ignores properties with no declared bounds', () => {
    expect(outOfRange(property(), -999)).toBeNull()
  })

  it('treats the bound itself as in range', () => {
    expect(outOfRange(halfLife, 0)).toBeNull()
  })
})

describe('formatValue', () => {
  it('renders probabilities as percentages', () => {
    expect(formatValue(property({ task_type: 'classification' }), 0.658)).toBe('66%')
  })

  it('renders counts as integers', () => {
    expect(formatValue(property({ units: 'count' }), 3.0)).toBe('3')
  })

  it('keeps two decimals for small magnitudes and drops them for large', () => {
    expect(formatValue(property(), -1.624)).toBe('-1.62')
    expect(formatValue(property(), 63.297)).toBe('63.3')
    expect(formatValue(property(), 180.159)).toBe('180')
  })
})

describe('benchmarkLabel', () => {
  it('prefers AUROC, then R², then MAE', () => {
    expect(benchmarkLabel(property({ benchmark: { AUROC: 0.99, 'R^2': 0.5 } }))).toBe('AUROC 0.99')
    expect(benchmarkLabel(property({ benchmark: { 'R^2': 0.82 } }))).toBe('R² 0.82')
    expect(benchmarkLabel(property({ benchmark: { MAE: 1.2 } }))).toBe('MAE 1.20')
  })

  it('reports a negative R², which marks a model worse than the mean', () => {
    expect(benchmarkLabel(property({ benchmark: { 'R^2': -2.39 } }))).toBe('R² -2.39')
  })

  it('returns nothing when a model reports no score', () => {
    expect(benchmarkLabel(property())).toBeNull()
  })
})

describe('compareCategories', () => {
  it('orders known categories by the reading order', () => {
    expect(['Toxicity', 'Physicochemical', 'Absorption'].sort(compareCategories)).toEqual([
      'Physicochemical',
      'Absorption',
      'Toxicity',
    ])
  })

  it('puts unknown categories last', () => {
    expect(['Quantum', 'Absorption'].sort(compareCategories)).toEqual(['Absorption', 'Quantum'])
  })
})
