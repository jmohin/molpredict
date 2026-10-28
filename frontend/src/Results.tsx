import type { ModelInfo, PredictResponse, PropertyInfo } from './api'

type Value = {
  model: ModelInfo
  // This model's own metadata for the property, which carries its own benchmark score.
  prop: PropertyInfo
  value: number
  percentile: number | null
  warning: string | null
}
type Row = { prop: PropertyInfo; values: Value[] }

// How far two models must be apart before we call it a disagreement. Classification values are
// probabilities, so the gap is absolute; regression targets here are mostly log-scale, where one
// unit is an order of magnitude.
const DISAGREEMENT_THRESHOLD = { classification: 0.2, regression: 1.0 }

function disagreement(row: Row): number | null {
  if (row.values.length < 2) return null
  const values = row.values.map((v) => v.value)
  const spread = Math.max(...values) - Math.min(...values)
  const limit =
    row.prop.task_type === 'classification'
      ? DISAGREEMENT_THRESHOLD.classification
      : DISAGREEMENT_THRESHOLD.regression
  return spread > limit ? spread : null
}

function outOfRange(prop: PropertyInfo, value: number): string | null {
  if (prop.minimum != null && value < prop.minimum) return `below the possible minimum of ${prop.minimum}`
  if (prop.maximum != null && value > prop.maximum) return `above the possible maximum of ${prop.maximum}`
  return null
}

const CATEGORY_ORDER = ['Physicochemical', 'Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']

function formatValue(prop: PropertyInfo, value: number): string {
  if (prop.task_type === 'classification') return `${Math.round(value * 100)}%`
  if (prop.units === 'count') return String(Math.round(value))
  const abs = Math.abs(value)
  return abs >= 100 ? value.toFixed(0) : abs >= 10 ? value.toFixed(1) : value.toFixed(2)
}

function benchmarkLabel(prop: PropertyInfo): string | null {
  const b = prop.benchmark
  if (b.AUROC != null) return `AUROC ${b.AUROC.toFixed(2)}`
  if (b['R^2'] != null) return `R² ${b['R^2'].toFixed(2)}`
  if (b.MAE != null) return `MAE ${b.MAE.toFixed(2)}`
  return null
}

function buildRows(models: ModelInfo[], data: PredictResponse): Map<string, Row[]> {
  const byModel = new Map(models.map((m) => [m.id, m]))
  const rows = new Map<string, Row>()
  for (const { model_id, predictions } of data.results) {
    const model = byModel.get(model_id)
    if (!model) continue
    const props = new Map(model.properties.map((p) => [p.id, p]))
    for (const p of predictions) {
      const prop = props.get(p.property_id)
      if (!prop) continue
      const row = rows.get(prop.id) ?? { prop, values: [] }
      row.values.push({
        model,
        prop,
        value: p.value,
        percentile: p.drugbank_percentile,
        warning: p.warning,
      })
      rows.set(prop.id, row)
    }
  }
  const grouped = new Map<string, Row[]>()
  for (const row of rows.values()) {
    const list = grouped.get(row.prop.category) ?? []
    list.push(row)
    grouped.set(row.prop.category, list)
  }
  return new Map(
    [...grouped].sort(
      ([a], [b]) => (CATEGORY_ORDER.indexOf(a) + 99) % 99 - (CATEGORY_ORDER.indexOf(b) + 99) % 99,
    ),
  )
}

export function Results({ models, data }: { models: ModelInfo[]; data: PredictResponse }) {
  const grouped = buildRows(models, data)
  return (
    <div className="results">
      {[...grouped].map(([category, rows]) => (
        <section key={category}>
          <h3>{category}</h3>
          <table>
            <thead>
              <tr>
                <th>Property</th>
                <th className="num">Prediction</th>
                <th>vs. approved drugs</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {rows.flatMap((row) =>
                row.values.map((v, i) => (
                  <tr key={`${row.prop.id}-${v.model.id}`}>
                    {i === 0 && (
                      <td rowSpan={row.values.length}>
                        <div className="prop-name">
                          {row.prop.dataset_url ? (
                            <a href={row.prop.dataset_url} target="_blank" rel="noreferrer">
                              {row.prop.name}
                            </a>
                          ) : (
                            row.prop.name
                          )}
                        </div>
                        <div className="muted">
                          {row.prop.task_type === 'classification' ? 'probability' : row.prop.units}
                        </div>
                        {disagreement(row) != null && (
                          <div className="flag" title="The models below differ by more than this property's agreement threshold.">
                            models disagree by {disagreement(row)!.toFixed(2)}
                          </div>
                        )}
                      </td>
                    )}
                    <td className="num value">
                      {formatValue(row.prop, v.value)}
                      {outOfRange(row.prop, v.value) && (
                        <span className="flag" title={`This value is ${outOfRange(row.prop, v.value)}, so the model has extrapolated outside the physically possible range.`}>
                          out of range
                        </span>
                      )}
                    </td>
                    <td>
                      {v.percentile != null && (
                        <div className="pct" title={`${v.percentile.toFixed(0)}th percentile of DrugBank approved drugs`}>
                          <div className="pct-bar">
                            <div className="pct-fill" style={{ width: `${v.percentile}%` }} />
                          </div>
                          <span className="muted">{v.percentile.toFixed(0)}</span>
                        </div>
                      )}
                    </td>
                    <td>
                      <div>{v.model.name}</div>
                      {benchmarkLabel(v.prop) && <div className="muted">{benchmarkLabel(v.prop)}</div>}
                      {v.warning && <div className="flag" title={v.warning}>limited training data</div>}
                    </td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  )
}

export function ModelCards({ models }: { models: ModelInfo[] }) {
  return (
    <div className="models">
      {models.map((m) => (
        <article key={m.id}>
          <h4>{m.name}</h4>
          <p>{m.description}</p>
          <p className="muted">{m.citation}</p>
          <p className="links">
            <a href={m.paper_url} target="_blank" rel="noreferrer">Paper</a>
            <a href={m.code_url} target="_blank" rel="noreferrer">Code</a>
            <span className="muted">{m.license}</span>
          </p>
        </article>
      ))}
    </div>
  )
}
