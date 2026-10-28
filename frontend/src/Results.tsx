import type { ModelInfo, PredictResponse, PropertyInfo } from './api'

type Row = {
  prop: PropertyInfo
  values: { model: ModelInfo; value: number; percentile: number | null }[]
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
      row.values.push({ model, value: p.value, percentile: p.drugbank_percentile })
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
                      </td>
                    )}
                    <td className="num value">{formatValue(row.prop, v.value)}</td>
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
                      {benchmarkLabel(row.prop) && <div className="muted">{benchmarkLabel(row.prop)}</div>}
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
